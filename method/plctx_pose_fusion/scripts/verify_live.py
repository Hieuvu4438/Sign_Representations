"""Read-only check of the installed job, actual service/PIDs and checkpoint."""
from pathlib import Path
import argparse,importlib.util,json,os,subprocess,sys,time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fp.utils import sha256,atomic_json


def require(value,message):
    if not value:raise AssertionError(message)


def command(pid):
    require(pid>0,'Missing live PID')
    path=Path(f'/proc/{pid}')
    cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    require(path.stat().st_uid==os.getuid(),'Wrong process owner')
    require((path/'stat').read_text().split(') ',1)[1][0]!='Z','Zombie process')
    return cmd


def main(job):
    job=Path(job).resolve()
    spec=importlib.util.spec_from_file_location('fusion_job',ROOT/'scripts/job.py')
    ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
    lock=json.loads((job/'lock.json').read_text())
    require(ops.hashes(job)==lock['sha256'],'Installed frozen code/config changed')
    result=subprocess.run(['systemctl','--user','show',ops.SERVICE,'-p','ActiveState','-p','SubState','-p','MainPID','-p','UnitFileState'],check=True,capture_output=True,text=True)
    state=dict(line.split('=',1) for line in result.stdout.splitlines())
    require(state['ActiveState']=='active' and state['SubState']=='running' and state['UnitFileState']=='enabled','Service not live and enabled')
    parent_cmd=command(int(state['MainPID']))
    require(lock['python'] in parent_cmd and str(job/'ops/job.py') in parent_cmd,'Wrong supervisor executable')
    supervisor=json.loads((job/'supervisor.json').read_text())
    require(supervisor['pid']==int(state['MainPID']),'Stale supervisor status')
    require(not (job/'pause.request').exists(),'Job is paused')
    linger=subprocess.run(['loginctl','show-user',str(os.getuid()),'-p','Linger'],check=True,capture_output=True,text=True)
    require('Linger=yes' in linger.stdout,'No logout/reboot persistence')
    evidence={'status':'PASS','service':state,'supervisor':supervisor,'parent_cmd':parent_cmd,
              'frozen_snapshot_intact':True,'linger':True,'checked_unix':time.time()}
    if supervisor['status'] in ('training','evaluating'):
        child_cmd=command(supervisor['child_pid'])
        require(lock['python'] in child_cmd and str(job/'config.json') in child_cmd,'Wrong live child command')
        evidence['child_cmd']=child_cmd
    else:require(supervisor['status'] in ('waiting_for_gpu','resource_requeue','waiting_for_disk'),'Unexpected job state')
    p=job/'train/last.pt'
    require(p.exists(),'No initial resumable checkpoint')
    # Load directly on CPU; no GPU interaction with the live workload.
    ck=torch.load(p,map_location='cpu',weights_only=False)
    scientific={str(f.relative_to(job/'code')):sha256(f) for f in sorted((job/'code/fp').rglob('*.py'))}
    reviewed=json.loads((ROOT/'runtime/verification.json').read_text())
    require(reviewed['status']=='PASS' and ck['proof']['source_sha256']==scientific==reviewed['proof']['source_sha256'],'Checkpoint/snapshot/review source mismatch')
    require(ck['config']==json.loads((job/'config.json').read_text()),'Wrong checkpoint config')
    require(ck['config']['steps']==200640 and not ck['config']['smoke'],'Not full protocol')
    require(all(k in ck for k in ('optimizer','rng','center','aux','progress','model')),'Resume state incomplete')
    progress=json.loads((job/'progress.json').read_text())
    evidence.update(checkpoint_step=ck['step'],pending_validation=ck['progress']['pending_validation'],
                    validated=ck['progress']['validated'],progress=progress,scientific_source_sha256=scientific,
                    setup_review_status='PASS',evaluation_chained='fp.final_eval after train/complete.json',
                    caveat='Training setup verified; full160 training and final test may still be running')
    if progress.get('phase')=='validation':
        require(ck['progress']['pending_validation'] is not None,'Validation has no transactional resume marker')
    atomic_json(ROOT/'runtime/live_verification.json',evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('scientific_source_sha256','supervisor')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,default=ROOT/'runtime/full160');a=p.parse_args();main(a.job)
