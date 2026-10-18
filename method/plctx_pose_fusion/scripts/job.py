"""Install/manage an isolated persistent fusion job. No other jobs are touched."""
from __future__ import annotations
import argparse,fcntl,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO=Path(os.environ.get('SIGNREP_WORKSPACE','/home/haipd/Sign_Representations')).resolve()
sys.path.insert(0,str(ROOT))
from fp.utils import sha256,atomic_json

SERVICE='plctx-pose-fusion-full.service'
DEFAULT_JOB=ROOT/'runtime/full160'
DEFAULT_PY=REPO/'.venv-signrep/bin/python'


def gpu_free():
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if result.returncode:raise RuntimeError('nvidia-smi failed: '+result.stderr[:200])
    return int(result.stdout.splitlines()[0])


def ready(free,required=10240):return free>=required


def hashes(job):
    return {str(p.relative_to(job)):sha256(p) for directory in (job/'code/fp',job/'ops')
            for p in sorted(directory.rglob('*.py'))}|{'config.json':sha256(job/'config.json')}


def setup(job,config,python):
    if job.exists():raise FileExistsError('Job already exists; use status/resume')
    job.mkdir(parents=True)
    shutil.copytree(ROOT/'fp',job/'code/fp',ignore=shutil.ignore_patterns('__pycache__'))
    (job/'ops').mkdir();shutil.copy2(Path(__file__),job/'ops/job.py')
    shutil.copy2(config,job/'config.json')
    atomic_json(job/'lock.json',{'sha256':hashes(job),'python':str(python.absolute()),'job':str(job.resolve()),
              'memory_threshold_mib':10240,'disk_min_free_gib':8,'created_unix':time.time(),
              'source_root':str(ROOT),'service':SERVICE})


def install(job):
    lock=json.loads((job/'lock.json').read_text())
    # Linger has been checked; service is independent of SSH/tmux lifetime.
    folder=Path.home()/'.config/systemd/user';folder.mkdir(parents=True,exist_ok=True)
    path=folder/SERVICE
    if path.exists():raise FileExistsError(path)
    body=f'''[Unit]
Description=PLctx + body/relations fusion full ASL Citizen160epochs
After=default.target
[Service]
Type=simple
WorkingDirectory={REPO}
Environment=PYTHONPATH={job}/code
Environment=PYTHONUNBUFFERED=1
Environment=OMP_NUM_THREADS=4
Environment=OPENBLAS_NUM_THREADS=4
Environment=MKL_NUM_THREADS=4
ExecStart={lock['python']} {job}/ops/job.py supervise --job {job}
Restart=on-abnormal
RestartSec=30
KillMode=mixed
TimeoutStopSec=300
Nice=5
CPUQuota=800%
MemoryHigh=24G
MemoryMax=32G
StandardOutput=append:{job}/service.log
StandardError=append:{job}/service.log
[Install]
WantedBy=default.target
'''
    path.write_text(body)
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','enable','--now',SERVICE],check=True)


def supervise(job):
    lock=json.loads((job/'lock.json').read_text())
    if hashes(job)!=lock['sha256']:raise ValueError('Frozen job code/config changed')
    handle=open(job/'supervisor.lock','a+')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another supervisor owns this job')
    stop={'value':False};child=None
    def halt(*_):
        stop['value']=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,halt)
    def status(phase,**kw):atomic_json(job/'supervisor.json',{'status':phase,'pid':os.getpid(),'updated_unix':time.time(),**kw})
    env=os.environ.copy();env.update(PYTHONPATH=str(job/'code'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    retry_after=0
    while not stop['value']:
        if (job/'pause.request').exists():status('paused');return 0
        if hashes(job)!=lock['sha256']:raise ValueError('Frozen job changed during execution')
        if (job/'evaluation/complete.json').exists():status('complete');return 0
        try:free=gpu_free()
        except Exception as exc:
            status('waiting_for_gpu',gpu_probe_error=str(exc));time.sleep(15);continue
        if not ready(free,lock['memory_threshold_mib']) or time.time()<retry_after:
            status('waiting_for_gpu',free_vram_mib=free,required_mib=lock['memory_threshold_mib']);time.sleep(15);continue
        if shutil.disk_usage(job).free<lock['disk_min_free_gib']*2**30:
            status('waiting_for_disk',free_disk_gib=shutil.disk_usage(job).free/2**30);time.sleep(30);continue
        train_done=(job/'train/complete.json').exists()
        module='fp.final_eval' if train_done else 'fp.train'
        command=[lock['python'],'-u','-m',module,'--config',str(job/'config.json')]
        command+=['--job',str(job)] if train_done else ['--root',str(job/'train')]
        logfile=job/('evaluation.log' if train_done else 'train.log')
        with logfile.open('a') as f:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
            status('evaluating' if train_done else 'training',child_pid=child.pid,command=command)
            code=child.wait()
        child=None
        if stop['value']:status('stopped');return 0
        if code==0:continue
        if code==75:
            status('paused');return 0
        if code==76 or code in (-9,137):
            # CUDA OOM or killed resource process: restore committed last checkpoint.
            retry_after=time.time()+60;status('resource_requeue',exit_code=code);continue
        status('failed',exit_code=code,log=str(logfile),reason='Nonresource error; no automatic restart');return 1
    status('stopped');return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','install','supervise','status','stop','resume'])
    p.add_argument('--job',type=Path,default=DEFAULT_JOB);p.add_argument('--config',type=Path,default=ROOT/'configs/full160.json')
    p.add_argument('--python',type=Path,default=DEFAULT_PY);a=p.parse_args();job=a.job.resolve()
    if a.action=='setup':setup(job,a.config,a.python)
    elif a.action=='install':install(job)
    elif a.action=='supervise':raise SystemExit(supervise(job))
    elif a.action=='status':
        result=subprocess.run(['systemctl','--user','show',SERVICE,'-p','ActiveState','-p','SubState','-p','MainPID'],capture_output=True,text=True)
        print(result.stdout)
        for name in ('supervisor.json','progress.json'):
            if (job/name).exists():print(name,(job/name).read_text())
    elif a.action=='stop':
        (job/'pause.request').touch();subprocess.run(['systemctl','--user','stop',SERVICE],check=True)
    elif a.action=='resume':
        if hashes(job)!=json.loads((job/'lock.json').read_text())['sha256']:raise ValueError('Frozen job changed')
        (job/'pause.request').unlink(missing_ok=True)
        subprocess.run(['systemctl','--user','reset-failed',SERVICE],check=False)
        subprocess.run(['systemctl','--user','start',SERVICE],check=True)


if __name__=='__main__':main()
