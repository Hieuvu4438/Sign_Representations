"""Sequential PLctx160 validation/LR continuation/test job, detached and resumable."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PACKAGE=Path(__file__).resolve().parents[1]
UNIT='plctx160-continuation.service'


def bootstrap(root):
    job=json.loads((root/'job.json').read_text())
    sys.path.insert(0,str(root/'code'))
    import pj.paths as paths
    workspace=Path(job['workspace'])
    paths.REPO=workspace;paths.RUNTIME=root/'assets';paths.OBS_JEPA=root/'code'
    paths.ASLC002_RUN=workspace/'method/jepa_baseline/runtime/ASLC002'
    paths.OBS_META=workspace/'method/obs_jepa/runtime/meta_aslc002.json'
    import oj.paths as op
    op.REPO=workspace;op.ROOT=workspace/'method/obs_jepa';op.RUNTIME=op.ROOT/'runtime'
    op.ASLC002_RUN=paths.ASLC002_RUN
    return job


def require(value,message):
    if not value:raise ValueError(message)


def setup(args):
    sys.path.insert(0,str(PACKAGE))
    from pj.continuation_support import atomic_json,sha256
    import numpy as np
    import torch
    root=Path(args.job_root).resolve();workspace=PACKAGE.parents[1]
    unit_path=Path.home()/'.config/systemd/user'/UNIT
    require(not root.exists() and not unit_path.exists(),'Continuation job/service exists; use resume/status')
    require(shutil.disk_usage(root.parent).free>8*2**30,'Need8GiB disk headroom')
    checkpoint=PACKAGE/'runtime/runs/PJ_PLctx_FULL_160ep/final.pt'
    cache=PACKAGE/'runtime/cache/aslc_train_full.npz'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    require(state['step']==200640 and state['config']['target']=='ctx' and state['config']['loss']=='cce','Wrong base checkpoint')
    require(not state['config']['glob'] and not state['config']['curriculum'],'Unexpected base recipe')
    del state
    dataset=Path('/home/dongvk/datasets/ASL_Citizen');splits={}
    raw={s:list(csv.DictReader((dataset/'splits'/f'{s}.csv').open())) for s in ['train','val','test']}
    require({s:len(v) for s,v in raw.items()}=={'train':40154,'val':10304,'test':32941},'Official split membership changed')
    labels={label:i for i,label in enumerate(sorted({row['Gloss'] for row in raw['train']}))}
    for split,records in raw.items():
        splits[split]=[{'sample_id':Path(row['Video file']).stem,'video':str(dataset/'videos'/row['Video file']),
                       'gloss':row['Gloss'],'label':labels[row['Gloss']],'signer':row['Participant ID'],
                       'split':split,'role':'gallery' if split=='train' else 'query'} for row in records]
        require(len({r['sample_id'] for r in splits[split]})==len(records),'Duplicate source IDs')
    for a,b in [('train','val'),('train','test'),('val','test')]:
        require(not {r['sample_id'] for r in splits[a]}&{r['sample_id'] for r in splits[b]},'Split ID overlap')
        require(not {r['signer'] for r in splits[a]}&{r['signer'] for r in splits[b]},'Signer split overlap')
    with np.load(cache,allow_pickle=False) as data:
        require(data['ids'].tolist()==[r['sample_id'] for r in splits['train']],'Full train cache IDs changed')
    base_manifest=json.loads(cache.with_suffix('.manifest.json').read_text())
    require(sha256(cache)==base_manifest['cache_sha256'],'Original full train cache changed')
    root.mkdir(parents=True);(root/'assets').mkdir();(root/'ops').mkdir()
    shutil.copytree(PACKAGE/'pj',root/'code/pj',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(workspace/'method/obs_jepa/oj',root/'code/oj',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(PACKAGE/'runtime/target_stats.npz',root/'assets/target_stats.npz')
    shutil.copyfile(Path(__file__),root/'ops/plctx_continue_job.py')
    import zipfile
    memmap=root/'assets/train_arrays';memmap.mkdir()
    with zipfile.ZipFile(cache) as packed:
        for key in ['xy','conf','offsets','fps','ids']:
            with packed.open(key+'.npy') as source,(memmap/(key+'.npy')).open('wb') as destination:
                shutil.copyfileobj(source,destination,1024*1024)
    # Availability/identity only. Test never goes into training or branch selection.
    asset_inventory=[]
    for records in splits.values():
        for row in records:
            for path in [Path(row['video']),dataset/'keypoints'/f"{row['sample_id']}.pkl"]:
                st=path.stat();asset_inventory.append({'path':str(path),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns})
    atomic_json(root/'assets/source_inventory.json',asset_inventory)
    baseline=PACKAGE/'runtime/eval/PJ_PLctx_FULL_160ep_dictionary/aslc'
    job={'workspace':str(workspace),'python':str(Path(sys.executable).absolute()),'base_checkpoint':str(checkpoint),
         'base_checkpoint_sha256':sha256(checkpoint),'training_cache':str(cache),'training_cache_sha256':sha256(cache),
         'memmap_cache':str(memmap),
         'video_metadata':str(cache.with_suffix('.meta.json')),'dataset_root':str(dataset),'baseline_dictionary':str(baseline),
         'manifests':splits,'sampler_seed':160200640,'train_workers':4,'evaluation_workers':4,'window_batch':32,
         'cuda_memory_fraction':.16,'validation_interval':5000,'patience':5,
         'arms':[{'name':'lr1e5','peak_lr':1e-5,'additional_steps':50160},
                 {'name':'lr3e5','peak_lr':3e-5,'additional_steps':50160}],
         'environment':{'PYTHONPATH':str(root/'code'),'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4',
                        'OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUDA_VISIBLE_DEVICES':'0'},
         'protocol_note':'Native PLctx raw/_grp/window64/stride2. Full official train gallery/val queries for selection; original zero-vector handling retained. Only validation winner is evaluated on test.',
         'legacy_checkpoint_resume_note':'Native base checkpoint lacks sampler/NumPy/CUDA state; continuation starts new deterministic sampler at epoch160 shared by both arms. All post-fork interruptions restore exact committed state.'}
    atomic_json(root/'job.json',job)
    pins={str(p):sha256(p) for p in [checkpoint,cache,Path(job['video_metadata']),root/'assets/target_stats.npz',
          *[dataset/'splits'/f'{s}.csv' for s in splits],baseline/'train.jsonl',baseline/'test.jsonl',baseline/'protocol.json',baseline/'results.json',baseline/'_grp_ranks.npy']}
    for p in sorted((baseline/'train').glob('*.npz')):pins[str(p)]=sha256(p)
    for path in memmap.glob('*.npy'):pins[str(path)]=sha256(path)
    files={str(p.relative_to(root)):sha256(p) for parent in ['code','ops'] for p in sorted((root/parent).rglob('*.py'))}
    for p in [root/'job.json',root/'assets/source_inventory.json']:files[str(p.relative_to(root))]=sha256(p)
    atomic_json(root/'job_lock.json',{'files':files,'inputs':pins})
    unit=f'''[Unit]
Description=PLctx160 sequential continuation LR experiments and validation-selected test
StartLimitIntervalSec=3600
StartLimitBurst=5

[Service]
Type=simple
WorkingDirectory={workspace}
ExecStart={job['python']} {root}/ops/plctx_continue_job.py supervise --job-root {root}
Restart=on-failure
RestartSec=30
KillMode=mixed
TimeoutStopSec=300
Nice=5
CPUQuota=800%
MemoryHigh=18G
MemoryMax=24G
Environment=PYTHONPATH={root}/code
Environment=PYTHONUNBUFFERED=1
StandardOutput=append:{root}/service.log
StandardError=append:{root}/service.log

[Install]
WantedBy=default.target
'''
    unit_path.parent.mkdir(parents=True,exist_ok=True);unit_path.write_text(unit)
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','enable',UNIT],check=True)
    return {'service':UNIT,'job_root':str(root),'arms':job['arms'],'source_step':200640,'maximum_total_step':250800}


def verify(root,job):
    from pj.continuation_support import sha256
    lock=json.loads((root/'job_lock.json').read_text())
    for name,digest in lock['files'].items():require(sha256(root/name)==digest,f'Pinned job source/config changed: {name}')
    for name,digest in lock['inputs'].items():require(sha256(name)==digest,f'Pinned original input changed: {name}')
    for row in json.loads((root/'assets/source_inventory.json').read_text()):
        st=Path(row['path']).stat()
        require(st.st_size==row['bytes'] and st.st_mtime_ns==row['mtime_ns'],f'Source asset changed: {row["path"]}')


def write_progress(root,phase,values):
    from pj.continuation_support import atomic_json
    atomic_json(root/'progress.json',{'phase':phase,'status':'running','pid':os.getpid(),'updated_unix':time.time(),**values})
    (root/'progress.txt').write_text(f'PLctx160 continuation | {phase}\n'+json.dumps(values,ensure_ascii=False)+'\n')


def evaluate_stage(root,job,mode):
    import torch
    import numpy as np
    from pj.continuation_support import atomic_json,sha256
    from pj.continuation_eval import evaluate_checkpoint
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(job['cuda_memory_fraction'])
    stop={'value':False}
    def halt(number,frame):stop['value']=True
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,halt)
    stopped=lambda:stop['value'] or (root/'pause.request').exists()
    try:
        if mode=='baseline':
            return evaluate_checkpoint(job['base_checkpoint'],job,root/'baseline/validation',stopped=stopped,
                progress=lambda values:write_progress(root,'baseline_validation',values),reuse_baseline_gallery=True)
        selection=json.loads((root/'selection.json').read_text());checkpoint=selection['checkpoint']
        require(sha256(checkpoint)==selection['checkpoint_sha256'],'Validation-selected checkpoint changed')
        if selection['selected']=='baseline':
            previous=Path(job['baseline_dictionary']);old=json.loads((previous/'results.json').read_text())['readouts']['_grp']
            ranks=np.load(previous/'_grp_ranks.npy');y=np.array([r['label'] for r in job['manifests']['test']])
            metrics={'DCG':old['DCG'],'MRR':float(np.mean(1/ranks)),**{f'Rec@{k}':float(np.mean(ranks<=k)) for k in [1,5,10]},
                     'macro_Rec@1':float(np.mean([np.mean(ranks[y==c]<=1) for c in np.unique(y)])),
                     'queries':len(y),'gallery':40154}
            target=root/'final_test';target.mkdir(exist_ok=True);np.save(target/'ranks.npy',ranks)
            result={'status':'complete','checkpoint_sha256':selection['checkpoint_sha256'],'split':'test','metrics':metrics,
                    'reused_original_official_test':True,'finished_unix':time.time()}
            atomic_json(target/'metrics.json',result);return result
        return evaluate_checkpoint(checkpoint,job,root/'final_test',split='test',stopped=stopped,
            progress=lambda values:write_progress(root,'final_official_test',values))
    except InterruptedError:return {'status':'paused'}


def select_and_report(root,job,final=False):
    from pj.continuation_support import atomic_json,sha256
    candidates=[{'name':'baseline','score':json.loads((root/'baseline/validation/metrics.json').read_text())['metrics']['Rec@1'],
                 'step':200640,'checkpoint':job['base_checkpoint']}]
    for arm in job['arms']:
        folder=root/'arms'/arm['name'];done=json.loads((folder/'complete.json').read_text())
        candidates.append({'name':arm['name'],'score':done['best_validation_Rec1'],'step':done['best_global_step'],'checkpoint':str(folder/'best.pt')})
    winner=max(candidates,key=lambda x:x['score'])
    selection={'selected':winner['name'],'checkpoint':winner['checkpoint'],'checkpoint_sha256':sha256(winner['checkpoint']),
               'global_step':winner['step'],'validation_Rec1':winner['score'],'candidates':candidates,
               'criterion':'highest raw full-validation Rec@1; ties retain baseline/earlier candidate','test_used_for_selection':False}
    if not final:
        atomic_json(root/'selection.json',selection);return selection
    frozen=json.loads((root/'selection.json').read_text())
    require(frozen==selection,'Selection changed after test')
    test=json.loads((root/'final_test/metrics.json').read_text())
    history=[]
    baseline=json.loads((root/'baseline/validation/metrics.json').read_text())
    history.append({'arm':'baseline','branch_step':0,'global_step':200640,**baseline['metrics']})
    for arm in job['arms']:
        for path in sorted((root/'arms'/arm['name']/'validation').glob('*/metrics.json')):
            j=json.loads(path.read_text());n=int(path.parent.name)
            history.append({'arm':arm['name'],'branch_step':n,'global_step':200640+n,**j['metrics']})
    original=json.loads((Path(job['baseline_dictionary'])/'results.json').read_text())['readouts']['_grp']
    report={'status':'complete','selection':selection,'validation_history':history,'official_test':test,
            'baseline_official_test':original,'protocol':job['protocol_note'],'continuation_notes':job['legacy_checkpoint_resume_note'],
            'limitations':'One seed; LR arms selected on validation. Test was viewed for original baseline before designing extension; this is exploratory improvement work, not a fresh untouched benchmark.',
            'finished_unix':time.time()}
    atomic_json(root/'report.json',report)
    with (root/'validation_iterations.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    lines=['# PLctx160 continuation — kết quả','',
           'Hai nhánh độc lập từ200.640updates; LRpeak1e-5/3e-5; tối đa50.160updates/nhánh.','',
           '| Nhánh | Best global step | ValidationRec@1 (%) |','|---|---:|---:|']
    for candidate in candidates:lines.append(f"| {candidate['name']} | {candidate['step']} | {100*candidate['score']:.2f} |")
    lines+=['',f"Chọn **{selection['selected']}** bằng validation; test không dùng chọn nhánh/checkpoint.",'',
            '| Metric | PLctx160 baseline test | Validation-selected test |','|---|---:|---:|']
    for label,old,new in [('DCG',original['DCG'],test['metrics']['DCG']),('Rec@1 (%)',100*original['Rec1'],100*test['metrics']['Rec@1']),
                          ('Rec@5 (%)',100*original['Rec5'],100*test['metrics']['Rec@5'])]:
        lines.append(f'| {label} | {old:.4f} | {new:.4f} |')
    lines+=['','Xem `validation_iterations.csv`, `report.json`, `final_test/metrics.json` cho đầy đủ DCG/MRR/Rec@1/5/10/macroRec@1.',
            '',report['limitations']]
    (root/'REPORT_VI.md').write_text('\n'.join(lines)+'\n');atomic_json(root/'complete.json',{'status':'complete','report':str(root/'report.json'),'finished_unix':time.time()})
    write_progress(root,'complete',{'status':'complete','selected':selection['selected'],'report':str(root/'REPORT_VI.md')})
    return report


def wait_gpu(root,job,stopped,cooldown=0):
    from pj.continuation_support import atomic_json
    started=time.monotonic()
    while not stopped():
        try:free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','--id=0'],text=True,timeout=15).strip())
        except (OSError,ValueError,subprocess.SubprocessError):free=0
        if free>=6144 and time.monotonic()-started>=cooldown:return True
        write_progress(root,'waiting_for_gpu',{'free_vram_MiB':free,'required_MiB':6144,'status':'waiting_for_gpu'})
        for _ in range(30):
            if stopped():return False
            time.sleep(1)
    return False


def supervise(root,job):
    from pj.continuation_support import atomic_json
    lock=(root/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop={'value':False};child=None
    def halt(number,frame):
        stop['value']=True
        if child and child.poll() is None:child.terminate()
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,halt)
    stopped=lambda:stop['value'] or (root/'pause.request').exists()
    env=os.environ.copy();env.update(job['environment']);script=str(root/'ops/plctx_continue_job.py')
    stages=[('baseline',[job['python'],script,'evaluate','--mode','baseline','--job-root',str(root)],root/'baseline/validation/metrics.json')]
    for arm in job['arms']:stages.append((arm['name'],[job['python'],script,'train','--arm',arm['name'],'--job-root',str(root)],root/'arms'/arm['name']/'complete.json'))
    stages.extend([('select',[job['python'],script,'select','--job-root',str(root)],root/'selection.json'),
                   ('test',[job['python'],script,'evaluate','--mode','test','--job-root',str(root)],root/'final_test/metrics.json'),
                   ('report',[job['python'],script,'report','--job-root',str(root)],root/'complete.json')])
    for phase,command,marker in stages:
        if stopped():return 0
        if marker.exists():continue
        verify(root,job);cooldown=0
        while True:
            if phase not in ['select','report'] and not wait_gpu(root,job,stopped,cooldown):return 0
            logs=root/'logs';logs.mkdir(exist_ok=True);path=logs/f'{phase}.log'
            with path.open('a') as log:
                log.write(f'\n--- attempt {time.ctime()} ---\n');log.flush();offset=path.stat().st_size
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                atomic_json(root/'supervisor.json',{'status':'running','phase':phase,'pid':os.getpid(),'child_pid':child.pid,'started_unix':time.time()})
                if stopped():child.terminate()
                code=child.wait()
            if stopped():atomic_json(root/'supervisor.json',{'status':'paused','phase':phase});return 0
            if code:
                with path.open('rb') as log:
                    log.seek(0,os.SEEK_END);end=log.tell();log.seek(max(offset,end-65536));tail=log.read().decode(errors='replace')
                if 'OutOfMemoryError: CUDA out of memory' in tail:
                    cooldown=120;continue
                atomic_json(root/'supervisor.json',{'status':'failed','phase':phase,'exit_code':code});return code
            if not marker.exists():raise RuntimeError(f'Stage did not complete: {phase}')
            break
    atomic_json(root/'supervisor.json',{'status':'complete','finished_unix':time.time()});return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['setup','start','resume','stop','status','supervise','train','evaluate','select','report'])
    parser.add_argument('--job-root',default=str(PACKAGE/'runtime/plctx160_continuation'))
    parser.add_argument('--arm');parser.add_argument('--mode',choices=['baseline','test'],default='baseline')
    args=parser.parse_args();root=Path(args.job_root).resolve()
    if args.command=='setup':result=setup(args)
    elif args.command in ['start','resume','stop','status']:
        require((root/'job.json').exists(),'Setup continuation job first')
        if args.command in ['start','resume']:
            (root/'pause.request').unlink(missing_ok=True)
            subprocess.run(['systemctl','--user','reset-failed',UNIT],check=True)
            subprocess.run(['systemctl','--user','start',UNIT],check=True)
            result={'requested':args.command,'service':UNIT}
        elif args.command=='stop':
            (root/'pause.request').touch();subprocess.run(['systemctl','--user','stop',UNIT],check=True)
            result={'requested':'stop','policy':'checkpoint at optimizer boundary, pause marker survives reboot'}
        else:
            result={'service':subprocess.check_output(['systemctl','--user','show',UNIT,'--property=ActiveState,SubState,MainPID,NRestarts,Result,UnitFileState'],text=True)}
            for name in ['supervisor','progress']:
                if (root/f'{name}.json').exists():result[name]=json.loads((root/f'{name}.json').read_text())
    else:
        job=bootstrap(root)
        if args.command=='supervise':raise SystemExit(supervise(root,job))
        elif args.command=='train':
            from pj.continuation_train import run
            result=run(job,root,next(arm for arm in job['arms'] if arm['name']==args.arm))
        elif args.command=='evaluate':result=evaluate_stage(root,job,args.mode)
        else:result=select_and_report(root,job,args.command=='report')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
