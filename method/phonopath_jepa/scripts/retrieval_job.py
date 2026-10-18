#!/usr/bin/env python3
"""Resumable official ASLC VAL-DICT evaluation of the frozen A1 epoch40 teacher.

Uses an immutable copy of the exact production model/extractor. It never loads test.
"""
from __future__ import annotations
import argparse
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

METHOD=Path('/home/haipd/Sign_Representations/method/phonopath_jepa')
PRODUCTION=METHOD/'runtime/full160'
DEFAULT=METHOD/'runtime/retrieval_e040'
PYTHON=Path('/home/haipd/Sign_Representations/.venv-signrep/bin/python')
SERVICE='phonopath-a1-e040-retrieval.service'
# Installed worker is job/ops/retrieval_job.py; tests and setup use production code.
LOCAL=Path(__file__).resolve().parents[1]
SNAPSHOT=LOCAL/'code' if (LOCAL/'code/pp').is_dir() else PRODUCTION/'code'
sys.path.insert(0,str(SNAPSHOT))
from pp.recovery import atomic_json,sync_directory

READOUTS=('grp','grp+P','mw','ord2')
DIMS={'grp':768,'grp+P':1536,'mw':1536,'ord2':3072}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default


def atomic_npz(path,**data):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('wb') as stream:
        np.savez(stream,**data);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path);sync_directory(path.parent)


def hashes(job):
    return {str(p.relative_to(job)):sha256(p) for folder in (job/'code/pp',job/'ops') for p in sorted(folder.rglob('*.py'))}


def setup(job):
    from pp.data import PoseSource
    import torch
    if (job/'lock.json').exists():raise FileExistsError('Evaluation job exists; use status/resume')
    checkpoint=PRODUCTION/'train/epoch_040.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['step']!=50160 or ck['epoch']!=40 or ck['config']['arm']!='A1':raise ValueError('Not the A1 epoch40 checkpoint')
    train=Path(ck['config']['cache']);val=train.with_name('metadata_val.jsonl')
    sources={s:PoseSource(p,s) for s,p in (('train',train),('val',val))}
    if [len(sources[s]) for s in ('train','val')]!=[40154,10304]:raise ValueError('Incorrect official split counts')
    if set(sources['train'].ids)&set(sources['val'].ids):raise ValueError('Split overlap')
    if any(len({r['gloss'] for r in src.rows})!=2731 for src in sources.values()):raise ValueError('Expected 2731 glosses')
    missing=[r['sample_id'] for src in sources.values() for r in src.rows if not (src.path.parent/r['pose_relpath']).is_file()]
    if missing:raise ValueError(f'Missing pose files: {missing[:5]}')
    job.mkdir(parents=True,exist_ok=True)
    shutil.copytree(PRODUCTION/'code/pp',job/'code/pp',ignore=shutil.ignore_patterns('__pycache__'))
    (job/'ops').mkdir();shutil.copy2(Path(__file__),job/'ops/retrieval_job.py')
    os.link(checkpoint,job/'checkpoint.pt')
    lock=dict(checkpoint_sha256=sha256(job/'checkpoint.pt'),checkpoint_source=str(checkpoint),checkpoint_step=50160,
              epoch=40,arm='A1',weights='teacher',sources={s:str(src.path) for s,src in sources.items()},
              source_sha256={s:sha256(src.path) for s,src in sources.items()},counts={s:len(src) for s,src in sources.items()},
              code_sha256=hashes(job),python=str(PYTHON),stride=2,fps=25,window_batch=32,shard_clips=128,dtype='float16',
              readouts=list(READOUTS),primary_readout='grp+P',direct_comparison_readout='grp',test_used=False,
              min_disk_gib=8,required_gpu_free_mib=4096,stale_seconds=1200,created_unix=time.time())
    atomic_json(job/'lock.json',lock);print(json.dumps(lock,indent=2))


def completed_shards(job,split,lock,source):
    import numpy as np
    paths=sorted((job/'shards'/split).glob('*.npz'));end=0
    for path in paths:
        with np.load(path,allow_pickle=False) as z:
            start,stop=int(z['start']),int(z['stop'])
            if start!=end or stop<=start or str(z['checkpoint'])!=lock['checkpoint_sha256'] or str(z['source'])!=lock['source_sha256'][split]:
                raise ValueError(f'Shard identity/order mismatch: {path}')
            if not np.array_equal(z['ids'],source.ids[start:stop]):raise ValueError(f'Shard ID mismatch: {path}')
            for key in READOUTS:
                if z[key].shape!=(stop-start,DIMS[key]) or not np.isfinite(z[key]).all():raise ValueError(f'Invalid features: {path}/{key}')
            end=stop
    return end,paths


def merge(job,split,lock,source,paths):
    import numpy as np
    arrays={key:[] for key in READOUTS};ids=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as z:
            ids.extend(z['ids'].tolist())
            for key in READOUTS:arrays[key].append(z[key])
    if ids!=source.ids.tolist():raise ValueError('Incomplete extraction cannot produce official feature cache')
    meta=dict(split=split,checkpoint_sha256=lock['checkpoint_sha256'],source_sha256=lock['source_sha256'][split],
              stride=2,fps=25,drop=0,jitter=0,reverse=False,limited=False,raw_window_mean=True,dtype='float16')
    rows=source.rows
    atomic_npz(job/f'{split}.npz',**{key:np.concatenate(value) for key,value in arrays.items()},ids=source.ids,
               labels=np.array([r['class_id'] for r in rows]),gloss=np.array([r['gloss'] for r in rows]),
               signer=np.array([r['signer_id'] for r in rows]),metadata=np.array(json.dumps(meta)),rows=np.array(json.dumps(rows)))


def extract(job,split):
    import numpy as np
    import torch
    from pp.data import PoseSource
    from pp.extract import Extractor
    torch.set_num_threads(4)
    lock=read(job/'lock.json');source=PoseSource(lock['sources'][split],split)
    if sha256(source.path)!=lock['source_sha256'][split]:raise ValueError('Source metadata changed')
    end,paths=completed_shards(job,split,lock,source)
    if end==len(source):merge(job,split,lock,source,paths);return
    stopping=False
    def stop(sig,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGHUP,signal.SIG_IGN)
    ext=Extractor(job/'checkpoint.pt','cuda');start_index=end;begin=end;buffers={key:[] for key in READOUTS}
    prior=read(job/f'{split}_progress.json',{}) or {};elapsed_prior=prior.get('elapsed_seconds',0.)
    started=time.monotonic();last_status=0.;recent=[];times=[]
    def progress(state):
        elapsed=time.monotonic()-started;count=end-start_index
        rate=count/elapsed if elapsed>0 and count else None
        value=dict(stage=state,split=split,pid=os.getpid(),completed=end,total=len(source),committed=begin,
                   percent=100*end/len(source),clips_per_second=rate,eta_seconds=(len(source)-end)/rate if rate else None,
                   elapsed_seconds=elapsed_prior+elapsed,updated_unix=time.time(),resumed_at=start_index,
                   checkpoint_sha256=lock['checkpoint_sha256'])
        atomic_json(job/f'{split}_progress.json',value)
    def flush():
        nonlocal begin,paths,buffers
        if end==begin:return
        path=job/'shards'/split/f'{begin:07d}_{end:07d}.npz'
        values={key:np.stack(value).astype(np.float16) for key,value in buffers.items()}
        if any(not np.isfinite(v).all() for v in values.values()):raise ValueError(f'Nonfinite {split} embeddings')
        atomic_npz(path,**values,ids=source.ids[begin:end],start=np.array(begin),stop=np.array(end),
                   checkpoint=np.array(lock['checkpoint_sha256']),source=np.array(lock['source_sha256'][split]))
        paths.append(path);begin=end;buffers={key:[] for key in READOUTS}
    progress('extracting')
    try:
        for i in range(end,len(source)):
            if stopping:break
            x,c,fps=source.clip(i)
            values=ext.clip(x,c,fps,stride=2,batch=32,target_fps=25.)
            for key in READOUTS:buffers[key].append(values[key])
            end=i+1
            if end-begin>=lock['shard_clips']:flush()
            if end==start_index+1 or time.monotonic()-last_status>=5:
                progress('extracting');last_status=time.monotonic()
            if end%250==0:print(json.dumps(read(job/f'{split}_progress.json')),flush=True)
        flush();progress('interrupted' if stopping else 'merging')
        if end==len(source):merge(job,split,lock,source,paths);progress('complete')
    except BaseException:
        flush();progress('failed');raise


def score(job):
    import numpy as np
    import torch
    from pp.evaluate import class_max_ranks,metrics,pca_pair
    torch.set_num_threads(4);lock=read(job/'lock.json')
    with np.load(job/'train.npz',allow_pickle=False) as gg,np.load(job/'val.npz',allow_pickle=False) as qq:
        if (len(gg['ids']),len(qq['ids']))!=(40154,10304):raise ValueError('Not official full VAL-DICT')
        if set(gg['ids'])&set(qq['ids']):raise ValueError('Gallery/query overlap')
        for z in (gg,qq):
            if json.loads(str(z['metadata']))['checkpoint_sha256']!=lock['checkpoint_sha256']:raise ValueError('Mixed checkpoints')
        summary=dict(arm='A1',epoch=40,checkpoint_step=50160,weights='teacher',checkpoint_sha256=lock['checkpoint_sha256'],
                     dataset='ASL Citizen',gallery_split='train',query_split='val',gallery_n=40154,query_n=10304,n_glosses=2731,
                     primary_readout='grp+P',test_used=False,readouts={})
        for key in READOUTS:
            name={'grp':'grp','grp+P':'grpP','mw':'mw','ord2':'ord2'}[key]
            result_path=job/'results'/f'{name}.json'
            existing=read(result_path)
            if existing:
                if existing['checkpoint_sha256']!=lock['checkpoint_sha256']:raise ValueError('Result identity mismatch')
                summary['readouts'][key]=existing;continue
            atomic_json(job/'score_progress.json',dict(stage='scoring',readout=key,updated_unix=time.time(),pid=os.getpid()))
            G=gg[key].astype(np.float32);Q=qq[key].astype(np.float32)
            if key!='grp':
                G,Q,transform=pca_pair(G,Q,768)
                atomic_npz(job/'results'/f'{name}_pca.npz',**transform)
            ranks=class_max_ranks(G,Q,gg['gloss'],qq['gloss'],device='cuda')
            atomic_npz(job/'results'/f'{name}_query_ranks.npz',ids=qq['ids'],gloss=qq['gloss'],signer=qq['signer'],ranks=ranks)
            result=dict(metrics=metrics(ranks,qq['gloss']),readout=key,pca=768 if key!='grp' else 0,
                        checkpoint_sha256=lock['checkpoint_sha256'],epoch=40,split='val',gallery_n=40154,query_n=10304,
                        ties='pessimistic class-max',created_unix=time.time())
            atomic_json(result_path,result);summary['readouts'][key]=result;atomic_json(job/'results.json',summary)
            print(json.dumps(result),flush=True)
        summary['status']='complete';atomic_json(job/'results.json',summary)
        atomic_json(job/'score_progress.json',dict(stage='complete',updated_unix=time.time()))
        text=['# A1 epoch40 — ASL Citizen VAL-DICT','',
              'Gallery official train: 40.154 clip. Query official val: 10.304 clip; 2.731 gloss. EMA teacher, cosine class-max, ties bi quan. Không dùng official test.','',
              '| Readout | R@1 (%) | R@5 (%) | R@10 (%) | MRR | DCG | Macro R@1 (%) |','|---|---:|---:|---:|---:|---:|---:|']
        for key,result in summary['readouts'].items():
            m=result['metrics'];text.append(f"| {key} | {m['R@1']:.3f} | {m['R@5']:.3f} | {m['R@10']:.3f} | {m['MRR']:.5f} | {m['DCG']:.5f} | {m['macro_R@1']:.3f} |")
        text+=['','`grp+P` là readout chính, PCA768 chỉ fit gallery train. `grp` không PCA dùng cho so sánh cùng chiều với A0. Không so trực tiếp VAL-DICT này với R@1 test PLctx full160.']
        (job/'REPORT_VI.md').write_text('\n'.join(text)+'\n')


def live(pid):
    try:return (Path('/proc')/str(pid)).stat().st_uid==os.getuid() and (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except OSError:return False


def admission():
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    owned=[int(v) for v in r.stdout.splitlines() if v.strip().isdigit() and live(int(v))]
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    free=int(r.stdout.splitlines()[0]);return dict(ready=len(owned)<=1 and free>=4096,owned_gpu_pids=owned,free_mib=free)


def supervise(job):
    stopping=False;child=None
    def stop(sig,frame):
        nonlocal stopping
        stopping=True
        if child and child.poll() is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGHUP,signal.SIG_IGN)
    with (job/'supervisor.lock').open('w') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock=read(job/'lock.json');failures=0
        def status(stage,**extra):atomic_json(job/'status.json',dict(stage=stage,supervisor_pid=os.getpid(),updated_unix=time.time(),**extra))
        if hashes(job)!=lock['code_sha256'] or sha256(job/'checkpoint.pt')!=lock['checkpoint_sha256']:raise ValueError('Frozen evaluation identity changed')
        while not stopping and not (job/'paused').exists():
            if (read(job/'results.json',{}) or {}).get('status')=='complete':status('complete',results=str(job/'results.json'));return
            if shutil.disk_usage(job).free/2**30<lock['min_disk_gib']:
                status('waiting_disk');time.sleep(20);continue
            available=admission()
            if not available['ready']:status('waiting_gpu',gpu=available);time.sleep(15);continue
            split=next((s for s in ('train','val') if not (job/f'{s}.npz').exists()),None)
            argv=[lock['python'],str(job/'ops/retrieval_job.py'),'extract' if split else 'score','--job',str(job)]
            if split:argv+=['--split',split]
            stage='extracting_'+split if split else 'scoring'
            env=os.environ.copy();env.update(PYTHONPATH=str(job/'code'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with (job/f'{stage}.log').open('a') as stream:
                child=subprocess.Popen(argv,cwd=job,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                status(stage,child_pid=child.pid,gpu=available);started=time.time();last=started;observed=-1;sent=None
                while child.poll() is None:
                    p=read(job/f'{split}_progress.json',{}) if split else read(job/'score_progress.json',{})
                    if p and split and p.get('pid')==child.pid and p.get('completed',0)>observed:
                        observed=p['completed'];last=time.time();status(stage,child_pid=child.pid,progress=p)
                    if stopping or (job/'paused').exists() or shutil.disk_usage(job).free/2**30<lock['min_disk_gib'] or (split and time.time()-last>lock['stale_seconds']):
                        if sent is None:child.send_signal(signal.SIGTERM);sent=time.time()
                    if sent and time.time()-sent>180:os.killpg(child.pid,signal.SIGKILL)
                    time.sleep(5)
                code=child.wait();child=None
            if stopping or (job/'paused').exists():break
            if code:failures+=1;status('retry_wait',exit_code=code,failures=failures);time.sleep(30)
            else:failures=0
            if failures>=8:
                (job/'paused').write_text('Eight consecutive evaluation errors; inspect logs before resume.\n');status('needs_attention');return
        status('paused' if (job/'paused').exists() else 'stopped')


def install(job):
    target=Path.home()/'.config/systemd/user'/SERVICE
    if target.exists():raise FileExistsError('Service already installed')
    target.write_text(f'''[Unit]
Description=PhonoPath-JEPA epoch40 official VAL-DICT retrieval
After=default.target
StartLimitIntervalSec=0
[Service]
Type=simple
WorkingDirectory={job}
Environment=PYTHONPATH={job}/code
Environment=PYTHONUNBUFFERED=1
Environment=OMP_NUM_THREADS=4
Environment=OPENBLAS_NUM_THREADS=4
Environment=MKL_NUM_THREADS=4
ExecStart={PYTHON} {job}/ops/retrieval_job.py supervise --job {job}
Restart=on-failure
RestartSec=30
KillMode=mixed
TimeoutStopSec=240
Nice=10
CPUQuota=600%
MemoryHigh=12G
MemoryMax=20G
StandardOutput=append:{job}/service.log
StandardError=append:{job}/service.log
[Install]
WantedBy=default.target
''')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','enable','--now',SERVICE],check=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['setup','install','supervise','extract','score','status','pause','resume'])
    ap.add_argument('--job',type=Path,default=DEFAULT);ap.add_argument('--split',choices=['train','val']);a=ap.parse_args();job=a.job.resolve()
    if a.action=='setup':setup(job)
    elif a.action=='install':install(job)
    elif a.action=='supervise':supervise(job)
    elif a.action=='extract':extract(job,a.split)
    elif a.action=='score':score(job)
    elif a.action=='status':print(json.dumps(dict(state=read(job/'status.json'),train=read(job/'train_progress.json'),val=read(job/'val_progress.json'),results=read(job/'results.json')),indent=2))
    elif a.action=='pause':(job/'paused').write_text('Paused by user\n');subprocess.run(['systemctl','--user','stop',SERVICE],check=True)
    elif a.action=='resume':(job/'paused').unlink(missing_ok=True);subprocess.run(['systemctl','--user','start',SERVICE],check=True)
if __name__=='__main__':main()
