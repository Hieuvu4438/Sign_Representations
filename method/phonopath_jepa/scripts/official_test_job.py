#!/usr/bin/env python3
"""Persistent official ASL Citizen retrieval for the completed A1 full160 teacher."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
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
DEFAULT=METHOD/'runtime/retrieval_e160_test'
SERVICE='phonopath-a1-e160-test-retrieval.service'
PYTHON=Path('/home/haipd/Sign_Representations/.venv-signrep/bin/python')
LOCAL=Path(__file__).resolve().parents[1]
BASE_PATH=LOCAL/'ops/retrieval_base.py' if (LOCAL/'ops/retrieval_base.py').exists() else METHOD/'scripts/retrieval_job.py'
spec=importlib.util.spec_from_file_location('official_retrieval_base',BASE_PATH)
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)


def ledger_update(job,state,readout=None,**extra):
    lock=B.read(job/'lock.json');path=METHOD/'runtime/test_ledger.json';path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('w') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);entries=B.read(path,[]) or []
        for key in B.READOUTS if readout is None else (readout,):
            identity=[lock['checkpoint_sha256'],lock['source_sha256']['test'],key]
            matching=[entry for entry in entries if entry.get('key')==identity]
            if matching:
                entry=matching[0]
                if entry.get('job')!=str(job):raise ValueError('This final checkpoint/test/readout is reserved by a different job')
            else:
                entry=dict(key=identity,job=str(job),dataset='ASL Citizen',query_split='test',epoch=160,
                           authorization='User explicitly requested official test after A1 full160 training completed',
                           no_test_selection=True,scheduled_unix=time.time());entries.append(entry)
            if entry.get('status')!='complete' or state=='complete':entry.update(status=state,updated_unix=time.time(),**extra)
        B.atomic_json(path,entries)


def setup(job):
    import torch
    from pp.data import PoseSource
    if (job/'lock.json').exists():raise FileExistsError('Job already exists; use status/resume')
    checkpoint=PRODUCTION/'train/final.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (ck['step'],ck['total'],ck['epoch'],ck['cursor'],ck['config']['arm'])!=(200640,200640,160,0,'A1'):
        raise ValueError('Expected completed A1 full160 final checkpoint')
    if B.read(PRODUCTION/'train/manifest.json')['status']!='complete':raise ValueError('Training is not marked complete')
    train=Path(ck['config']['cache']);test=train.with_name('metadata_test.jsonl')
    sources={split:PoseSource(path,split) for split,path in (('train',train),('test',test))}
    if [len(sources[s]) for s in ('train','test')]!=[40154,32941]:raise ValueError('Wrong official full-test counts')
    if set(sources['train'].ids)&set(sources['test'].ids):raise ValueError('Gallery/test overlap')
    if any(len({r['gloss'] for r in src.rows})!=2731 for src in sources.values()):raise ValueError('Expected2731 glosses')
    if B.sha256(train)!=ck['source_hash']:raise ValueError('Training manifest differs from checkpoint provenance')
    missing=[r['sample_id'] for src in sources.values() for r in src.rows if not (src.path.parent/r['pose_relpath']).is_file()]
    if missing:raise ValueError(f'Missing poses: {missing[:5]}')
    job.mkdir(parents=True,exist_ok=True)
    shutil.copytree(PRODUCTION/'code/pp',job/'code/pp',ignore=shutil.ignore_patterns('__pycache__'))
    (job/'ops').mkdir();shutil.copy2(Path(__file__),job/'ops/official_test_job.py')
    shutil.copy2(METHOD/'scripts/retrieval_job.py',job/'ops/retrieval_base.py')
    os.link(checkpoint,job/'checkpoint.pt')
    lock=dict(checkpoint_sha256=B.sha256(job/'checkpoint.pt'),checkpoint_source=str(checkpoint),checkpoint_step=200640,
              epoch=160,arm='A1',weights='teacher',sources={s:str(src.path) for s,src in sources.items()},
              source_sha256={s:B.sha256(src.path) for s,src in sources.items()},counts={'train':40154,'test':32941},
              code_sha256=B.hashes(job),python=str(PYTHON),stride=2,fps=25,window_batch=32,shard_clips=128,dtype='float16',
              readouts=list(B.READOUTS),primary_readout='grp+P',direct_comparison_readout='grp',
              test_used=True,no_test_selection=True,phase=4,pca_fit_split='train',
              min_disk_gib=8,required_gpu_free_mib=2048,gpu_memory_fraction=.045,stale_seconds=1200,
              authorization='Explicit user request: chạy official test đi, after full160 completion',created_unix=time.time())
    B.atomic_json(job/'lock.json',lock);ledger_update(job,'scheduled')
    B.atomic_json(job/'status.json',dict(stage='ready',updated_unix=time.time()))
    print(json.dumps({k:v for k,v in lock.items() if k!='code_sha256'},indent=2,ensure_ascii=False))


def validate_cache_metadata(gallery,query,lock):
    import numpy as np
    if (len(gallery['ids']),len(query['ids']))!=(40154,32941):raise ValueError('Incomplete official test')
    if set(gallery['ids'])&set(query['ids']):raise ValueError('Gallery/test overlap')
    for z,split in ((gallery,'train'),(query,'test')):
        meta=json.loads(str(z['metadata']))
        if meta['split']!=split or meta['checkpoint_sha256']!=lock['checkpoint_sha256'] or meta['source_sha256']!=lock['source_sha256'][split] or meta['limited']:
            raise ValueError(f'Mixed or partial feature cache: {split}')
        if len(np.unique(z['gloss']))!=2731:raise ValueError(f'Wrong vocabulary: {split}')


def score(job):
    import numpy as np
    import torch
    from pp.evaluate import class_max_ranks,metrics,pca_pair
    torch.set_num_threads(4);lock=B.read(job/'lock.json');ledger_update(job,'scoring')
    with np.load(job/'train.npz',allow_pickle=False) as gg,np.load(job/'test.npz',allow_pickle=False) as qq:
        validate_cache_metadata(gg,qq,lock)
        summary=dict(arm='A1',epoch=160,checkpoint_step=200640,checkpoint_sha256=lock['checkpoint_sha256'],weights='teacher',
                     dataset='ASL Citizen',gallery_split='train',query_split='test',gallery_n=40154,query_n=32941,n_glosses=2731,
                     primary_readout='grp+P',test_used=True,no_test_selection=True,pca_fit_split='train',readouts={})
        for key in B.READOUTS:
            name={'grp':'grp','grp+P':'grpP','mw':'mw','ord2':'ord2'}[key];result_path=job/'results'/f'{name}.json'
            existing=B.read(result_path)
            if existing:
                if existing['checkpoint_sha256']!=lock['checkpoint_sha256']:raise ValueError('Result checkpoint mismatch')
                ledger_update(job,'complete',key,result=str(result_path),metrics=existing['metrics'])
                summary['readouts'][key]=existing;continue
            B.atomic_json(job/'score_progress.json',dict(stage='scoring_test',readout=key,pid=os.getpid(),updated_unix=time.time()))
            G=gg[key].astype(np.float32);Q=qq[key].astype(np.float32);pca_hash=None
            if key!='grp':
                # Frozen model/readouts: only gallery statistics are fit. Q never
                # participates in PCA or any model/temperature selection.
                G,Q,transform=pca_pair(G,Q,768)
                path=job/'results'/f'{name}_pca.npz';B.atomic_npz(path,**transform);pca_hash=B.sha256(path)
            ranks=class_max_ranks(G,Q,gg['gloss'],qq['gloss'],device='cuda')
            B.atomic_npz(job/'results'/f'{name}_query_ranks.npz',ids=qq['ids'],gloss=qq['gloss'],signer=qq['signer'],ranks=ranks)
            result=dict(metrics=metrics(ranks,qq['gloss']),readout=key,pca=768 if key!='grp' else 0,pca_fit_split='train',
                        pca_sha256=pca_hash,checkpoint_sha256=lock['checkpoint_sha256'],epoch=160,split='test',
                        gallery_n=40154,query_n=32941,no_test_selection=True,ties='pessimistic class-max',created_unix=time.time())
            B.atomic_json(result_path,result);ledger_update(job,'complete',key,result=str(result_path),metrics=result['metrics'])
            summary['readouts'][key]=result;B.atomic_json(job/'results.json',summary);print(json.dumps(result),flush=True)
        summary['status']='complete';B.atomic_json(job/'results.json',summary)
        B.atomic_json(job/'score_progress.json',dict(stage='complete',updated_unix=time.time()))
        lines=['# A1 full160 — official ASL Citizen test','',
               'EMA teacher final160,200.640updates. Gallery train40.154/query test32.941,2.731gloss. Cosine class-max/pessimistic ties. PCA fit gallery train only; no selection on test.','',
               '| Readout | R@1 (%) | R@5 (%) | R@10 (%) | MRR | DCG | Macro R@1 (%) |','|---|---:|---:|---:|---:|---:|---:|']
        for key,res in summary['readouts'].items():
            m=res['metrics'];lines.append(f"| {key} | {m['R@1']:.3f} | {m['R@5']:.3f} | {m['R@10']:.3f} | {m['MRR']:.5f} | {m['DCG']:.5f} | {m['macro_R@1']:.3f} |")
        early=B.read(METHOD/'runtime/retrieval_e040_test/results.json')
        if early:
            lines+=['','## Same-readout change versus epoch40','','| Readout | Epoch40 R@1 (%) | Epoch160 R@1 (%) | Delta (points) |','|---|---:|---:|---:|']
            for key,res in summary['readouts'].items():
                a=early['readouts'][key]['metrics']['R@1'];b=res['metrics']['R@1'];lines.append(f'| {key} | {a:.3f} | {b:.3f} | {b-a:+.3f} |')
        lines+=['','Primary readout remains grp+P. Earlier epoch40 test was explicitly requested and recorded; no test score was used to stop or choose this final160 checkpoint.']
        (job/'REPORT_VI.md').write_text('\n'.join(lines)+'\n')


def admission(lock):
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    owned=[int(v) for v in r.stdout.splitlines() if v.strip().isdigit() and B.live(int(v))]
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    free=int(r.stdout.splitlines()[0]);return dict(ready=len(owned)<=1 and free>=lock['required_gpu_free_mib'],
                                                  owned_gpu_pids=owned,free_mib=free,required_mib=lock['required_gpu_free_mib'])


def supervise(job):
    stopping=False;child=None
    def stop(sig,frame):
        nonlocal stopping
        stopping=True
        if child and child.poll() is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGHUP,signal.SIG_IGN)
    with (job/'supervisor.lock').open('w') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);lock=B.read(job/'lock.json');failures=0
        def status(stage,**extra):B.atomic_json(job/'status.json',dict(stage=stage,supervisor_pid=os.getpid(),updated_unix=time.time(),**extra))
        if B.hashes(job)!=lock['code_sha256'] or B.sha256(job/'checkpoint.pt')!=lock['checkpoint_sha256']:raise ValueError('Frozen full160 evaluation identity changed')
        while not stopping and not (job/'paused').exists():
            if (B.read(job/'results.json',{}) or {}).get('status')=='complete':status('complete');return
            if shutil.disk_usage(job).free/2**30<lock['min_disk_gib']:status('waiting_disk');time.sleep(20);continue
            available=admission(lock)
            if not available['ready']:status('waiting_gpu',gpu=available);time.sleep(15);continue
            split=next((s for s in ('train','test') if not (job/f'{s}.npz').exists()),None)
            stage='extracting_'+split if split else 'scoring_test'
            argv=[lock['python'],str(job/'ops/official_test_job.py'),'extract' if split else 'score','--job',str(job)]
            if split:argv+=['--split',split]
            if split=='test':ledger_update(job,'extracting')
            env=os.environ.copy();env.update(PYTHONPATH=str(job/'code'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
            before=(B.read(job/f'{split}_progress.json',{}) or {}).get('committed',0) if split else 0
            with (job/f'{stage}.log').open('a') as stream:
                child=subprocess.Popen(argv,cwd=job,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                status(stage,child_pid=child.pid,gpu=available);last=time.time();observed=before;sent=None
                while child.poll() is None:
                    p=(B.read(job/f'{split}_progress.json',{}) or {}) if split else {}
                    if p.get('pid')==child.pid and p.get('completed',0)>observed:
                        observed=p['completed'];last=time.time();status(stage,child_pid=child.pid,progress=p)
                    if stopping or (job/'paused').exists() or shutil.disk_usage(job).free/2**30<lock['min_disk_gib'] or (split and time.time()-last>lock['stale_seconds']):
                        if sent is None:child.send_signal(signal.SIGTERM);sent=time.time()
                    if sent and time.time()-sent>180:os.killpg(child.pid,signal.SIGKILL)
                    time.sleep(5)
                code=child.wait();child=None
            if stopping or (job/'paused').exists():break
            after=(B.read(job/f'{split}_progress.json',{}) or {}).get('committed',0) if split else 0
            if code:failures=1 if after>before else failures+1;status('retry_wait',exit_code=code,failures=failures);time.sleep(30)
            else:failures=0
            if failures>=8:(job/'paused').write_text('Eight failures without new committed output; inspect logs before resume\n');status('needs_attention');return
        status('paused' if (job/'paused').exists() else 'stopped')


def install(job):
    target=Path.home()/'.config/systemd/user'/SERVICE
    if target.exists():raise FileExistsError('Full160 evaluation service already exists')
    target.write_text(f'''[Unit]
Description=PhonoPath-JEPA A1 full160 official ASL Citizen test retrieval
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
ExecStart={PYTHON} {job}/ops/official_test_job.py supervise --job {job}
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
    ap.add_argument('--job',type=Path,default=DEFAULT);ap.add_argument('--split',choices=['train','test']);a=ap.parse_args();job=a.job.resolve()
    if a.action=='setup':setup(job)
    elif a.action=='install':install(job)
    elif a.action=='supervise':supervise(job)
    elif a.action in ('extract','score'):
        import torch
        torch.cuda.set_per_process_memory_fraction(B.read(job/'lock.json')['gpu_memory_fraction'])
        if a.action=='extract':B.extract(job,a.split)
        else:score(job)
    elif a.action=='status':print(json.dumps(dict(state=B.read(job/'status.json'),train=B.read(job/'train_progress.json'),test=B.read(job/'test_progress.json'),results=B.read(job/'results.json')),indent=2))
    elif a.action=='pause':(job/'paused').write_text('Paused by user\n');subprocess.run(['systemctl','--user','stop',SERVICE],check=True)
    elif a.action=='resume':(job/'paused').unlink(missing_ok=True);subprocess.run(['systemctl','--user','start',SERVICE],check=True)
if __name__=='__main__':main()
