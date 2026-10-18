#!/usr/bin/env python3
"""User-authorized epoch40 ASLC official test, queued after fixed VAL-DICT.

Reuses the gallery and gallery-fitted PCA without selecting on test.
"""
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
VALIDATION=METHOD/'runtime/retrieval_e040'
DEFAULT=METHOD/'runtime/retrieval_e040_test'
SERVICE='phonopath-a1-e040-test-retrieval.service'
PYTHON=Path('/home/haipd/Sign_Representations/.venv-signrep/bin/python')
LOCAL=Path(__file__).resolve().parents[1]
BASE_PATH=LOCAL/'ops/retrieval_base.py' if (LOCAL/'ops/retrieval_base.py').exists() else METHOD/'scripts/retrieval_job.py'
spec=importlib.util.spec_from_file_location('epoch40_retrieval_base',BASE_PATH)
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)


def ledger_update(job,state,readout=None,**extra):
    lock=B.read(job/'lock.json');path=METHOD/'runtime/test_ledger.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('w') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        entries=B.read(path,[]) or []
        for key in B.READOUTS if readout is None else (readout,):
            identity=[lock['checkpoint_sha256'],lock['source_sha256']['test'],key]
            matching=[e for e in entries if e.get('key')==identity]
            if matching:
                entry=matching[0]
                if entry.get('job')!=str(job):raise ValueError('This checkpoint/test/readout belongs to another test job')
            else:
                entry=dict(key=identity,job=str(job),dataset='ASL Citizen',query_split='test',epoch=40,
                           authorization='User explicitly requested test in addition to validation at epoch40',
                           no_test_selection=True,scheduled_unix=time.time())
                entries.append(entry)
            if entry.get('status')!='complete' or state=='complete':entry.update(status=state,updated_unix=time.time(),**extra)
        B.atomic_json(path,entries)


def setup(job):
    from pp.data import PoseSource
    original=B.read(VALIDATION/'lock.json');test=Path(original['sources']['train']).with_name('metadata_test.jsonl')
    source=PoseSource(test,'test')
    if len(source)!=32941 or len({r['gloss'] for r in source.rows})!=2731:raise ValueError('Expected official32941 test clips/2731 glosses')
    for split in ('train','val'):
        other=PoseSource(original['sources'][split],split)
        if set(source.ids)&set(other.ids):raise ValueError('Test ID overlap')
    if any(not (source.path.parent/r['pose_relpath']).is_file() for r in source.rows):raise ValueError('Missing test pose')
    if (job/'lock.json').exists():raise FileExistsError('Test job exists; use status/resume')
    job.mkdir(parents=True,exist_ok=True)
    shutil.copytree(VALIDATION/'code/pp',job/'code/pp',ignore=shutil.ignore_patterns('__pycache__'))
    (job/'ops').mkdir();shutil.copy2(Path(__file__),job/'ops/test_retrieval_job.py')
    shutil.copy2(VALIDATION/'ops/retrieval_job.py',job/'ops/retrieval_base.py')
    os.link(VALIDATION/'checkpoint.pt',job/'checkpoint.pt')
    lock=dict(original)
    lock.update(sources={'train':original['sources']['train'],'test':str(test)},
                source_sha256={'train':original['source_sha256']['train'],'test':B.sha256(test)},counts={'train':40154,'test':32941},
                gallery_file=str(VALIDATION/'train.npz'),dependency_results=str(VALIDATION/'results.json'),
                pca_folder=str(VALIDATION/'results'),test_used=True,
                authorization='Explicit user request: đánh giá trên cả test luôn',no_test_selection=True,
                research_stage='epoch40 early test requested by user; not a held-out final160 selection',
                created_unix=time.time(),code_sha256=B.hashes(job))
    B.atomic_json(job/'lock.json',lock);ledger_update(job,'scheduled')
    B.atomic_json(job/'status.json',dict(stage='queued_after_validation',updated_unix=time.time(),test_clips=32941))
    print(json.dumps({k:v for k,v in lock.items() if k!='code_sha256'},indent=2,ensure_ascii=False))


def prerequisite(job):
    lock=B.read(job/'lock.json');summary=B.read(lock['dependency_results'],{}) or {}
    if summary.get('status')!='complete':return False
    if summary['checkpoint_sha256']!=lock['checkpoint_sha256'] or summary.get('gallery_n')!=40154 or summary.get('query_n')!=10304:
        raise ValueError('Validation dependency identity mismatch')
    for name in ('grpP','mw','ord2'):
        if not (Path(lock['pca_folder'])/f'{name}_pca.npz').exists():raise ValueError('Missing gallery-fitted PCA')
    return True


def transform_frozen(x,mean,components):
    """Exactly sklearn PCA.transform semantics, using train-fitted parameters only."""
    # sklearn BasePCA._transform with whiten=False projects first, then
    # subtracts the projected training mean. Preserve that operation order.
    projected=x@components.T
    projected-=mean.reshape(1,-1)@components.T
    return projected


def score(job):
    import numpy as np
    import torch
    from pp.evaluate import class_max_ranks,metrics
    torch.set_num_threads(4);lock=B.read(job/'lock.json')
    if not prerequisite(job):raise ValueError('Validation/gallery/PCA not ready')
    ledger_update(job,'scoring')
    with np.load(lock['gallery_file'],allow_pickle=False) as gg,np.load(job/'test.npz',allow_pickle=False) as qq:
        if (len(gg['ids']),len(qq['ids']))!=(40154,32941):raise ValueError('Wrong official test protocol counts')
        if set(gg['ids'])&set(qq['ids']):raise ValueError('Gallery/test overlap')
        for z,split in ((gg,'train'),(qq,'test')):
            meta=json.loads(str(z['metadata']))
            if meta['split']!=split or meta['checkpoint_sha256']!=lock['checkpoint_sha256'] or meta['source_sha256']!=lock['source_sha256'][split]:
                raise ValueError('Feature identity mismatch')
        summary=dict(arm='A1',epoch=40,checkpoint_step=50160,checkpoint_sha256=lock['checkpoint_sha256'],weights='teacher',
                     dataset='ASL Citizen',gallery_split='train',query_split='test',gallery_n=40154,query_n=32941,n_glosses=2731,
                     primary_readout='grp+P',test_used=True,no_test_selection=True,readouts={})
        for key in B.READOUTS:
            name={'grp':'grp','grp+P':'grpP','mw':'mw','ord2':'ord2'}[key];result_path=job/'results'/f'{name}.json'
            existing=B.read(result_path)
            if existing:
                if existing['checkpoint_sha256']!=lock['checkpoint_sha256']:raise ValueError('Existing result identity mismatch')
                ledger_update(job,'complete',key,result=str(result_path),metrics=existing['metrics'])
                summary['readouts'][key]=existing;continue
            B.atomic_json(job/'score_progress.json',dict(stage='scoring_test',readout=key,pid=os.getpid(),updated_unix=time.time()))
            G=gg[key].astype(np.float32);Q=qq[key].astype(np.float32);pca_hash=None
            if key!='grp':
                path=Path(lock['pca_folder'])/f'{name}_pca.npz';pca_hash=B.sha256(path)
                with np.load(path,allow_pickle=False) as z:G=transform_frozen(G,z['mean'],z['components']);Q=transform_frozen(Q,z['mean'],z['components'])
            ranks=class_max_ranks(G,Q,gg['gloss'],qq['gloss'],device='cuda')
            B.atomic_npz(job/'results'/f'{name}_query_ranks.npz',ids=qq['ids'],gloss=qq['gloss'],signer=qq['signer'],ranks=ranks)
            result=dict(metrics=metrics(ranks,qq['gloss']),readout=key,pca=768 if key!='grp' else 0,pca_fit_split='train',
                        frozen_pca_sha256=pca_hash,checkpoint_sha256=lock['checkpoint_sha256'],epoch=40,split='test',
                        gallery_n=40154,query_n=32941,no_test_selection=True,created_unix=time.time())
            B.atomic_json(result_path,result);ledger_update(job,'complete',key,result=str(result_path),metrics=result['metrics'])
            summary['readouts'][key]=result;B.atomic_json(job/'results.json',summary);print(json.dumps(result),flush=True)
        summary['status']='complete';B.atomic_json(job/'results.json',summary)
        lines=['# A1 epoch40 — ASL Citizen official test','',
               'Gallery train40.154 / query test32.941;2.731 gloss. Fixed EMA checkpoint epoch40. User requested early test; no checkpoint/readout selection on test.','',
               '| Readout | R@1 (%) | R@5 (%) | R@10 (%) | MRR | DCG | Macro R@1 (%) |','|---|---:|---:|---:|---:|---:|---:|']
        for key,res in summary['readouts'].items():
            m=res['metrics'];lines.append(f"| {key} | {m['R@1']:.3f} | {m['R@5']:.3f} | {m['R@10']:.3f} | {m['MRR']:.5f} | {m['DCG']:.5f} | {m['macro_R@1']:.3f} |")
        lines+=['','PCA reused from the train gallery of VAL-DICT. This is epoch40, not the final160 model; account for different training budgets in baseline comparisons.']
        (job/'REPORT_VI.md').write_text('\n'.join(lines)+'\n')


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
        if B.hashes(job)!=lock['code_sha256'] or B.sha256(job/'checkpoint.pt')!=lock['checkpoint_sha256']:raise ValueError('Frozen test job changed')
        while not stopping and not (job/'paused').exists():
            if (B.read(job/'results.json',{}) or {}).get('status')=='complete':status('complete');return
            if not prerequisite(job):status('waiting_validation_and_gallery',dependency=lock['dependency_results']);time.sleep(15);continue
            if shutil.disk_usage(job).free/2**30<lock['min_disk_gib']:status('waiting_disk');time.sleep(20);continue
            available=B.admission()
            if not available['ready']:status('waiting_gpu',gpu=available);time.sleep(15);continue
            extracting=not (job/'test.npz').exists();stage='extracting_test' if extracting else 'scoring_test'
            argv=[lock['python'],str(job/'ops/test_retrieval_job.py'),'extract' if extracting else 'score','--job',str(job)]
            if extracting:ledger_update(job,'extracting')
            env=os.environ.copy();env.update(PYTHONPATH=str(job/'code'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with (job/f'{stage}.log').open('a') as stream:
                child=subprocess.Popen(argv,cwd=job,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                status(stage,child_pid=child.pid);last=time.time();observed=-1;sent=None
                while child.poll() is None:
                    p=B.read(job/'test_progress.json',{}) or {}
                    if extracting and p.get('pid')==child.pid and p.get('completed',0)>observed:
                        observed=p['completed'];last=time.time();status(stage,child_pid=child.pid,progress=p)
                    if stopping or (job/'paused').exists() or shutil.disk_usage(job).free/2**30<lock['min_disk_gib'] or (extracting and time.time()-last>lock['stale_seconds']):
                        if sent is None:child.send_signal(signal.SIGTERM);sent=time.time()
                    if sent and time.time()-sent>180:os.killpg(child.pid,signal.SIGKILL)
                    time.sleep(5)
                code=child.wait();child=None
            if stopping or (job/'paused').exists():break
            if code:failures+=1;status('retry_wait',exit_code=code,failures=failures);time.sleep(30)
            else:failures=0
            if failures>=8:(job/'paused').write_text('Repeated test evaluation errors; inspect logs before resume\n');status('needs_attention');return
        status('paused' if (job/'paused').exists() else 'stopped')


def install(job):
    target=Path.home()/'.config/systemd/user'/SERVICE
    if target.exists():raise FileExistsError('Test service already exists')
    target.write_text(f'''[Unit]
Description=PhonoPath-JEPA epoch40 user-authorized official test retrieval
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
ExecStart={PYTHON} {job}/ops/test_retrieval_job.py supervise --job {job}
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
    ap.add_argument('--job',type=Path,default=DEFAULT);a=ap.parse_args();job=a.job.resolve()
    if a.action=='setup':setup(job)
    elif a.action=='install':install(job)
    elif a.action=='supervise':supervise(job)
    elif a.action=='extract':B.extract(job,'test')
    elif a.action=='score':score(job)
    elif a.action=='status':print(json.dumps(dict(state=B.read(job/'status.json'),progress=B.read(job/'test_progress.json'),results=B.read(job/'results.json')),indent=2))
    elif a.action=='pause':(job/'paused').write_text('Paused by user\n');subprocess.run(['systemctl','--user','stop',SERVICE],check=True)
    elif a.action=='resume':(job/'paused').unlink(missing_ok=True);subprocess.run(['systemctl','--user','start',SERVICE],check=True)
if __name__=='__main__':main()
