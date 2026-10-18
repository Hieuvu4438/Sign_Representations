#!/usr/bin/env python3
"""Persistent, isolated A1 full160 supervisor and control CLI (systemd user service)."""
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

ROOT=Path(__file__).resolve().parents[1]
SERVICE='phonopath-a1-full160.service'
DEFAULT_JOB=Path('/home/haipd/Sign_Representations/method/phonopath_jepa/runtime/full160')
PYTHON=Path('/home/haipd/Sign_Representations/.venv-signrep/bin/python')
# Snapshot execution imports exclusively from this job's code directory.
sys.path.insert(0,str(ROOT/'code' if (ROOT/'code/pp').exists() else ROOT))
from pp.recovery import atomic_json


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def pid_identity(pid):
    try:
        p=Path('/proc')/str(pid)
        stat=(p/'stat').read_text();tail=stat.rsplit(')',1)[1].split()
        if tail[0]=='Z':return None
        return dict(pid=int(pid),uid=p.stat().st_uid,start_ticks=int(tail[19]))
    except (OSError,ValueError,IndexError):return None


def same_process(saved):
    if not saved:return False
    return pid_identity(saved['pid'])==saved


def gpu_admission(required_mib=14000,exclude_pid=None):
    """The proposal limits THIS user's jobs to two; never modify another job."""
    p=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    free=int(p.stdout.splitlines()[0].strip())
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    owned=[];others=[]
    for value in p.stdout.splitlines():
        if not value.strip().isdigit():continue
        pid=int(value);ident=pid_identity(pid)
        if ident and pid!=exclude_pid:
            (owned if ident['uid']==os.getuid() else others).append(pid)
    return dict(ready=len(owned)<=1 and free>=required_mib,free_mib=free,required_mib=required_mib,
                other_owned_gpu_pids=owned,other_user_gpu_pids=others)


def code_hashes(job):
    return {str(p.relative_to(job)):sha256(p) for directory in (job/'code/pp',job/'ops')
            for p in sorted(directory.rglob('*.py'))}|{name:sha256(job/name) for name in ('config.json','fit_config.json')}


def setup(job):
    from pp.config import Config
    from pp.data import PoseSource
    job.mkdir(parents=True,exist_ok=True)
    if (job/'lock.json').exists():raise FileExistsError('Job already installed; use status/resume')
    cfg=Config.load(job/'config.json').validate()
    assert cfg.arm=='A1' and cfg.epochs==160 and cfg.batch==32 and cfg.device=='cuda'
    source=PoseSource(cfg.cache,'train')
    if len(source)!=40154:raise ValueError(f'Expected 40154 train clips, got {len(source)}')
    missing=[r['sample_id'] for r in source.rows if not (source.path.parent/r['pose_relpath']).is_file()]
    if missing:raise ValueError(f'Missing poses: {missing[:10]}')
    shutil.copytree(ROOT/'pp',job/'code/pp',ignore=shutil.ignore_patterns('__pycache__'))
    (job/'ops').mkdir();shutil.copy2(Path(__file__),job/'ops/full160_job.py')
    report=dict(n_train=len(source),train_sha256=sha256(source.path),missing_poses=0,steps_per_epoch=len(source)//32,
                total_steps=160*(len(source)//32),free_disk_gib=shutil.disk_usage(job).free/2**30,
                gpu=gpu_admission(),scope='User explicitly requested direct A1 full160, without screening or official test',
                test_evaluation=False,created_unix=time.time())
    atomic_json(job/'preflight.json',report)
    atomic_json(job/'lock.json',dict(job=str(job.resolve()),python=str(PYTHON),service=SERVICE,uid=os.getuid(),
               source_root=str(ROOT),code_sha256=code_hashes(job),train_sha256=report['train_sha256'],
               required_free_mib=14000,min_disk_free_gib=8,checkpoint_steps=200,checkpoint_seconds=60,
               stale_seconds=600,restart_delay_seconds=30,created_unix=time.time()))
    print(json.dumps(report,indent=2))


def install(job):
    lock=json.loads((job/'lock.json').read_text());folder=Path.home()/'.config/systemd/user';folder.mkdir(parents=True,exist_ok=True)
    unit=folder/SERVICE
    if unit.exists():raise FileExistsError(f'{unit} already exists; inspect before replacing')
    unit.write_text(f'''[Unit]
Description=PhonoPath-JEPA A1 full160 persistent train and recovery
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
ExecStart={lock['python']} {job}/ops/full160_job.py supervise --job {job}
Restart=on-failure
RestartSec=30
KillMode=mixed
TimeoutStopSec=240
Nice=5
CPUQuota=800%
MemoryHigh=20G
MemoryMax=32G
StandardOutput=append:{job}/service.log
StandardError=append:{job}/service.log
[Install]
WantedBy=default.target
''')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','enable','--now',SERVICE],check=True)


def tail_json(path):
    if not path.exists():return None
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return None


def supervise(job):
    lock=json.loads((job/'lock.json').read_text());out=job/'train';out.mkdir(exist_ok=True)
    stopping=False;child=None
    def request_stop(sig,frame):
        nonlocal stopping
        stopping=True
        if child and child.poll() is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGHUP,signal.SIG_IGN)
    with (job/'supervisor.lock').open('w') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=tail_json(job/'status.json') or {};attempts=int(state.get('attempts',0));failures=0
        def status(stage,**extra):
            state.update(stage=stage,updated_unix=time.time(),supervisor=pid_identity(os.getpid()),attempts=attempts,**extra)
            atomic_json(job/'status.json',state)
        if code_hashes(job)!=lock['code_sha256']:raise RuntimeError('Immutable job code/config changed')
        from pp.config import Config
        cfg=Config.load(job/'config.json')
        if sha256(cfg.cache)!=lock['train_sha256']:raise RuntimeError('Training manifest changed')
        # A supervisor crash may leave its training child alive. The saved process
        # birth tick prevents confusing a reused PID with this job's child.
        previous=state.get('child')
        if same_process(previous):
            status('adopting_live_child',child=previous)
            seen=time.time();step=(tail_json(out/'progress.json') or {}).get('step',0);stop_sent=None
            while same_process(previous) and not stopping and not (job/'paused').exists():
                progress=tail_json(out/'progress.json') or {}
                if progress.get('step',0)>step:step=progress['step'];seen=time.time()
                if time.time()-seen>lock['stale_seconds'] and stop_sent is None:
                    os.kill(previous['pid'],signal.SIGTERM);stop_sent=time.time()
                if stop_sent and time.time()-stop_sent>180:
                    os.killpg(previous['pid'],signal.SIGKILL)
                time.sleep(5)
            if stopping or (job/'paused').exists():
                if same_process(previous):
                    os.kill(previous['pid'],signal.SIGTERM);deadline=time.monotonic()+180
                    while same_process(previous) and time.monotonic()<deadline:time.sleep(2)
                status('paused' if (job/'paused').exists() else 'stopped');return
        while not stopping and not (job/'paused').exists():
            if code_hashes(job)!=lock['code_sha256']:raise RuntimeError('Snapshot identity changed')
            manifest=tail_json(out/'manifest.json') or {}
            if (out/'final.pt').exists() and manifest.get('status')=='complete':
                status('complete',child=None,train_progress=tail_json(out/'progress.json'));return
            free=shutil.disk_usage(job).free/2**30
            if free<lock['min_disk_free_gib']:
                status('waiting_disk',free_disk_gib=free,child=None);time.sleep(30);continue
            if not (job/'stats.pt').exists():
                status('fitting_train_statistics',child=None)
                argv=[lock['python'],'-m','pp.fit_stats','--config',str(job/'fit_config.json'),'--out',str(job/'stats.pt'),'--clips','640']
            else:
                from pp.signature import RobustStats
                try:stats=RobustStats.load(job/'stats.pt')
                except Exception as exc:
                    (job/'stats.pt').rename(job/f'stats.pt.corrupt.{time.time_ns()}')
                    status('recovering_statistics',error=repr(exc));continue
                provenance=stats.provenance
                if provenance['clips']!=640 or provenance['source_sha256']!=lock['train_sha256']:raise RuntimeError('Statistics provenance mismatch')
                admission=gpu_admission(lock['required_free_mib'])
                if not admission['ready']:
                    status('waiting_gpu',gpu=admission,child=None);time.sleep(15);continue
                # GPU smoke has its own checkpoint; production weights always start
                # from the registered seed and never from this diagnostic model.
                if not (job/'gpu_preflight.json').exists():
                    status('gpu_preflight',gpu=admission,child=None)
                    argv=[lock['python'],str(job/'ops/full160_job.py'),'gpu-smoke','--job',str(job)]
                else:
                    status('starting_train',gpu=admission,child=None)
                    argv=[lock['python'],'-m','pp.train','--config',str(job/'config.json'),'--out',str(out),
                          '--checkpoint-steps',str(lock['checkpoint_steps']),'--checkpoint-seconds',str(lock['checkpoint_seconds'])]
                    if any((out/n).exists() for n in ('latest.pt','previous.pt','initial.pt')):argv.append('--resume')
                    elif manifest:raise RuntimeError('Training was recorded but every checkpoint is missing; refusing reset')
            attempts+=1
            stage='training' if 'pp.train' in argv else 'gpu_preflight' if 'gpu-smoke' in argv else 'fitting_train_statistics'
            path=job/('train_stdout.log' if stage=='training' else stage+'.log')
            before=tail_json(out/'progress.json') or {};before_step=before.get('step',0)
            env=os.environ.copy();env.update(PYTHONPATH=str(job/'code'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with path.open('a') as stream:
                child=subprocess.Popen(argv,cwd=job,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                status(stage,child=pid_identity(child.pid),command=argv,started_unix=time.time())
                launched=time.time();last_progress=launched;last_step=before_step;stale_stop=None;pressure_stop=None
                while child.poll() is None:
                    progress=tail_json(out/'progress.json') or {}
                    if stage=='training' and progress.get('pid')==child.pid:
                        if progress.get('step',0)>last_step:last_progress=time.time();last_step=progress['step']
                        status(stage,train_progress=progress)
                    if stopping or (job/'paused').exists():
                        if stale_stop is None:child.send_signal(signal.SIGTERM);stale_stop=time.time()
                    elif stage=='training' and time.time()-last_progress>lock['stale_seconds']:
                        if stale_stop is None:child.send_signal(signal.SIGTERM);stale_stop=time.time();status('recovering_stalled_train')
                    if stage=='training' and shutil.disk_usage(job).free/2**30<lock['min_disk_free_gib'] and pressure_stop is None:
                        child.send_signal(signal.SIGTERM);pressure_stop=time.time();stale_stop=pressure_stop;status('saving_for_disk_pressure')
                    if stale_stop and time.time()-stale_stop>180:
                        os.killpg(child.pid,signal.SIGKILL)
                    time.sleep(5)
                code=child.wait();child=None
            status('child_exited',child=None,last_exit_code=code,train_progress=tail_json(out/'progress.json'))
            if stopping or (job/'paused').exists():break
            completed=tail_json(out/'progress.json') or {}
            if code==0:failures=0
            elif completed.get('step',0)>before_step:failures=1
            else:failures+=1
            if failures>=8:
                (job/'paused').write_text('Automatic pause: eight failures without training progress; inspect logs then resume.\n')
                status('needs_attention',consecutive_failures=failures);return
            status('retry_wait' if code else 'next_stage',consecutive_failures=failures)
            if code:time.sleep(lock['restart_delay_seconds'])
        status('paused' if (job/'paused').exists() else 'stopped',child=None,train_progress=tail_json(out/'progress.json'))


def gpu_smoke(job):
    import numpy as np
    import torch
    from pp.config import Config
    from pp.data import PoseSource,Windows,collate
    from pp.train import Trainer
    from pp.signature import RobustStats
    lock=json.loads((job/'lock.json').read_text());cfg=Config.load(job/'config.json')
    admission=gpu_admission(lock['required_free_mib'])
    if not admission['ready']:raise RuntimeError(f'GPU slot changed: {admission}')
    torch.set_num_threads(4);torch.manual_seed(cfg.seed);torch.cuda.reset_peak_memory_stats()
    source=PoseSource(cfg.cache);ds=Windows(source,cfg);stats=RobustStats.load(job/'stats.pt',cfg.asinh)
    tr=Trainer(cfg,stats);batch=collate([ds[i] for i in range(cfg.batch)])
    torch.cuda.synchronize();start=time.monotonic();loss,gn,terms=tr.step(batch,0,200640);torch.cuda.synchronize()
    seconds=time.monotonic()-start
    path=job/'gpu_smoke.pt';tr.save(path,1,0,1,lock['train_sha256'],200640)
    restored=Trainer(cfg,stats);assert restored.load(path,lock['train_sha256'],200640)==(1,0,1)
    torch.cuda.synchronize();start=time.monotonic();loss2,gn2,terms2=restored.step(batch,1,200640);torch.cuda.synchronize()
    second=time.monotonic()-start
    if not np.isfinite([loss,gn,loss2,gn2]).all():raise ValueError('Nonfinite CUDA/BF16 smoke')
    report=dict(device=torch.cuda.get_device_name(),batch=cfg.batch,bf16=True,losses=[loss,loss2],grad_norms=[gn,gn2],
                seconds=[seconds,second],steps_per_second_warm=1/second,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                cuda_checkpoint_reload=True,closure=terms2,official_test_used=False,created_unix=time.time())
    atomic_json(job/'gpu_preflight.json',report);path.unlink();path.with_name(path.name+'.json').unlink(missing_ok=True)
    print(json.dumps(report,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['setup','install','supervise','status','pause','resume','checkpoint','gpu-smoke'])
    ap.add_argument('--job',type=Path,default=DEFAULT_JOB);a=ap.parse_args();job=a.job.resolve()
    if a.action=='setup':setup(job)
    elif a.action=='install':install(job)
    elif a.action=='supervise':supervise(job)
    elif a.action=='gpu-smoke':gpu_smoke(job)
    elif a.action=='status':
        result=dict(job=str(job),state=tail_json(job/'status.json'),train=tail_json(job/'train/progress.json'))
        result['service']=subprocess.run(['systemctl','--user','show',SERVICE,'-p','ActiveState','-p','SubState','-p','MainPID','-p','NRestarts'],capture_output=True,text=True).stdout
        print(json.dumps(result,indent=2))
    elif a.action=='pause':
        (job/'paused').write_text('Paused explicitly by user.\n');subprocess.run(['systemctl','--user','stop',SERVICE],check=True)
    elif a.action=='resume':
        (job/'paused').unlink(missing_ok=True);subprocess.run(['systemctl','--user','start',SERVICE],check=True)
    elif a.action=='checkpoint':
        ident=(tail_json(job/'status.json') or {}).get('child')
        state=tail_json(job/'train/progress.json') or {}
        if not same_process(ident) or state.get('pid')!=ident['pid']:raise RuntimeError('No verified live trainer')
        os.kill(ident['pid'],signal.SIGUSR1);print('Checkpoint requested at next optimizer boundary')
if __name__=='__main__':main()
