#!/usr/bin/env python3
"""Sequential subprocess runner with GPU-slot wait and bounded restart watchdog.

Default is dry-run; --run executes exactly the plan JSON, never shell strings.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pp.train import atomic_json


def gpu_processes():
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    # Count all compute processes conservatively; do not disrupt existing jobs.
    return [int(s.strip()) for s in r.stdout.splitlines() if s.strip().isdigit() and int(s.strip())!=os.getpid()]


def plan():
    runs=[dict(arm='A0',seed=s) for s in range(3)]+[dict(arm=a,seed=0) for a in ('A1','A2','A5','A6','A4','A9','A3','A7','A8a','A8b','A10')]
    return [dict(name=f"{r['arm']}_s{r['seed']}",argv=[sys.executable,'-m','pp.train','--config',f"configs/{r['arm']}_s{r['seed']}.json",
                  '--out',f"runtime/runs/{r['arm']}_s{r['seed']}"]) for r in runs]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan');ap.add_argument('--run',action='store_true')
    ap.add_argument('--restarts',type=int,default=2);ap.add_argument('--stale-seconds',type=int,default=900)
    ap.add_argument('--state',default='runtime/schedule.json');a=ap.parse_args()
    jobs=json.loads(Path(a.plan).read_text()) if a.plan else plan()
    if not a.run:print(json.dumps(jobs,indent=2));return
    statepath=ROOT/a.state;statepath.parent.mkdir(parents=True,exist_ok=True)
    lockpath=statepath.with_suffix('.lock')
    import fcntl
    with lockpath.open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(statepath.read_text()) if statepath.exists() else dict(jobs={})
        for job in jobs:
            name=job['name'];previous=state['jobs'].get(name,{})
            if previous.get('status')=='complete':continue
            argv=list(job['argv']);out=None
            if 'pp.train' in argv and '--out' in argv:out=ROOT/argv[argv.index('--out')+1]
            attempts=previous.get('attempts',0)
            if previous.get('status')=='running' and previous.get('pid'):
                try:os.kill(previous['pid'],0)
                except ProcessLookupError:pass
                else:raise RuntimeError(f"{name} still has live PID {previous['pid']}; inspect before resuming")
            while True:
                while len(gpu_processes())>1:
                    state['waiting_for_gpu']=True;atomic_json(statepath,state);time.sleep(30)
                if out and (out/'manifest.json').exists() and '--resume' not in argv:argv.append('--resume')
                logpath=ROOT/'runtime/logs'/f'{name}.log';logpath.parent.mkdir(parents=True,exist_ok=True)
                with logpath.open('a') as log:
                    child=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state['jobs'][name]=dict(status='running',pid=child.pid,attempts=attempts,argv=argv,
                        started=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    atomic_json(statepath,state)
                    launched=time.time();observed=False
                    while child.poll() is None:
                        if out:
                            progress=out/'train_log.jsonl'
                            if progress.exists():observed=True
                            changed=progress.stat().st_mtime if progress.exists() else launched
                            if time.time()-changed>a.stale_seconds:
                                child.terminate()
                                try:child.wait(timeout=30)
                                except subprocess.TimeoutExpired:child.kill();child.wait()
                                break
                        time.sleep(10)
                code=child.returncode
                complete=(out is None or ((out/'final.pt').exists() and json.loads((out/'manifest.json').read_text()).get('status')=='complete'))
                if code==0 and complete:
                    state['jobs'][name].update(status='complete',exit_code=code);atomic_json(statepath,state);break
                attempts+=1
                state['jobs'][name].update(status='failed',exit_code=code,attempts=attempts);atomic_json(statepath,state)
                if attempts>a.restarts:raise SystemExit(f'Watchdog exhausted retries for {name}')
if __name__=='__main__':main()
