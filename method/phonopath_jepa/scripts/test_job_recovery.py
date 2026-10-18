#!/usr/bin/env python3
"""Real CLI signal/checkpoint/corruption/resume test, isolated CPU smoke artifacts."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pp.recovery import atomic_json


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return {}


def wait_for(check,child,timeout=60):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=check()
        if value:return value
        if child.poll() is not None:raise RuntimeError(f'Child exited early: {child.returncode}')
        time.sleep(.1)
    raise TimeoutError('Recovery test did not reach expected boundary')


def main():
    job=ROOT/'runtime/recovery_cli_test';job.mkdir(parents=True,exist_ok=False)
    cfg=read(ROOT/'runtime/cli_smoke_final/config.json');cfg['stats']=str(ROOT/'runtime/cli_smoke_final/stats.pt')
    atomic_json(job/'config.json',cfg);out=job/'train'
    command=[sys.executable,'-m','pp.train','--config',str(job/'config.json'),'--out',str(out),'--max-steps','80',
             '--checkpoint-steps','10','--checkpoint-seconds','1']
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    with (job/'first.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            initial=wait_for(lambda:read(out/'progress.json').get('step',0)>=1,child)
            child.send_signal(signal.SIGUSR1)
            checkpoint=wait_for(lambda:read(out/'latest.pt.json').get('step',0)>=1,child)
            child.send_signal(signal.SIGTERM);code=child.wait(timeout=60)
        finally:
            if child.poll() is None:child.kill();child.wait()
    state=read(out/'manifest.json');assert code==0 and state['status']=='interrupted'
    saved=read(out/'latest.pt.json')['step'];previous=read(out/'previous.pt.json') if (out/'previous.pt.json').exists() else {}
    assert 0<saved<80 and (out/'previous.pt').exists()
    (out/'latest.pt').write_bytes(b'injected corruption for isolated recovery test')
    with (job/'resume.log').open('w') as log:subprocess.run(command+['--resume'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    state=read(out/'manifest.json');progress=read(out/'progress.json')
    assert state['status']=='complete' and state['steps_done']==80 and progress['epoch']==20
    assert progress['recovery_errors'] and progress['resumed_from'].endswith('previous.pt')
    result=dict(sigusr1_checkpoint=True,sigterm_saved=True,interrupted_step=saved,corrupt_latest_fallback=True,
                resumed_from=progress['resumed_from'],final_step=80,epoch_cursor_consistent=True,passed=True)
    atomic_json(ROOT/'evidence/job_recovery_cli.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
