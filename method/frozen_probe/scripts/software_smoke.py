"""Record synthetic protocol checks with exclusive status and code snapshots."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time
import traceback
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_signrep import sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if args.seed!=config['projection_seed']:raise ValueError('Seed/config mismatch')
    args.output_dir.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    status={'run_id':args.run_id,'protocol':config['protocol'],'status':'running','seed':args.seed,
            'GPU_hours':0,'pid':os.getpid(),'host':platform.node(),'start_utc':datetime.now(timezone.utc).isoformat(),
            'command':[sys.executable]+sys.argv,'config_sha256':sha(args.config),'script_sha256':sha(__file__)}
    def save(name,value):
        temp=args.output_dir/(name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(args.output_dir/name)
    def timeout_handler(*_):raise TimeoutError('Software fixture budget exhausted')
    save('status.json',status);signal.signal(signal.SIGALRM,timeout_handler);signal.alarm(config['timeout_seconds'])
    try:
        import numpy as np
        paths=sorted(p for p in ROOT.rglob('*.py') if 'runtime' not in p.relative_to(ROOT).parts
                     and '__pycache__' not in p.relative_to(ROOT).parts)
        snapshot=args.output_dir/'source_snapshot';snapshot.mkdir()
        hashes={str(path.relative_to(ROOT)):sha(path) for path in paths}
        for path in paths:
            dest=snapshot/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        shutil.copyfile(args.config,args.output_dir/'config.json')
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=config['test_pattern'])
        with (args.output_dir/'tests.txt').open('w') as log:
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        report={'run_id':args.run_id,'scope':'synthetic/unit software checks only','tests_run':result.testsRun,
                'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
                'passed':result.wasSuccessful() and not result.skipped,'code_hashes':hashes,'python':sys.version,'numpy':np.__version__,
                'configuration':config,'log_sha256':sha(args.output_dir/'tests.txt')}
        save('report.json',report)
        if not report['passed']:raise ValueError('Software fixture failed')
        status.update(status='completed',exit_code=0,output_sha256=sha(args.output_dir/'report.json'))
    except Exception as error:
        status.update(status='failed',exit_code=1,error=repr(error));(args.output_dir/'error.txt').write_text(traceback.format_exc())
    finally:
        signal.alarm(0);status.update(end_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started)
        save('status.json',status)
    print(json.dumps(status),flush=True)
    if status['status']!='completed':sys.exit(1)


if __name__=='__main__':main()
