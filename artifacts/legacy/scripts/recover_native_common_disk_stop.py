"""One reviewed recovery after disk guard stop; retain original attempts.

Wait for60GiBfree, finish CPU preprocessing, then run the soleGPU0consumer
on complete inputs and the same locked CPU fair suite in a separate output.
No automatic retry of a failure in this recovery attempt.
"""
import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from _common import ROOT
from signrepr.io import sha256, write_json

OUTPUT = ROOT / 'state/native_common_disk_recovery_v1'
FAIR_OUTPUT = ROOT / 'runs/fair_common_v1_recovery1'
MIN_FREE_GIB = 60


def read(path):
    return json.loads(path.read_text())


def free_gib():
    return shutil.disk_usage(ROOT).free / 1024**3


def stage(command, log, environment):
    with log.open('x') as stream:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'Child exited{completed.returncode}; preserve log{log.relative_to(ROOT)}; no retry')


def main():
    if OUTPUT.exists() or FAIR_OUTPUT.exists():
        raise ValueError('Preserve prior recovery attempt')
    audit_path = ROOT / 'reports/native_common_disk_recovery_audit.json'
    audit = read(audit_path)
    if audit['status'] != 'PASS_EXISTING_CACHE_IDENTITIES_SOURCE_STREAM_AND_SHARD_HASHES':
        raise ValueError('Reviewed recovery cache audit required')
    if sha256(ROOT / 'configs/protocol_native_common_v1.json') != audit['native_protocol_sha256']:
        raise ValueError('Reviewed native protocol changed')
    if sha256(ROOT / 'features/shubert_native_common_v1/index.jsonl') != audit['feature_index_sha256']:
        raise ValueError('Reviewed native feature index changed')
    for name in ['features/shubert_native_common_v1', 'features/shubert_native_common_v1_preprocessing']:
        state = read(ROOT / name / 'status.json')
        if state['status'] != 'FAIL' or 'reserved disk limit' not in state.get('failure_reason', ''):
            raise ValueError('Only reviewed disk-guard terminal failure may resume')
    old_fair = read(ROOT / 'runs/fair_common_v1/status.json')
    if old_fair['status'] != 'FAIL' or (ROOT / 'runs/fair_common_v1/summary.json').exists():
        raise ValueError('Preserve original pre-fitting failure; no existing test summary allowed')
    OUTPUT.mkdir(parents=True)
    state = {'status': 'WAITING_FOR_CAPACITY', 'pid': os.getpid(), 'device': 'cpu',
             'minimum_free_gib': MIN_FREE_GIB, 'audit_sha256': sha256(audit_path),
             'script_sha256': sha256(Path(__file__)), 'gpu_workers_when_extracting': 1,
             'fair_output': str(FAIR_OUTPUT.relative_to(ROOT))}
    started = time.perf_counter()
    try:
        with (OUTPUT / '.worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while free_gib() < MIN_FREE_GIB:
                state.update(free_gib=free_gib(), elapsed_seconds=time.perf_counter()-started,
                             updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                write_json(OUTPUT / 'status.json', state)
                time.sleep(30)
            # Original cache workers retain their exclusive locks and identity
            # validation. The previous state/log/attempt records are archived.
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            state.update(status='CPU_PREPROCESSING_RECOVERY', free_gib=free_gib())
            write_json(OUTPUT / 'status.json', state)
            stage([str(ROOT / '.venv-shubert-native/bin/python'), 'scripts/prepare_shubert_cohort.py',
                   '--manifest', 'data/manifests/common_v1.jsonl', '--output',
                   'features/shubert_native_common_v1_preprocessing', '--resume'],
                  ROOT / 'logs/shubert_native_common_v1_preprocessing_recovery1.log', env)
            prep = read(ROOT / 'features/shubert_native_common_v1_preprocessing/status.json')
            if prep['status'] != 'PASS_NATIVE_PREPROCESSING' or prep['passed'] != prep['samples']:
                raise ValueError('Complete CPU preprocessing required before GPU recovery')
            state.update(status='GPU0_FEATURE_RECOVERY')
            write_json(OUTPUT / 'status.json', state)
            stage([str(ROOT / '.venv-shubert-native/bin/python'), '-X', 'faulthandler',
                   'scripts/extract_shubert_cohort.py', '--device', 'cuda:0', '--resume'],
                  ROOT / 'logs/shubert_native_common_v1_extraction_recovery1.log', {**env, 'CUDA_VISIBLE_DEVICES': '0'})
            native = read(ROOT / 'features/shubert_native_common_v1/status.json')
            if native['status'] != 'PASS' or native['success'] != native['samples']:
                raise ValueError('Full native PASS required before any lexical evaluation')
            state.update(status='CPU_EQUAL_GRID_CONTROLS_RECOVERY')
            write_json(OUTPUT / 'status.json', state)
            code = "import sys;sys.path.insert(0,'scripts');from pathlib import Path;import run_fair_common as suite;suite.OUTPUT=Path('runs/fair_common_v1_recovery1').resolve();suite.run(False)"
            stage([str(ROOT / '.venv-signrep/bin/python'), '-c', code], ROOT / 'logs/fair_common_v1_recovery1.log', env)
            measured = read(FAIR_OUTPUT / 'summary.json')
            if not measured['status'].startswith('PASS'):
                raise ValueError('Fair recovery suite did not PASS')
            state.update(status='PASS_REVIEWED_DISK_RECOVERY_AND_LOCKED_CONTROLS', elapsed_seconds=time.perf_counter()-started)
            write_json(OUTPUT / 'status.json', state)
    except Exception as error:
        state.update(status='FAIL', failure_reason=type(error).__name__+': '+str(error), elapsed_seconds=time.perf_counter()-started)
        write_json(OUTPUT / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
