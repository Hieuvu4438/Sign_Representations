"""Wait on a verified extraction PID, then run frozen controls serially on one GPU."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from _common import ROOT
from signrepr.io import write_json


def process_alive(pid):
    try:
        command = Path('/proc') / str(pid) / 'cmdline'
        return b'scripts/extract_features.py' in command.read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction-pid', type=int, required=True)
    parser.add_argument('--wait-timeout-hours', type=float, default=8.)
    args = parser.parse_args()
    state_path = ROOT / 'state/common_suite.json'
    started = time.monotonic()
    report_path = ROOT / 'features/signrep_common_v1/extraction_report.json'
    if not report_path.exists() and not process_alive(args.extraction_pid):
        raise ValueError('Extraction handle is not live and no terminal report exists')
    while not report_path.exists():
        if not process_alive(args.extraction_pid):
            write_json(state_path, {'status': 'FAIL', 'reason': 'Extraction PID terminal without report', 'pid': os.getpid()})
            raise SystemExit(1)
        if time.monotonic() - started > args.wait_timeout_hours * 3600:
            write_json(state_path, {'status': 'FAIL', 'reason': 'Wait timeout; extraction not killed or restarted', 'pid': os.getpid()})
            raise SystemExit(1)
        index = ROOT / 'features/signrep_common_v1/index.jsonl'
        cached = sum(1 for _ in index.open()) if index.exists() else 0
        write_json(state_path, {'status': 'WAITING_ON_LIVE_EXTRACTION', 'pid': os.getpid(), 'extraction_pid': args.extraction_pid,
                               'cached_samples': cached, 'heartbeat_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        print(json.dumps({'status': 'WAITING_ON_LIVE_EXTRACTION', 'cached_samples': cached}), flush=True)
        time.sleep(30)
    report = json.loads(report_path.read_text())
    if report['status'] != 'PASS':
        write_json(state_path, {'status': 'FAIL', 'reason': 'Extraction report failed; probes not run'})
        raise SystemExit(1)
    while process_alive(args.extraction_pid):
        time.sleep(1)
    jobs = [('B00', 0), ('B01', 0)] + [(baseline, seed) for baseline in ['B02', 'B03'] for seed in [0, 1, 2]]
    for baseline, seed in jobs:
        command = [sys.executable, str(ROOT / 'scripts/run_probe.py'), '--config', str(ROOT / 'configs/protocol_common_v1.yaml'),
                   '--baseline', baseline, '--seed', str(seed)]
        log_path = ROOT / ('logs/common_v1_' + baseline + '_seed' + str(seed) + '.log')
        with log_path.open('w') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
            write_json(state_path, {'status': 'RUNNING_PROBE', 'pid': os.getpid(), 'probe_pid': child.pid,
                                   'baseline': baseline, 'seed': seed, 'command': command, 'log': str(log_path)})
            exit_code = child.wait()
        print(json.dumps({'baseline': baseline, 'seed': seed, 'exit_code': exit_code}), flush=True)
        if exit_code:
            write_json(state_path, {'status': 'FAIL', 'baseline': baseline, 'seed': seed, 'exit_code': exit_code,
                                   'reason': 'Probe failed; no automatic retry', 'log': str(log_path)})
            raise SystemExit(exit_code)
    write_json(state_path, {'status': 'PASS', 'completed_jobs': jobs, 'statistics_status': 'PENDING'})


if __name__ == '__main__':
    main()
