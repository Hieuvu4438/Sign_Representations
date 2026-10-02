"""Wait for the existing suite and produce registered statistics and review tables."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from _common import ROOT
from signrepr.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-pid', type=int, required=True)
    parser.add_argument('--wait-timeout-hours', type=float, default=4)
    args = parser.parse_args()
    state = ROOT / 'state/common_postprocess.json'
    if state.exists():
        raise ValueError('Postprocess state already exists; inspect it before another launch')
    begun = time.monotonic()
    try:
        while True:
            suite = json.loads((ROOT / 'state/common_suite.json').read_text())
            if suite['status'] == 'PASS':
                break
            if suite['status'] == 'FAIL':
                raise ValueError('Suite failed; no resampling of incomplete learned-seed runs')
            command = Path('/proc') / str(args.suite_pid) / 'cmdline'
            if not command.exists() or b'scripts/run_common_suite.py' not in command.read_bytes():
                raise ValueError('Suite PID is no longer live without terminal state')
            if time.monotonic() - begun > args.wait_timeout_hours * 3600:
                raise ValueError('Wait timeout; no training process killed or restarted')
            write_json(state, {'status': 'WAITING_ON_SUITE', 'pid': os.getpid(), 'suite_pid': args.suite_pid,
                              'heartbeat_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
            time.sleep(30)
        def predictions(baseline, seeds):
            return [str(ROOT / f'runs/common_v1_{baseline}_signrep_seed{seed}/predictions.csv') for seed in seeds]
        primary = ROOT / 'reports/common_v1_primary_bootstrap.json'
        commands = [[sys.executable, 'scripts/bootstrap_metrics.py', '--predictions', *predictions('B03', [0, 1, 2]),
                     '--reference', *predictions('B02', [0, 1, 2]), '--output', str(primary)]]
        for baseline in ['B00', 'B01', 'B02', 'B03']:
            seeds = [0] if baseline in ['B00', 'B01'] else [0, 1, 2]
            commands.append([sys.executable, 'scripts/bootstrap_metrics.py', '--predictions', *predictions(baseline, seeds),
                             '--output', str(ROOT / f'reports/common_v1_{baseline}_bootstrap.json')])
        commands += [[sys.executable, 'scripts/audit_errors.py', '--predictions', *predictions('B03', [0, 1, 2])],
                     [sys.executable, 'scripts/update_run_registry.py'],
                     [sys.executable, 'scripts/build_report.py', '--registry', 'runs/experiment_registry.csv',
                      '--gates', 'reports/gate_decisions.json', '--output', 'reports/research_status.md']]
        for command in commands:
            write_json(state, {'status': 'RUNNING_CPU_POSTPROCESS', 'pid': os.getpid(), 'command': command})
            subprocess.run(command, cwd=ROOT, check=True)
        write_json(state, {'status': 'PASS_STATISTICS_EXPERT_REVIEW_PENDING', 'primary_statistics': str(primary),
                          'error_audit': 'reports/common_v1_error_audit/report.json',
                          'novelty_status': 'NOT_ESTABLISHED'})
    except Exception as error:
        write_json(state, {'status': 'FAIL', 'failure_reason': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    main()
