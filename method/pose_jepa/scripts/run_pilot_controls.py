"""Run fixed-update controls sequentially and preserve exact process evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python', required=True)
    p.add_argument('--configs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ('masked_geometry', 'masked_motion', 'jepa_only', 'jepa_hybrid', 'jepa_shuffled'):
        config = args.configs/f'pilot_{name}.yaml'
        command = [args.python, '-m', 'signrepr.cli', 'pretrain', '--config', str(config), '--profile', 'pilot', '--device', args.device, '--cpu-threads', '2']
        started = time.time()
        with (args.output/f'{name}.log').open('w') as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            live = {'run': name, 'pid': process.pid, 'command': command, 'started_unix': started, 'status': 'running'}
            (args.output/'live.json').write_text(json.dumps(live, indent=2))
            print(json.dumps(live), flush=True)
            code = process.wait()
        record = {**live, 'exit_code': code, 'wall_clock_sec': time.time()-started, 'status': 'complete' if code == 0 else 'failed'}
        records.append(record)
        (args.output/'results.json').write_text(json.dumps(records, indent=2))
        (args.output/'live.json').write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
        if code: raise SystemExit(code)


if __name__ == '__main__': main()
