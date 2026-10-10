"""Read-only job monitoring through shared artifacts, without starting GPU jobs."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import time


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    experiment = Path(__file__).resolve().parents[1] / 'experiments' / read(folder / 'run_config.json')['experiment']
    previous = None
    while True:
        pipeline = read(folder / 'pipeline_status.json')
        initial = read(folder / 'step0_status.json')
        training = read(folder / 'training_status.json')
        history = []
        metrics = folder / 'train/metrics.jsonl'
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    # The producer may be appending its last record.
                    break
        finite = bool(history) and all(
            math.isfinite(row.get('loss', float('nan'))) and
            math.isfinite(row.get('gradient_norm', float('nan')))
            for row in history)
        updates = len(history)
        record = {
            'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'pipeline_stage': pipeline.get('stage'),
            'step0_completed_videos': initial.get('completed_videos'),
            'recorded_training_updates': updates,
            'training_progress_confirmed': finite and updates >= 25,
            'all_observed_losses_and_gradients_finite': finite if updates else None,
            'latest_training_status': training,
            'pipeline_error': pipeline.get('error'),
            'monitor_pid': os.getpid(),
        }
        temporary = experiment / 'monitor_status.json.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        os.replace(temporary, experiment / 'monitor_status.json')
        fingerprint = (record['pipeline_stage'], record['step0_completed_videos'], updates)
        if fingerprint != previous:
            with (experiment / 'monitor_history.jsonl').open('a') as file:
                file.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            previous = fingerprint
        if pipeline.get('stage') in {'COMPLETED', 'FAILED', 'WAITING_FOR_GPU'}:
            return
        if updates and not finite:
            raise ValueError('Observed nonfinite training loss or gradient; inspect the GPU job')
        time.sleep(30)


if __name__ == '__main__':
    main()
