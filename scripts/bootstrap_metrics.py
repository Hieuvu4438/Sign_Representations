"""Summarize locked predictions with paired recording-group bootstrap."""
import argparse
import json
from pathlib import Path

from _common import ROOT
from signrepr.io import sha256, write_json
from signrepr.statistics import align_predictions, grouped_bootstrap, read_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True, nargs='+')
    parser.add_argument('--reference', type=Path, nargs='+')
    parser.add_argument('--group-key', default='recording_group')
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/bootstrap_metrics.json')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    paths = args.predictions + (args.reference or [])
    tables = [read_predictions(path, args.group_key) for path in paths]
    _, gold, groups, correct = align_predictions(tables, args.group_key)
    seeds = []
    for table in tables:
        registered = {row['seed'] for row in table.values()}
        if len(registered) != 1:
            raise ValueError('A prediction table must contain one training seed')
        seeds.append(next(iter(registered)))
    count = len(args.predictions)
    if len(set(seeds[:count])) != count:
        raise ValueError('Duplicate training seeds cannot inflate replication count')
    if args.reference and seeds[:count] != seeds[count:]:
        raise ValueError('Pair the same ordered training seeds on both sides')
    inputs = [{'path': str(path), 'sha256': sha256(path)} for path in paths]
    if args.dry_run:
        print(json.dumps({'inputs': inputs, 'samples': len(gold), 'groups': len(set(groups)), 'training_seeds': seeds[:count]}))
        return
    if args.output.exists():
        raise ValueError('Output exists; use a new output path to preserve earlier statistics')
    result = grouped_bootstrap(gold, groups, correct[:count], correct[count:] if args.reference else None,
                               args.resamples, args.seed)
    result.update(status='PASS', group_key=args.group_key, inputs=inputs, ordered_seeds=seeds[:count],
                  implementation_sha256=sha256(ROOT / 'src/signrepr/statistics.py'))
    write_json(args.output, result)
    print(json.dumps({'output': str(args.output), 'metrics': result['metrics']}))


if __name__ == '__main__':
    main()
