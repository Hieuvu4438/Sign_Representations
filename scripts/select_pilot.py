"""Select at most 20 deterministic train-only smoke clips and lock their IDs."""
import argparse
import hashlib
import json
from pathlib import Path

from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/manifests/asl_citizen.jsonl')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/manifests/pilot.jsonl')
    args = parser.parse_args()
    if not 1 <= args.count <= 20:
        raise ValueError('Smoke count must be between 1 and 20')
    eligible = [row for _, row in read_jsonl(args.manifest) if row['official_split'] == 'train' and row['source_exists']]
    selected = sorted(eligible, key=lambda row: hashlib.sha256((str(args.seed) + ':' + row['sample_id']).encode()).hexdigest())[:args.count]
    write_jsonl(args.output, selected)
    write_json(ROOT / 'reports/pilot_selection.json', {'seed': args.seed, 'source_manifest_sha256': sha256(args.manifest),
               'pilot_manifest_sha256': sha256(args.output), 'split': 'train',
               'sample_ids': [row['sample_id'] for row in selected],
               'scope': 'Deterministic train-only smoke; natural short/variable-FPS coverage must be checked after decode.'})
    print(json.dumps({'selected': len(selected), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
