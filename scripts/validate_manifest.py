"""Validate sample schema, intervals, paths, and declared annotation status."""
import argparse
import json
from pathlib import Path

from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json
from signrepr.validation import validate_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({'manifest_bytes': args.manifest.stat().st_size, 'writes': str(args.report)}))
        return
    report = validate_rows([row for _, row in read_jsonl(args.manifest)])
    report['manifest_sha256'] = sha256(args.manifest)
    write_json(args.report, report)
    print(json.dumps({key: report[key] for key in ['status', 'counts']}))
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
