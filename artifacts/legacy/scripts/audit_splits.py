"""Audit recording/signer split overlap without inferring identity from video."""
import argparse
import json
from pathlib import Path

from _common import ROOT
from signrepr.io import read_jsonl, sha256, write_json
from signrepr.validation import audit_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({'manifest_bytes': args.manifest.stat().st_size, 'output': str(args.output)}))
        return
    report = audit_rows([row for _, row in read_jsonl(args.manifest)])
    report['manifest_sha256'] = sha256(args.manifest)
    report['protocol_config_sha256'] = sha256(args.config) if args.config else None
    write_json(args.output, report)
    print(json.dumps({'status': report['status'], 'recording_overlaps': len(report['recording_overlaps']), 'signer_overlaps': len(report['signer_overlaps']), 'unresolved': report['unresolved_recording_counts']}))
    if report['recording_overlaps']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
