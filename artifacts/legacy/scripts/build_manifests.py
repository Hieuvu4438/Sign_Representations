"""Normalize local datasets, preserving official split and missing-file evidence."""
import argparse
import json
from collections import Counter
from pathlib import Path

from _common import ROOT
from signrepr import manifests as adapters
from signrepr.io import sha256, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/manifests')
    parser.add_argument('--dataset', nargs='+', choices=['asl_citizen', 'wlasl2000', 'ms_asl', 'how2sign', 'csl_daily', 'csl_daily_sentence_crop', 'phoenix14t'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    lab, shared = Path('/home/dongvk/datasets'), Path('/home/shared_data/sign_language')
    factories = {
        'asl_citizen': lambda: adapters.asl_citizen(lab / 'ASL_Citizen'),
        'wlasl2000': lambda: adapters.wlasl(lab / 'WLASL2000'),
        'ms_asl': lambda: adapters.msasl(lab / 'MS-ASL'),
        'how2sign': lambda: adapters.how2sign(shared / 'How2Sign', lab / 'How2Sign'),
        'csl_daily': lambda: adapters.csl(shared / 'CSLDaily', 'csl_daily', 'videos/videos'),
        'csl_daily_sentence_crop': lambda: adapters.csl(lab / 'CSL_Daily_Sentence_Crop', 'csl_daily_sentence_crop', 'videos'),
        'phoenix14t': lambda: adapters.phoenix(lab / 'phoenix14T'),
    }
    if args.dry_run:
        print(json.dumps({'datasets': args.dataset or list(factories), 'copies_source_video': False, 'output': str(args.output)}))
        return
    coverage, all_rows, isolated, exclusions = {}, [], [], []
    for name in args.dataset or list(factories):
        rows = list(factories[name]())
        write_jsonl(args.output / (name + '.jsonl'), rows)
        sources = sorted({row['metadata_source'] for row in rows})
        coverage[name] = {'annotation_rows': len(rows), 'video_or_frames_present': sum(row['source_exists'] for row in rows),
                          'split_counts': dict(Counter(row['official_split'] for row in rows)),
                          'unresolved_recording_groups': sum(row.get('grouping_status') == 'SOURCE_RECORDING_UNKNOWN' for row in rows),
                          'metadata_sha256': {path: sha256(path) for path in sources},
                          'manifest_sha256': sha256(args.output / (name + '.jsonl')),
                          'gold_grammar': 'NOT_ESTABLISHED', 'gold_sign_occurrences': 'NOT_ESTABLISHED'}
        all_rows.extend(rows)
        if name in {'asl_citizen', 'wlasl2000', 'ms_asl'}:
            isolated.extend(rows)
        exclusions.extend({'sample_id': row['sample_id'], 'reason': 'MISSING_EXISTING_DATA', 'path': str(Path(row['source_root']) / row['relative_path'])} for row in rows if not row['source_exists'])
        print(json.dumps({'dataset': name, **{k: v for k, v in coverage[name].items() if k not in ['metadata_sha256']}}), flush=True)
    write_jsonl(args.output / 'all.jsonl', all_rows)
    write_jsonl(args.output / 'isolated.jsonl', isolated)
    write_jsonl(args.output / 'exclusions.jsonl', exclusions)
    write_json(ROOT / 'reports/manifest_coverage.json', coverage)


if __name__ == '__main__':
    main()
