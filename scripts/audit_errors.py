"""Prepare a fixed error sample and audit technical evidence; expert review stays explicit."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import yaml

from _common import ROOT
from signrepr.io import atomic_text, read_jsonl, sha256, write_json
from signrepr.statistics import align_predictions, read_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, nargs='+', required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/protocol_common_v1.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/common_v1_error_audit')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    statistics = json.loads((ROOT / 'configs/statistics_common_v1.json').read_text())
    if sha256(args.config) != statistics['protocol_sha256']:
        raise ValueError('Protocol changed after error-sampling registration')
    protocol = yaml.safe_load(args.config.read_text())
    tables = [read_predictions(path) for path in args.predictions]
    identifiers, _, _, _ = align_predictions(tables)
    baseline = statistics['error_audit']['baseline']
    if any({row['baseline'] for row in table.values()} != {baseline} for table in tables):
        raise ValueError('Error audit baseline differs from registration')
    seeds = [next(iter({row['seed'] for row in table.values()})) for table in tables]
    if seeds != ['0', '1', '2'] or any(len({row['seed'] for row in table.values()}) != 1 for table in tables):
        raise ValueError('Supply the registered ordered seeds [0,1,2]')
    pool = [identifier for identifier in identifiers if any(table[identifier]['correct'] == '0' for table in tables)]
    selected = sorted(pool, key=lambda identifier: hashlib.sha256(('42:' + identifier).encode()).hexdigest())[:30]
    if args.dry_run:
        print(json.dumps({'error_pool': len(pool), 'selected': len(selected), 'expert_review': 'PENDING'}))
        return
    if args.output.exists():
        raise ValueError('Audit output exists; preserve prior review evidence')
    manifest = {row['sample_id']: row for _, row in read_jsonl(ROOT / protocol['manifest'])}
    features = ROOT / protocol['feature_directory']
    index = {row['sample_id']: row for _, row in read_jsonl(features / 'index.jsonl')}
    rows = []
    for identifier in selected:
        item, cached = manifest[identifier], index[identifier]
        source_path = Path(item['source_root']) / item['relative_path']
        shard_path = Path(cached['shard'])
        metadata = json.loads(shard_path.with_suffix('.json').read_text())
        measurements = metadata['measurements']
        with np.load(shard_path, allow_pickle=False) as shard:
            tokens = shard['embeddings']
            times = np.stack([shard['window_start_sec'], shard['window_end_sec']], axis=1)
            valid = shard['valid_mask'] & shard['valid_frame_mask'].all(axis=1)
            finite = bool(np.isfinite(tokens).all() and np.isfinite(times).all())
            timing = bool((times >= 0).all() and (times <= measurements['duration_sec'] + 1e-6).all())
            valid_count = int(valid.sum())
        source_exists = source_path.is_file()
        checksum = sha256(shard_path) == metadata['shard_sha256']
        rows.append({'sample_id': identifier, 'source_path': str(source_path), 'signer_id': item['signer_id'],
                     'recording_group': item['recording_group'], 'lexical_id': item['lexical_id'],
                     'gold': tables[0][identifier]['gold'],
                     **{'prediction_seed' + seed: table[identifier]['prediction'] for seed, table in zip(seeds, tables)},
                     'source_exists': source_exists, 'decode_status': cached['status'],
                     'feature_checksum_match': checksum,
                     'finite_features_and_times': finite, 'timestamps_in_video': timing,
                     'valid_windows': valid_count, 'fps': measurements['fps_observed'],
                     'technical_review': 'PASS' if cached['status'] == 'SUCCESS' and source_exists and checksum and finite and timing and valid_count > 0 else 'FAIL',
                     'video_identity_visual_review': 'UNREVIEWED', 'lexical_gold_ambiguity': 'UNREVIEWED_REQUIRES_ASL_EXPERT',
                     'manual_lexical_mismatch': 'UNREVIEWED_REQUIRES_ASL_EXPERT',
                     'signer_domain_explanation': 'UNRESOLVED', 'landmarks': 'NA_SIGNREP_RGB',
                     'grammar_scope': 'NA_NO_GRAMMAR_GOLD', 'text_prior': 'NA_NO_TEXT_INPUT', 'review_notes': ''})
    args.output.mkdir(parents=True)
    stream = io.StringIO()
    fields = list(rows[0]) if rows else ['sample_id', 'technical_review', 'review_notes']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(args.output / 'review.csv', stream.getvalue())
    write_json(args.output / 'report.json', {'status': 'TECHNICAL_AUDIT_COMPLETE_EXPERT_REVIEW_PENDING',
               'error_pool': len(pool), 'selected': len(selected), 'selection': statistics['error_audit'],
               'selected_sample_ids': selected, 'technical_failures': sum(row['technical_review'] == 'FAIL' for row in rows),
               'linguistic_reviews_completed': 0, 'prediction_inputs': [{'path': str(p), 'sha256': sha256(p)} for p in args.predictions],
               'protocol_sha256': sha256(args.config), 'implementation_sha256': sha256(Path(__file__)),
               'limitations': 'Automated technical audit does not determine whether signing or the lexical gold label is correct.'})
    print(json.dumps({'output': str(args.output), 'selected': len(selected), 'expert_review': 'PENDING'}))


if __name__ == '__main__':
    main()
