"""Freeze a train-selected lexical diagnostic cohort before test inference."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from _common import ROOT
from signrepr.io import atomic_text, read_jsonl, sha256, write_json, write_jsonl
from signrepr.validation import audit_rows


def order(value, seed):
    return hashlib.sha256((str(seed) + ':' + str(value)).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/manifests/asl_citizen.jsonl')
    parser.add_argument('--classes', type=int, default=200)
    parser.add_argument('--train-per-class', type=int, default=12)
    parser.add_argument('--val-per-class', type=int, default=4)
    parser.add_argument('--test-per-class', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=ROOT / 'configs/protocol_common_v1.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    rows = [row for _, row in read_jsonl(args.manifest) if row['source_exists']]
    train_counts = Counter(row['lexical_id'] for row in rows if row['official_split'] == 'train')
    eligible = [label for label, count in train_counts.items() if count >= args.train_per_class]
    labels = sorted(sorted(eligible, key=lambda label: order(label, args.seed))[:args.classes])
    by_label_split = defaultdict(list)
    for row in rows:
        if row['lexical_id'] in labels:
            by_label_split[(row['lexical_id'], row['official_split'])].append(row)
    selected = []
    limits = {'train': args.train_per_class, 'val': args.val_per_class, 'test': args.test_per_class}
    for label in labels:
        for split, limit in limits.items():
            selected.extend(sorted(by_label_split[(label, split)], key=lambda row: order(row['sample_id'], args.seed))[:limit])
    audit = audit_rows(selected)
    if audit['recording_overlaps'] or audit['signer_overlaps']:
        raise ValueError('Controlled cohort must be recording and signer disjoint')
    counts = dict(Counter(row['official_split'] for row in selected))
    if args.dry_run:
        print(json.dumps({'classes': len(labels), 'split_counts': counts, 'selection_uses': 'train label counts and deterministic IDs only'}))
        return
    if args.output.exists():
        raise ValueError('Protocol already exists; never overwrite a locked protocol')
    manifest = ROOT / 'data/manifests/common_v1.jsonl'
    write_jsonl(manifest, selected)
    split_hashes = {}
    for split in ['train', 'val', 'test']:
        path = ROOT / ('data/splits/common_v1_' + split + '.jsonl')
        write_jsonl(path, [row for row in selected if row['official_split'] == split])
        split_hashes[split] = sha256(path)
    write_json(ROOT / 'data/splits/common_v1_labels.json', {'labels': labels, 'mapping': {label: index for index, label in enumerate(labels)}, 'fit_scope': 'train_only'})
    protocol = {
        'protocol_version': 'common_v1', 'locked': True, 'task': 'common_lexical_diagnostic',
        'population': 'ASL Citizen train-selected 200-class cohort; official signer-disjoint splits; capped clips per class',
        'scientific_claim_role': 'pipeline validation and lexical retention reference; no novelty or full benchmark reproduction',
        'source_manifest': str(args.manifest.relative_to(ROOT)), 'source_manifest_sha256': sha256(args.manifest),
        'manifest': str(manifest.relative_to(ROOT)), 'manifest_sha256': sha256(manifest),
        'split_sha256': split_hashes, 'label_mapping': 'data/splits/common_v1_labels.json',
        'label_mapping_sha256': sha256(ROOT / 'data/splits/common_v1_labels.json'),
        'class_selection': {'source': 'train_only', 'minimum_train_count': args.train_per_class,
                            'class_count': len(labels), 'selection_seed': args.seed, 'ranking': 'SHA256(seed:label)',
                            'clips_per_class_cap': limits},
        'split_counts': counts, 'primary_metric': 'macro_recall', 'secondary_metrics': ['top1', 'top5', 'coverage'],
        'primary_comparison': 'B03_shallow_temporal_vs_B02_mean_pool_linear',
        'secondary_comparisons_policy': 'exploratory', 'seeds': [0, 1, 2],
        'model_config': 'configs/signrep.yaml', 'model_config_sha256': sha256(ROOT / 'configs/signrep.yaml'),
        'checkpoint_sha256': 'f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b',
        'backbone_commit': '06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2',
        'feature_directory': 'features/signrep_common_v1', 'pooling': 'mean_over_valid_nonpadded_windows_then_L2',
        'exclusions': ['missing source video', 'decode failure', 'short clips requiring padding'],
        'normalization_fit': 'per-sample L2 only; no test population fit', 'supervision': 'official train lexical labels only',
        'tuning': {'learning_rates': [0.0001, 0.001, 0.01], 'capacities': [64, 128], 'epochs': 100,
                   'selection_metric': 'val_macro_recall', 'tie_policy': 'first_registered_config',
                   'test_tuning_allowed': False, 'optimizer': 'AdamW', 'weight_decay': 0.01},
        'statistics': {'resamples': 2000, 'seed': 42, 'group_key': 'recording_group', 'interval': 'paired_percentile_95'},
        'pretraining_overlap': 'UNKNOWN', 'adaptation_allowed': False,
    }
    atomic_text(args.output, yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True))
    write_json(ROOT / 'state/protocol_common_v1.lock.json', {'path': str(args.output.relative_to(ROOT)), 'sha256': sha256(args.output),
               'test_inference_has_run': False, 'immutability': 'Any change requires a new protocol version and separate results.'})
    write_json(ROOT / 'reports/split_audit_common_v1.json', {**audit, 'manifest_sha256': sha256(manifest)})
    print(json.dumps({'protocol': str(args.output), 'classes': len(labels), 'split_counts': counts, 'sha256': sha256(args.output)}))


if __name__ == '__main__':
    main()
