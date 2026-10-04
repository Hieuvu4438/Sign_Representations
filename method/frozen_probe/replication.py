"""P004 fixed one-example replication contracts; original P003 stays immutable."""
import json
from pathlib import Path

import numpy as np

from analysis import participant_bootstrap
from baseline import support_ids
from cache import expected_centers, file_sha, sample_map
from confirmatory import METHODS, SEEDS
from metrics import evaluate


def check_gate(gate_path, config_path, repo):
    gate = json.loads(Path(gate_path).read_text())
    name = str(Path(config_path).resolve().relative_to(repo))
    if (gate['verdict'] != 'REGISTERED_FIXED_POLICY_DEV_PILOT'
            or gate['protocol'] != 'P004_v1'
            or gate['config_hashes'].get(name) != file_sha(config_path)):
        raise ValueError('P004 registration/config mismatch')
    for name, digest in gate['file_hashes'].items():
        if file_sha(repo / name) != digest:
            raise ValueError(f'Registered source/artifact changed: {name}')
    return gate


def validate_cohort(cohort, audit, expected, contract):
    indexed = sample_map(cohort)
    videos = {s['sample_id']: s for s in audit['samples']}
    if (not audit['passed'] or len(videos) != len(audit['samples'])
            or set(videos) != set(indexed) or len(cohort['classes']) != expected['classes']):
        raise ValueError('Incomplete audited replication vocabulary/sources')
    actual = {}
    for split in ('train', 'val'):
        ids = [k for k, s in indexed.items() if s['split'] == split]
        actual[split] = len(ids)
        if {indexed[k]['class_id'] for k in ids} != set(range(len(cohort['classes']))):
            raise ValueError('Every fixed class must have train and dev coverage')
    actual['windows'] = 0
    for key, sample in indexed.items():
        if any(videos[key][field] != sample[field] for field in (
                'class_id', 'split', 'participant_id', 'source_sha256')):
            raise ValueError('Audit/source split-label identity mismatch')
        count = len(expected_centers(videos[key], contract))
        if count != videos[key]['windows']:
            raise ValueError('Native window count mismatch')
        actual['windows'] += count
    if any(actual[k] != expected[k] for k in actual):
        raise ValueError('Registered source/window counts changed')
    for seed in SEEDS:
        ids = support_ids(cohort, seed, 1)
        if (len(ids) != len(cohort['classes'])
                or any(indexed[k]['split'] != 'train' for k in ids)
                or sorted(indexed[k]['class_id'] for k in ids) != list(range(len(cohort['classes'])))):
            raise ValueError('Gallery must contain exactly one train example per class')
    return indexed, videos


def quality_report(schema, rankings):
    ids, labels, people, classes = (schema[k] for k in (
        'query_ids', 'labels', 'participant_ids', 'class_ids'))
    expected = {f'{m}_s{s}' for m in METHODS for s in SEEDS}
    if (schema['split'] != 'val' or schema['gallery_budget'] != 1
            or schema['support_seeds'] != list(SEEDS)
            or not ids or len(set(ids)) != len(ids) or len(labels) != len(ids)
            or len(people) != len(ids) or classes != list(range(len(classes)))
            or not classes or set(rankings) != expected):
        raise ValueError('Invalid complete fixed-policy dev schema')
    refs = [dict(sample_id=k, label=c) for k, c in zip(ids, labels)]
    metrics, dcg = {}, {}
    for name in sorted(expected):
        rank = np.asarray(rankings[name])
        if rank.shape != (len(ids), len(classes)) or not np.issubdtype(rank.dtype, np.integer):
            raise ValueError('Invalid complete dev ranking shape/type')
        metrics[name] = evaluate([dict(sample_id=k, ranking=row.tolist())
                                  for k, row in zip(ids, rank)], refs, classes)
        position = np.argmax(rank == np.asarray(labels)[:, None], axis=1) + 1
        dcg[name] = 1 / np.log2(position + 1)
    means = {m: {k: float(np.mean([metrics[f'{m}_s{s}'][k] for s in SEEDS]))
                 for k in metrics[next(iter(metrics))]} for m in METHODS}
    delta = np.mean([dcg[f'uniform8_s{s}'] - dcg[f'full_activity_s{s}']
                     for s in SEEDS], axis=0)
    losses = {str(s): float(metrics[f'full_activity_s{s}']['dcg']
                            - metrics[f'uniform8_s{s}']['dcg']) for s in SEEDS}
    boot = participant_bootstrap(delta, people, 2000, 1701)
    interval = [v / 100 for v in boot['interval95_pp']]
    boot.update(interval95_dcg=interval, estimate_dcg=float(delta.mean()),
                limitation='conditional on fixed fresh classes/checkpoint/three galleries; five dev groups; not population equivalence')
    passed = bool(delta.mean() >= -.02 and interval[0] >= -.02
                  and all(v <= .02 for v in losses.values()))
    return dict(mean_metrics=means, metrics_by_gallery=metrics,
                uniform_minus_dense_dcg=float(delta.mean()), per_gallery_dcg_loss=losses,
                conditional_participant_bootstrap=boot, quality_screen=passed,
                margin=.02, primary='uniform8 versus full_activity; mean DCG over three fixed one-example galleries',
                new_policy_trials=0, selection='none; no best-policy re-selection',
                scope='adapted 100-class within-ASL-Citizen fresh-vocabulary dev replication; cached quality, no latency measurement')
