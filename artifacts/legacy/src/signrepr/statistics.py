"""Paired cluster resampling of fixed predictions, without retraining or tuning."""
import csv

import numpy as np


UNKNOWN_GROUPS = {'', 'unknown', 'none', 'null', 'nan', 'source_recording_unknown'}


def read_predictions(path, group_key='recording_group'):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty prediction table')
    result = {}
    for row in rows:
        identifier = row['sample_id']
        if not identifier or identifier in result:
            raise ValueError('Missing or duplicate prediction sample ID')
        if row['split'] != 'test':
            raise ValueError('Only locked test predictions are accepted')
        if row[group_key].strip().lower() in UNKNOWN_GROUPS:
            raise ValueError('Unknown grouping identity; sample ID fallback is forbidden')
        gold, predicted = int(row['gold']), int(row['prediction'])
        if min(gold, predicted) < 0 or int(row['correct']) != int(gold == predicted):
            raise ValueError('Invalid class ID or inconsistent correctness column')
        result[identifier] = row
    return result


def align_predictions(tables, group_key='recording_group'):
    if not tables:
        raise ValueError('No prediction tables')
    identifiers = sorted(tables[0])
    reference = tables[0]
    for table in tables:
        if set(table) != set(reference):
            raise ValueError('Paired coverage mismatch; do not silently intersect evaluation samples')
        for identifier in identifiers:
            for key in ['gold', 'split', group_key]:
                if table[identifier][key] != reference[identifier][key]:
                    raise ValueError('Paired label/split/group identity mismatch')
    gold = np.asarray([int(reference[i]['gold']) for i in identifiers])
    groups = [reference[i][group_key] for i in identifiers]
    correct = np.asarray([[int(t[i]['correct']) for i in identifiers] for t in tables], dtype=float)
    return identifiers, gold, groups, correct


def weighted_metrics(gold, correct, weights):
    """Macro averages classes with support, matching the locked probe estimator."""
    classes, labels = np.unique(gold, return_inverse=True)
    support = np.bincount(labels, weights=weights, minlength=len(classes))
    supported = support > 0
    if not supported.any() or weights.sum() <= 0:
        raise ValueError('Empty bootstrap denominator')
    recalls = []
    for seed_correct in correct:
        numerator = np.bincount(labels, weights=weights * seed_correct, minlength=len(classes))
        recalls.append(np.mean(numerator[supported] / support[supported]))
    return {'macro_recall': np.asarray(recalls),
            'top1': (correct * weights).sum(axis=1) / weights.sum()}, int(supported.sum())


def grouped_bootstrap(gold, groups, correct, reference=None, resamples=2000, seed=42):
    gold, correct = np.asarray(gold), np.asarray(correct, dtype=float)
    if correct.ndim != 2 or correct.shape[1] != len(gold) or len(groups) != len(gold):
        raise ValueError('Prediction dimensions mismatch')
    if not len(gold) or not np.isfinite(correct).all() or not np.isin(correct, [0, 1]).all():
        raise ValueError('Invalid evaluation values')
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        if reference.shape != correct.shape or not np.isfinite(reference).all() or not np.isin(reference, [0, 1]).all():
            raise ValueError('Reference seed/sample dimensions or values mismatch')
    if resamples < 2000:
        raise ValueError('At least 2000 resamples are required')
    unique_groups, group_ids = np.unique(groups, return_inverse=True)
    if len(unique_groups) < 2:
        raise ValueError('Fewer than two independent groups; interval unavailable')
    point, class_count = weighted_metrics(gold, correct, np.ones(len(gold)))
    ref_point = weighted_metrics(gold, reference, np.ones(len(gold)))[0] if reference is not None else None
    samples = {metric: [] for metric in point}
    deltas = {metric: [] for metric in point} if reference is not None else None
    supports = []
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        # A sampled group contributes all of its rows, with the same multiplicity
        # for every trained seed and both sides of a paired comparison.
        multiplicity = np.bincount(rng.integers(len(unique_groups), size=len(unique_groups)), minlength=len(unique_groups))
        weights = multiplicity[group_ids]
        measured, support = weighted_metrics(gold, correct, weights)
        supports.append(support)
        ref = weighted_metrics(gold, reference, weights)[0] if reference is not None else None
        for metric in samples:
            samples[metric].append(float(measured[metric].mean()))
            if ref is not None:
                deltas[metric].append(float((measured[metric] - ref[metric]).mean()))
    result = {'samples': len(gold), 'groups': len(unique_groups), 'classes_with_support': class_count,
              'training_seeds': correct.shape[0], 'resamples': resamples, 'bootstrap_seed': seed,
              'interval_method': 'paired_cluster_percentile_95' if reference is not None else 'cluster_percentile_95',
              'seed_aggregation': 'arithmetic mean of metrics over all supplied trained seeds; no best-seed selection',
              'interval_scope': 'conditional on fixed trained heads and observed evaluation cohort; does not include training-seed or signer-population uncertainty',
              'bootstrap_macro_denominator': 'classes with positive resampled support, same estimator as point metric',
              'resamples_with_missing_classes': sum(s < class_count for s in supports),
              'resampled_classes_min': min(supports), 'metrics': {}}
    for metric in samples:
        values = point[metric]
        item = {'point': float(values.mean()), 'per_seed': values.tolist(),
                'training_seed_sd': float(values.std(ddof=1)) if len(values) > 1 else None,
                'ci95': np.quantile(samples[metric], [.025, .975]).tolist()}
        if ref_point is not None:
            item.update(reference_point=float(ref_point[metric].mean()),
                        reference_per_seed=ref_point[metric].tolist(),
                        delta=float((values - ref_point[metric]).mean()),
                        delta_ci95=np.quantile(deltas[metric], [.025, .975]).tolist())
        result['metrics'][metric] = item
    result['limitations'] = ['Percentile coverage is approximate; many groups do not imply many independent signers.',
                            'Groups are sampled uniformly with replacement; their complete rows are retained.',
                            'Class absence in a draw changes its macro denominator and is counted explicitly.']
    if len(unique_groups) < 30:
        result['limitations'].append('Fewer than 30 groups: intervals are descriptive and cannot establish reliable population inference.')
    return result
