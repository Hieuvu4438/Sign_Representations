"""Registered source-only native-window policies and exact class lookup."""
import hashlib
import numpy as np

from analysis import participant_bootstrap
from baseline import support_ids
from cache import sample_map
from metrics import evaluate, ranked_classes
from pooling import activity_mean
from probes import nearest_class_scores, normalize_rows

POLICIES = ('uniform4', 'uniform8', 'random4_s2718', 'random4_s31415',
            'random8_s2718', 'random8_s31415')
METHODS = ('full_mean', 'full_activity') + POLICIES


def selected_indices(policy, windows, source_sha256):
    if (type(windows) is not int or windows < 1 or policy not in METHODS
            or len(source_sha256) != 64
            or any(c not in '0123456789abcdef' for c in source_sha256)):
        raise ValueError('Invalid policy/window/source identity')
    if policy.startswith('full_'):
        return list(range(windows))
    step = 2 if '4' == policy.split('_')[0][-1] else 4
    uniform = list(range(0, windows, step))
    if policy.startswith('uniform'):
        return uniform
    seed = int(policy.split('_s')[1])
    digest = hashlib.sha256(f'{seed}:{source_sha256}'.encode('ascii')).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
    return sorted(rng.choice(windows, size=len(uniform), replace=False).tolist())


def pooled_vector(features, activity, policy, indices):
    x = np.asarray(features)
    a = np.asarray(activity)
    if (policy not in METHODS or not indices or len(set(indices)) != len(indices)
            or indices != sorted(indices)
            or any(type(i) is not int or i < 0 or i >= len(x) for i in indices)):
        raise ValueError('Invalid selected native indices')
    x = x[indices].astype(np.float64)
    a = a[indices].astype(np.float64)
    value = x.mean(axis=0) if policy == 'full_mean' else activity_mean(x, a)[0]
    return normalize_rows(value[None])[0]


def pool_policies(reader, cohort):
    samples = sample_map(cohort)
    ids = sorted(samples)
    rows = {method: [] for method in METHODS}
    selections = {}
    counts = {method: 0 for method in METHODS}
    for key in ids:
        features, activity, centers = reader.load(key)
        selected = {}
        for method in METHODS:
            indices = selected_indices(method, len(centers), samples[key]['source_sha256'])
            rows[method].append(pooled_vector(features, activity, method, indices))
            selected[method] = indices
            counts[method] += len(indices)
        selections[key] = selected
    return ids, {m: np.stack(x) for m, x in rows.items()}, selections, counts


def lookup(cohort, ids, matrices, seeds, budget=8):
    indexed = sample_map(cohort)
    if len(set(ids)) != len(ids) or set(ids) != set(indexed) or set(matrices) != set(METHODS):
        raise ValueError('Incomplete lookup source/policy coverage')
    shape = {np.asarray(x).shape for x in matrices.values()}
    if len(shape) != 1 or len(next(iter(shape))) != 2 or next(iter(shape))[0] != len(ids):
        raise ValueError('Unequal policy feature capacity')
    positions = {key: i for i, key in enumerate(ids)}
    query_ids = sorted(k for k, s in indexed.items() if s['split'] == 'val')
    query_rows = [positions[k] for k in query_ids]
    classes = list(range(len(cohort['classes'])))
    references = [{'sample_id': k, 'label': indexed[k]['class_id']} for k in query_ids]
    results, rankings = [], {}
    for seed in seeds:
        supports = support_ids(cohort, seed, budget)
        train_rows = [positions[k] for k in supports]
        labels = np.array([indexed[k]['class_id'] for k in supports], dtype=np.int64)
        for method, x in matrices.items():
            values = nearest_class_scores(x[train_rows], labels, x[query_rows], len(classes))
            predictions = [{'sample_id': k, 'ranking': ranked_classes(v.tolist(), classes)}
                           for k, v in zip(query_ids, values)]
            name = f'{method}_s{seed}'
            rankings[name] = np.array([p['ranking'] for p in predictions], dtype=np.int16)
            results.append(dict(name=name, method=method, support_seed=seed,
                                metrics=evaluate(predictions, references, classes)))
    return query_ids, results, rankings


def quality_analysis(query_ids, results, rankings, cohort, counts, seeds):
    samples = sample_map(cohort)
    indexed = {r['name']: r for r in results}
    expected = {f'{m}_s{s}' for m in METHODS for s in seeds}
    if len(indexed) != len(results) or set(indexed) != expected or set(rankings) != expected:
        raise ValueError('Incomplete policy/seed results')
    classes = len(cohort['classes'])
    labels = np.array([samples[k]['class_id'] for k in query_ids])
    participants = [samples[k]['participant_id'] for k in query_ids]
    dcg = {}
    for name, rank in rankings.items():
        rank = np.asarray(rank)
        if rank.shape != (len(query_ids), classes):
            raise ValueError('Incomplete ranking shape')
        position = np.argmax(rank == labels[:, None], axis=1) + 1
        dcg[name] = 1 / np.log2(position + 1)
    primary = {m: float(np.mean([indexed[f'{m}_s{s}']['metrics']['dcg'] for s in seeds])) for m in METHODS}
    summary = {}
    for method in POLICIES:
        delta = np.mean([dcg[f'{method}_s{s}'] - dcg[f'full_activity_s{s}'] for s in seeds], axis=0)
        interval = participant_bootstrap(delta, participants, 2000, 1701)
        loss = primary['full_activity'] - primary[method]
        reduction = counts['full_activity'] / counts[method]
        summary[method] = dict(primary_dcg=primary[method], dcg_loss=loss,
            per_support_seed_dcg_loss={str(s): indexed[f'full_activity_s{s}']['metrics']['dcg'] -
                                       indexed[f'{method}_s{s}']['metrics']['dcg'] for s in seeds},
            encoded_windows=counts[method], window_reduction_factor=reduction,
            quality_count_screen=loss <= .02 + 1e-12 and counts[method] * 2 <= counts['full_activity'],
            paired_participant_interval95_dcg_difference=[v / 100 for v in interval['interval95_pp']],
            participant_count=interval['participant_count'], bootstrap_seed=1701, bootstrap_replicates=2000)
    return dict(primary_dcg=primary, policies=summary, dense_windows=counts['full_activity'],
                scope='cached quality simulation; no measured encoder/latency savings',
                uncertainty='five actual participant groups; exploratory/conditional, not selection-adjusted')
