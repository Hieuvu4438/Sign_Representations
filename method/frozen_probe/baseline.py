"""Train-only deterministic controls, complete class rankings, ID-based scoring."""
import numpy as np

from cache import sample_map
from metrics import evaluate, ranked_classes
from pooling import gaussian_projection, pool_controls
from probes import fit_ridge, nearest_class_scores, normalize_rows, predict_ridge

VARIANTS = ('mean', 'max', 'activity', 'ordered', 'shuffled', 'mean_projected')


def pool_cache(reader, cohort, projection_seed=1337):
    samples = sample_map(cohort)
    ids = sorted(samples)
    dim = reader.contract['output_dim']
    projection = gaussian_projection(4 * dim, dim, projection_seed)
    matrices = {name: [] for name in VARIANTS}
    metadata = {}
    for key in ids:
        features, activity, centers = reader.load(key)
        vectors, details = pool_controls(features, centers, activity, samples[key]['source_sha256'], projection)
        for name in VARIANTS:
            matrices[name].append(vectors[name])
        metadata[key] = details
    return ids, {name: normalize_rows(np.stack(rows)) for name, rows in matrices.items()}, projection, metadata


def support_ids(cohort, seed, budget):
    indexed = sample_map(cohort)
    grouped = cohort['supports'][str(seed)][str(budget)]
    if set(grouped) != set(cohort['classes']):
        raise ValueError('Support class coverage differs')
    selected = []
    for c, gloss in enumerate(cohort['classes']):
        ids = grouped[gloss]
        if len(ids) != budget or len(set(ids)) != budget:
            raise ValueError('Support budget differs')
        for key in ids:
            s = indexed[key]
            if s['split'] != 'train' or s['class_id'] != c:
                raise ValueError('Support uses dev or incorrect class')
        selected.extend(ids)
    if len(selected) != len(set(selected)):
        raise ValueError('Repeated support source')
    return selected


def run_controls(cohort, ids, matrices, seeds, budgets, alpha=1.0):
    indexed = sample_map(cohort)
    if len(ids) != len(set(ids)) or set(ids) != set(indexed) or set(matrices) != set(VARIANTS):
        raise ValueError('Incomplete pooled source/control coverage')
    shapes = {np.asarray(x).shape for x in matrices.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2 or next(iter(shapes))[0] != len(ids):
        raise ValueError('Unequal pooling capacity or source count')
    # Require per-sample normalization; never fit a dev-dependent transform.
    for x in matrices.values():
        if not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-10, rtol=1e-10):
            raise ValueError('Head input is nonfinite or not normalized')
    position = {key: i for i, key in enumerate(ids)}
    query_ids = sorted(key for key, s in indexed.items() if s['split'] == 'val')
    queries = [position[key] for key in query_ids]
    references = [{'sample_id': key, 'label': indexed[key]['class_id']} for key in query_ids]
    classes = list(range(len(cohort['classes'])))
    results, rankings = [], {}
    for seed in seeds:
        for budget in budgets:
            support = support_ids(cohort, seed, budget)
            rows = [position[key] for key in support]
            labels = np.array([indexed[key]['class_id'] for key in support], dtype=np.int64)
            for variant in VARIANTS:
                x = matrices[variant]
                model = fit_ridge(x[rows], labels, len(classes), alpha)
                scores = {'ridge': predict_ridge(model, x[queries])}
                if variant in ('mean', 'max', 'activity'):
                    scores['nearest'] = nearest_class_scores(x[rows], labels, x[queries], len(classes))
                for head, values in scores.items():
                    name = f'{variant}_{head}_s{seed}_n{budget}'
                    predictions = [{'sample_id': key, 'ranking': ranked_classes(value.tolist(), classes)}
                                   for key, value in zip(query_ids, values)]
                    measured = evaluate(predictions, references, classes)
                    rankings[name] = np.array([p['ranking'] for p in predictions], dtype=np.int16)
                    results.append({'name': name, 'variant': variant, 'head': head, 'support_seed': seed,
                                    'labels_per_class': budget, 'support_ids': support, 'metrics': measured})
    return query_ids, results, rankings
