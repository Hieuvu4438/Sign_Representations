"""Compact numerical records; no encoder, fitting, ranking or metric selection."""
import numpy as np
from bounded_artifacts import array_spec

FAMILIES = ('residual', 'total', 'shuffled')
METRIC_COLUMNS = ('dcg', 'top1', 'top5', 'top10', 'top20', 'mrr')


def compact_specs(dimension, prediction_rows, class_count):
    if type(class_count) is not int or not 1 <= class_count <= 32768:
        raise ValueError('Class count cannot be represented as int16 zero-based indices')
    specs = {'mean.npy': array_spec((dimension,), '<f8'),
             'ranks.npy': array_spec((prediction_rows, class_count), '<i2'),
             'row_metrics.npy': array_spec((prediction_rows, len(METRIC_COLUMNS)), '<f8')}
    for family in FAMILIES:
        specs[family + '_covariance_upper.npy'] = array_spec((dimension * (dimension + 1) // 2,), '<f8')
        specs[family + '_eigenvalues.npy'] = array_spec((dimension,), '<f8')
    return specs


def pack_geometry(fitted):
    mean = np.asarray(fitted['mean'])
    if mean.dtype.str != '<f8' or mean.ndim != 1 or not len(mean) or not np.isfinite(mean).all():
        raise ValueError('Invalid FP64 fitted mean')
    d = len(mean)
    expected = {'mean'} | {f + suffix for f in FAMILIES for suffix in ('_covariance', '_eigenvalues')}
    derived = {f + '_whitening' for f in FAMILIES}
    if not expected <= set(fitted) or set(fitted) - expected - derived:
        raise ValueError('Unexpected or missing fitted geometry keys')
    result = {'mean.npy': mean}
    for family in FAMILIES:
        cov = np.asarray(fitted[family + '_covariance'])
        eigenvalues = np.asarray(fitted[family + '_eigenvalues'])
        # Exact symmetry is required: never average away original numerical values.
        if cov.dtype.str != '<f8' or cov.shape != (d, d) or not np.isfinite(cov).all() or cov.tobytes() != cov.T.tobytes():
            raise ValueError('Covariance must be finite and exactly symmetric FP64')
        if eigenvalues.dtype.str != '<f8' or eigenvalues.shape != (d,) or not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
            raise ValueError('Invalid saved eigenvalues')
        result[family + '_covariance_upper.npy'] = cov[np.triu_indices(d)]
        result[family + '_eigenvalues.npy'] = eigenvalues
    # Whitening matrices are derived from covariance, intentionally omitted.
    # Recomputed eigenvectors need not be bitwise identical across environments.
    return result


def unpack_covariance(upper, dimension):
    a = np.asarray(upper)
    if type(dimension) is not int or dimension <= 0 or a.dtype.str != '<f8' or a.shape != (dimension * (dimension + 1) // 2,) or not np.isfinite(a).all():
        raise ValueError('Invalid upper-triangle covariance record')
    result = np.empty((dimension, dimension), dtype='<f8')
    indices = np.triu_indices(dimension)
    result[indices] = a
    result[(indices[1], indices[0])] = a
    return result


def validate_rank_arrays(ranks, row_metrics, class_count):
    r, m = np.asarray(ranks), np.asarray(row_metrics)
    if type(class_count) is not int or not 1 <= class_count <= 32768:
        raise ValueError('Unsupported class count')
    if r.dtype.str != '<i2' or r.ndim != 2 or r.shape[1] != class_count or not len(r):
        raise ValueError('Invalid ranking shape/dtype')
    if not np.array_equal(np.sort(r, axis=1), np.broadcast_to(np.arange(class_count), r.shape)):
        raise ValueError('Every ranking must be a complete zero-based class-index permutation')
    if m.dtype.str != '<f8' or m.shape != (len(r), len(METRIC_COLUMNS)) or not np.isfinite(m).all() or np.any(m < 0) or np.any(m > 1):
        raise ValueError('Invalid saved per-row metric array')
    return dict(prediction_rows=len(r), class_count=class_count, metric_columns=list(METRIC_COLUMNS))


def ranking_metadata(classes, modes, support_seeds, query_ids, labels, participants):
    """One shared metadata row per query; indices map back to original class IDs."""
    if not classes or len(set(classes)) != len(classes) or not all(type(c) is int for c in classes):
        raise ValueError('Class vocabulary must contain unique integer IDs')
    if not modes or len(set(modes)) != len(modes) or not all(isinstance(m, str) and m for m in modes):
        raise ValueError('Modes must be unique nonempty strings')
    if not support_seeds or len(set(support_seeds)) != len(support_seeds) or not all(type(s) is int for s in support_seeds):
        raise ValueError('Support seeds must be unique integers')
    if not query_ids or len(set(query_ids)) != len(query_ids) or not all(isinstance(k, str) and k for k in query_ids):
        raise ValueError('Queries must contain unique nonempty IDs')
    if len(labels) != len(query_ids) or len(participants) != len(query_ids) or any(type(label) is not int or label not in classes for label in labels):
        raise ValueError('Invalid shared query references')
    if any(not isinstance(p, str) or not p for p in participants):
        raise ValueError('Missing participant ID')
    return dict(format_version=1, class_ids=list(classes), modes=list(modes), support_seeds=list(support_seeds),
                query_ids=list(query_ids), labels=list(labels), participants=list(participants),
                row_order=['mode', 'support_seed', 'query_id'],
                prediction_rows=len(modes) * len(support_seeds) * len(query_ids),
                metric_columns=list(METRIC_COLUMNS), ranking_values='zero-based indices into class_ids')


def pack_predictions(rows, metadata):
    """Validate the complete Cartesian table before converting IDs to int16 indices."""
    checked = ranking_metadata(metadata['class_ids'], metadata['modes'], metadata['support_seeds'],
                               metadata['query_ids'], metadata['labels'], metadata['participants'])
    if metadata != checked or len(rows) != metadata['prediction_rows']:
        raise ValueError('Prediction metadata or row count mismatch')
    classes = metadata['class_ids']
    if len(classes) > 32768:
        raise ValueError('Too many classes for int16 indices')
    indexed = {}
    fields = {'mode', 'support_seed', 'sample_id', 'label', 'participant_id', 'ranking'} | set(METRIC_COLUMNS)
    for row in rows:
        if set(row) != fields:
            raise ValueError('Unexpected prediction fields; no silent dropping')
        if type(row['support_seed']) is not int or type(row['label']) is not int:
            raise ValueError('Prediction seed/label must be integer IDs')
        key = (row['mode'], row['support_seed'], row['sample_id'])
        if key in indexed:
            raise ValueError('Duplicate prediction row')
        indexed[key] = row
    class_index = {c: i for i, c in enumerate(classes)}
    ranks, metrics = [], []
    for mode in metadata['modes']:
        for seed in metadata['support_seeds']:
            for qi, query_id in enumerate(metadata['query_ids']):
                key = (mode, seed, query_id)
                if key not in indexed:
                    raise ValueError('Missing prediction from declared Cartesian table')
                row = indexed[key]
                ranking = row['ranking']
                if row['label'] != metadata['labels'][qi] or row['participant_id'] != metadata['participants'][qi]:
                    raise ValueError('Changed query reference')
                if len(ranking) != len(classes) or any(type(c) is not int for c in ranking) or set(ranking) != set(classes):
                    raise ValueError('Ranking must exactly permute declared class IDs')
                if any(type(row[name]) not in (int, float) for name in METRIC_COLUMNS):
                    raise ValueError('Saved metrics must be numeric without implicit type conversion')
                ranks.append([class_index[c] for c in ranking])
                metrics.append([row[name] for name in METRIC_COLUMNS])
    r, m = np.asarray(ranks, dtype='<i2'), np.asarray(metrics, dtype='<f8')
    validate_rank_arrays(r, m, len(classes))
    return {'ranks.npy': r, 'row_metrics.npy': m}


def unpack_predictions(ranks, row_metrics, metadata):
    checked = ranking_metadata(metadata['class_ids'], metadata['modes'], metadata['support_seeds'],
                               metadata['query_ids'], metadata['labels'], metadata['participants'])
    if checked != metadata:
        raise ValueError('Prediction metadata mismatch')
    validate_rank_arrays(ranks, row_metrics, len(metadata['class_ids']))
    if len(ranks) != metadata['prediction_rows']:
        raise ValueError('Incomplete Cartesian ranking table')
    rows, index = [], 0
    for mode in metadata['modes']:
        for seed in metadata['support_seeds']:
            for qi, query_id in enumerate(metadata['query_ids']):
                rows.append(dict(mode=mode, support_seed=seed, sample_id=query_id,
                                 label=metadata['labels'][qi], participant_id=metadata['participants'][qi],
                                 ranking=[metadata['class_ids'][int(c)] for c in ranks[index]],
                                 **{name: float(value) for name, value in zip(METRIC_COLUMNS, row_metrics[index])}))
                index += 1
    return rows
