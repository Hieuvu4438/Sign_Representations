"""Fixed saved-geometry transfer evaluation; no target fitting or selection."""
import math
import numpy as np

from axis_spectrum import transforms, unit_rows
from analysis import participant_bootstrap
from compact_geometry_artifacts import METRIC_COLUMNS, ranking_metadata
from dictionary_metrics import evaluate_dictionary
from metrics import ranked_classes, score_ranking

EVAL_MODES = ('RR', 'TR', 'RT', 'TT', 'raw', 'centered')


def frozen_geometry(values, mean, bases, spectra):
    x = np.asarray(values)
    mu = np.asarray(mean)
    if (x.dtype != np.float64 or x.ndim != 2 or not len(x)
            or mu.dtype != np.float64 or mu.shape != (x.shape[1],)
            or not np.isfinite(x).all() or not np.isfinite(mu).all()
            or set(bases) != {'R', 'T'} or set(spectra) != {'R', 'T'}):
        raise ValueError('Invalid fixed geometry input')
    for letter in ('R', 'T'):
        v, lam = bases[letter], spectra[letter]
        if (v.dtype != np.float64 or v.shape != (x.shape[1], x.shape[1])
                or lam.dtype != np.float64 or lam.shape != (x.shape[1],)
                or not np.isfinite(v).all() or not np.isfinite(lam).all()
                or np.any(lam <= 1e-12) or np.any(np.diff(lam) < 0)
                or np.max(np.abs(v.T @ v - np.eye(x.shape[1]))) > 1e-10):
            raise ValueError('Saved basis/spectrum invalid; no refactor or fallback')
    centered = x - mu
    w = transforms(bases, spectra)
    result = {mode: unit_rows(centered @ w[mode]) for mode in EVAL_MODES[:4]}
    result['raw'], result['centered'] = unit_rows(x), unit_rows(centered)
    return result


def validate_dictionary(items, dictionary):
    classes, seeds, queries = (dictionary[k] for k in ('class_ids', 'support_seeds', 'query_ids'))
    if (not classes or classes != list(range(len(classes))) or len(classes) > 32768
            or not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int for s in seeds)
            or set(dictionary['supports']) != set(map(str, seeds))
            or not queries or queries != sorted(set(queries))):
        raise ValueError('Invalid complete dictionary axes')
    indexed = {x['sample_id']: x for x in items}
    if len(indexed) != len(items) or list(indexed) != sorted(indexed):
        raise ValueError('Extraction IDs must be unique and sorted')
    required = set(queries)
    for seed in seeds:
        ids = dictionary['supports'][str(seed)]
        if len(ids) != len(classes) or len(set(ids)) != len(ids) or set(ids) & set(queries):
            raise ValueError('Incomplete, duplicate or query-contaminated gallery')
        for class_id, key in zip(classes, ids):
            if key not in indexed or indexed[key]['split'] != 'train' or indexed[key]['class_id'] != class_id:
                raise ValueError('Gallery label/order/split mismatch')
        required.update(ids)
    if required != set(indexed):
        raise ValueError('Feature IDs differ from exact query/gallery union')
    if any(indexed[k]['split'] != 'val' or indexed[k]['class_id'] not in classes for k in queries):
        raise ValueError('Query split/label mismatch')
    supports = required - set(queries)
    for field in ('parent_id', 'signer_id'):
        if {indexed[k][field] for k in queries} & {indexed[k][field] for k in supports}:
            raise ValueError('Train/query source or official signer overlap')
    labels = [indexed[k]['class_id'] for k in queries]
    if set(labels) != set(classes):
        raise ValueError('Macro requires support for every class')
    people = [str(indexed[k]['signer_id']) for k in queries]
    if len(set(people)) < 2:
        raise ValueError('Grouped uncertainty requires at least two official signer IDs')
    return ranking_metadata(classes, list(EVAL_MODES), seeds, queries, labels, people)


def score_frozen_dictionary(projected, items, dictionary, gain_floor=.005, reps=2000, seed=1701):
    metadata = validate_dictionary(items, dictionary)
    if set(projected) != set(EVAL_MODES):
        raise ValueError('Missing fixed mode; no secondary selection')
    positions = {x['sample_id']: i for i, x in enumerate(items)}
    classes, seeds, queries = metadata['class_ids'], metadata['support_seeds'], metadata['query_ids']
    refs = [dict(sample_id=k, label=l) for k, l in zip(queries, metadata['labels'])]
    nrows = metadata['prediction_rows']
    scores_all = np.empty((nrows, len(classes)), dtype='<f8')
    ranks = np.empty((nrows, len(classes)), dtype='<i2')
    row_metrics = np.empty((nrows, len(METRIC_COLUMNS)), dtype='<f8')
    metrics, diagnostics, dcg = {}, {}, {}
    offset = 0
    for mode in EVAL_MODES:
        z = projected[mode]
        if (z.dtype != np.float64 or z.ndim != 2 or len(z) != len(items)
                or not np.isfinite(z).all()
                or np.max(np.abs(np.linalg.norm(z, axis=1) - 1)) > 1e-12):
            raise ValueError('Nonfinite, incomplete or nonunit frozen projection')
        metrics[mode], diagnostics[mode], by_seed = {}, {}, []
        q = z[[positions[k] for k in queries]]
        for support_seed in seeds:
            g = z[[positions[k] for k in dictionary['supports'][str(support_seed)]]]
            scores = q @ g.T
            predictions, rows, tie_pairs, min_gap = [], [], 0, None
            for qi, key in enumerate(queries):
                ordered = ranked_classes(scores[qi].tolist(), classes)
                gaps = np.diff(np.sort(scores[qi]))
                tie_pairs += int(np.count_nonzero(gaps == 0))
                if len(gaps):
                    row_gap = float(np.min(gaps))
                    min_gap = row_gap if min_gap is None else min(min_gap, row_gap)
                measured = score_ranking(ordered, metadata['labels'][qi], classes)
                rows.append([measured[k] for k in METRIC_COLUMNS])
                predictions.append(dict(sample_id=key, ranking=ordered))
                ranks[offset + qi] = ordered  # declared vocabulary is exactly 0..C-1
            scores_all[offset:offset + len(queries)] = scores
            row_metrics[offset:offset + len(queries)] = rows
            metrics[mode][str(support_seed)] = evaluate_dictionary(predictions, refs, classes)
            diagnostics[mode][str(support_seed)] = dict(exact_adjacent_tie_pairs=tie_pairs,
                minimum_adjacent_score_gap=min_gap, tie_rule='descending score, ascending class ID')
            by_seed.append(np.asarray(rows)[:, 0])
            offset += len(queries)
        dcg[mode] = np.mean(by_seed, axis=0)
    assert offset == nrows
    per_gallery = {str(s): metrics['RR'][str(s)]['micro']['dcg'] - metrics['TR'][str(s)]['micro']['dcg'] for s in seeds}
    delta = dcg['RR'] - dcg['TR']
    effect = math.fsum(delta) / len(delta)
    bootstrap = participant_bootstrap(delta, metadata['participants'], reps, seed)
    uncertainty = dict(estimate_DCG=bootstrap['estimate_pp'] / 100,
        interval95_DCG=[x / 100 for x in bootstrap['interval95_pp']],
        signer_count=bootstrap['participant_count'], replicates=reps, seed=seed,
        unit='official signer ID', weighting='micro over all clips of repeated sampled groups',
        limitation='Conditional on the three fixed galleries, encoder, base parameters and collections; no support/seed bootstrap or population claim.')
    averages = {m: {kind: {metric: math.fsum(metrics[m][str(s)][kind][metric] for s in seeds) / len(seeds)
        for metric in METRIC_COLUMNS} for kind in ('micro', 'macro')} for m in EVAL_MODES}
    summary = dict(metrics=metrics, mean_over_fixed_galleries=averages,
        primary=dict(comparison='RR_minus_TR', delta_DCG=effect, per_gallery_delta_DCG=per_gallery,
            practical_gain_floor=gain_floor, every_gallery_nonnegative_required=True,
            every_gallery_nonnegative_observed=bool(min(per_gallery.values()) >= 0),
            passed=bool(effect >= gain_floor and min(per_gallery.values()) >= 0), conditional_bootstrap=uncertainty),
        numerical_diagnostics=diagnostics, secondary='All six modes and metrics reported; no secondary winner or inferential selection',
        supervision='1313 labeled ASL Citizen clips/100 base classes for geometry plus100 MS-ASL train gallery labels per seed',
        scope='Adapted MS-ASL validation dictionary task; no official classifier benchmark, novel algorithm, encoder-unseen, cross-language or causal nuisance claim')
    return {'scores.npy': scores_all, 'ranks.npy': ranks, 'row_metrics.npy': row_metrics}, metadata, summary
