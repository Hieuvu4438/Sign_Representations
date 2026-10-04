"""P001_v3 exploratory analysis from complete frozen rankings; no selection edits."""
import numpy as np

from metrics import evaluate

METHODS = ('mean_ridge', 'max_ridge', 'activity_ridge', 'ordered_ridge', 'shuffled_ridge',
           'mean_projected_ridge', 'mean_nearest', 'max_nearest', 'activity_nearest')
BASELINES = tuple(m for m in METHODS if m not in ('ordered_ridge', 'shuffled_ridge'))
SEEDS = (1701, 1702, 1703)
BUDGETS = (1, 2, 4, 8)


def participant_bootstrap(paired_clip_differences, participants, reps=2000, seed=1701):
    differences = np.asarray(paired_clip_differences, dtype=np.float64)
    participants = np.asarray(participants)
    if (differences.ndim != 1 or len(differences) != len(participants) or not len(differences)
            or not np.isfinite(differences).all() or type(reps) is not int or reps < 1):
        raise ValueError('Invalid paired group bootstrap inputs')
    groups = sorted(set(participants.tolist()))
    if len(groups) < 2: raise ValueError('At least two participants required')
    counts = np.array([np.count_nonzero(participants == p) for p in groups])
    sums = np.array([differences[participants == p].sum() for p in groups])
    sampled = np.random.default_rng(seed).integers(0, len(groups), size=(reps, len(groups)))
    values = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return dict(estimate_pp=float(differences.mean() * 100),
                interval95_pp=(np.quantile(values, [.025, .975]) * 100).tolist(),
                participant_count=len(groups), replicates=reps, seed=seed,
                unit='participant', weighting='micro over all repeated clips',
                limitation='conditional on dev-selected baseline; exploratory, not selection-adjusted')


def analyze(schema, results, rankings):
    ids = schema['query_ids']
    labels = schema['labels']
    participants = schema['participant_ids']
    classes = schema['class_ids']
    if (not ids or len(set(ids)) != len(ids) or len(labels) != len(ids) or len(participants) != len(ids)
            or classes != list(range(len(classes))) or not classes):
        raise ValueError('Invalid prediction schema')
    refs = [{'sample_id': key, 'label': label} for key, label in zip(ids, labels)]
    expected = {f'{m}_s{s}_n{n}' for m in METHODS for s in SEEDS for n in BUDGETS}
    indexed = {r['name']: r for r in results}
    if len(indexed) != len(results) or set(indexed) != expected or set(rankings) != expected:
        raise ValueError('Incomplete method/seed/budget coverage')
    accuracy, clip_correct = {}, {}
    for name in sorted(expected):
        rank = np.asarray(rankings[name])
        if rank.shape != (len(ids), len(classes)) or not np.issubdtype(rank.dtype, np.integer):
            raise ValueError('Ranking shape/dtype differs')
        predictions = [{'sample_id': key, 'ranking': row.tolist()} for key, row in zip(ids, rank)]
        measured = evaluate(predictions, refs, classes)
        if set(measured) != set(indexed[name]['metrics']) or any(
                not np.isclose(measured[k], indexed[name]['metrics'][k], atol=1e-12, rtol=0) for k in measured):
            raise ValueError('Saved metrics disagree with complete rankings')
        accuracy[name] = measured['top1']
        clip_correct[name] = (rank[:, 0] == np.asarray(labels)).astype(np.float64)
    primary = {m: float(np.mean([accuracy[f'{m}_s{s}_n{n}'] for s in SEEDS for n in (2, 4)])) for m in METHODS}
    baseline = sorted(BASELINES, key=lambda m: (-primary[m], m))[0]
    per_seed = {str(s): float(np.mean([accuracy[f'ordered_ridge_s{s}_n{n}'] -
                                      accuracy[f'{baseline}_s{s}_n{n}'] for n in (2, 4)]) * 100) for s in SEEDS}
    paired = np.mean([clip_correct[f'ordered_ridge_s{s}_n{n}'] - clip_correct[f'{baseline}_s{s}_n{n}']
                      for s in SEEDS for n in (2, 4)], axis=0)
    advantage = (primary['ordered_ridge'] - primary[baseline]) * 100
    order_advantage = (primary['ordered_ridge'] - primary['shuffled_ridge']) * 100
    signal = (advantage >= 2 - 1e-12 and order_advantage >= 1 - 1e-12
              and sum(value > 0 for value in per_seed.values()) >= 2)
    summaries = {}
    for method in METHODS:
        summaries[method] = {}
        for budget in BUDGETS:
            values = [accuracy[f'{method}_s{s}_n{budget}'] * 100 for s in SEEDS]
            summaries[method][str(budget)] = dict(seed_values_percent=values,
                                                mean_percent=float(np.mean(values)),
                                                sample_SD_pp=float(np.std(values, ddof=1)))
    return dict(primary_percent={m: v * 100 for m, v in primary.items()}, strongest_baseline=baseline,
                ordered_minus_baseline_pp=advantage, ordered_minus_shuffled_pp=order_advantage,
                per_support_seed_primary_difference_pp=per_seed, practical_screen_signal=signal,
                conditional_participant_bootstrap=participant_bootstrap(paired, participants),
                label_curves=summaries, baseline_selection='aggregate dev primary; lexical name breaks ties',
                inference_scope='exploratory failure pilot; no novelty, significance or population utility conclusion')
