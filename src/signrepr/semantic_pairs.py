"""Gold-gated semantic pair accounting; half-credit ties and utterance units."""
import hashlib

import numpy as np


def candidate_order(pair_id, seed):
    swap = hashlib.sha256(f'{seed}:{pair_id}'.encode()).digest()[0] & 1
    return [1, 0] if swap else [0, 1]


def matched_credit(mean_nll, matched_position, tolerance=0):
    values = np.asarray(mean_nll, float)
    if values.shape != (2,) or not np.isfinite(values).all() or matched_position not in [0, 1] or tolerance < 0:
        raise ValueError('Invalid paired candidate scores/tie policy')
    difference = values[1-matched_position] - values[matched_position]
    return .5 if abs(difference) <= tolerance else float(difference > 0)


def validate_pairs(rows):
    if not rows:
        raise ValueError('Empty verified pair manifest')
    seen, utterances = set(), {}
    for r in rows:
        required = ['pair_id', 'utterance_id', 'recording_group', 'phenomenon', 'matched_text',
                    'mismatched_text', 'gold_provenance', 'review_status']
        if any(not isinstance(r.get(k), str) or not r[k].strip() for k in required):
            raise ValueError('Missing canonical pair/gold/group metadata')
        if r['pair_id'] in seen:
            raise ValueError('Duplicate canonical pair ID')
        seen.add(r['pair_id'])
        if r['review_status'] != 'EXPERT_VERIFIED_CANONICAL_VIDEO_AND_SEMANTIC_PAIR':
            raise ValueError('Pair mapping and semantic gold require expert verification')
        if r['matched_text'].strip() == r['mismatched_text'].strip():
            raise ValueError('A semantic pair requires distinct source candidates')
        if r['recording_group'].strip().lower() in ['unknown', 'none', 'null', 'nan', 'source_recording_unknown']:
            raise ValueError('Unknown recording grouping; sample-ID fallback forbidden')
        if utterances.setdefault(r['utterance_id'], r['recording_group']) != r['recording_group']:
            raise ValueError('One utterance cannot span different recording groups')
    return rows


def wrong_video_mapping(rows, seed):
    """Each recipient uses a different recording; deterministic, no gold scoring."""
    ids = sorted({r['utterance_id'] for r in rows})
    groups = {r['utterance_id']: r['recording_group'] for r in rows}
    mapping = {}
    for sid in ids:
        choices = [s for s in ids if groups[s] != groups[sid]]
        if not choices:
            raise ValueError('Wrong-video control needs at least two recording groups')
        mapping[sid] = min(choices, key=lambda s: hashlib.sha256(f'{seed}:{sid}:{s}'.encode()).digest())
    return mapping


def summarize(rows, credits, resamples=2000, seed=42):
    """Average pairs within utterance/phenomenon, then macro over phenomena.

    Identical recording multiplicities apply to all conditions and contrasts.
    Scores may contain .5 for exact ties; never convert ties into false labels.
    """
    validate_pairs(rows)
    values = {k: np.asarray(v, float) for k, v in credits.items()}
    if 'video' not in values or resamples < 2000 or any(v.shape != (len(rows),) or not np.isin(v, [0, .5, 1]).all() for v in values.values()):
        raise ValueError('Invalid paired credit arrays or bootstrap budget')
    units = {}
    for i, r in enumerate(rows):
        units.setdefault((r['utterance_id'], r['phenomenon']), []).append(i)
    keys = sorted(units)
    unit_values = {k: np.asarray([v[units[u]].mean() for u in keys]) for k, v in values.items()}
    group_for = {r['utterance_id']: r['recording_group'] for r in rows}
    group_names, group_ids = np.unique([group_for[s] for s, _ in keys], return_inverse=True)
    if len(group_names) < 2:
        raise ValueError('At least two recording groups required for descriptive CI')
    phenomena = sorted({p for _, p in keys})
    masks = {p: np.asarray([u[1] == p for u in keys]) for p in phenomena}
    def measured(v, weights):
        return {p: float(np.average(v[m], weights=weights[m])) for p, m in masks.items() if weights[m].sum() > 0}
    points = {k: measured(v, np.ones(len(keys))) for k, v in unit_values.items()}
    samples, differences, missing = {k: [] for k in values}, {k: [] for k in values if k != 'video'}, 0
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        weights = np.bincount(rng.integers(len(group_names), size=len(group_names)), minlength=len(group_names))[group_ids]
        draws = {k: measured(v, weights) for k, v in unit_values.items()}
        missing += len(draws['video']) != len(phenomena)
        macros = {k: float(np.mean(list(v.values()))) for k, v in draws.items()}
        for k, v in macros.items():
            samples[k].append(v)
            if k != 'video':
                differences[k].append(macros['video'] - v)
    return {'pair_rows': len(rows), 'utterances': len(group_for), 'utterance_phenomenon_units': len(keys),
            'recording_groups': len(group_names), 'resamples': resamples, 'bootstrap_seed': seed,
            'resamples_missing_phenomena': missing, 'tie_credit': .5,
            'aggregation': 'Mean pairs within utterance/phenomenon; macro over supported phenomena. Cluster bootstrap by recording, identical draws for all conditions.',
            'conditions': {k: {'macro_accuracy': float(np.mean(list(v.values()))), 'phenomenon_accuracy': v,
                               'ci95': np.quantile(samples[k], [.025, .975]).tolist()} for k, v in points.items()},
            'video_minus_control': {k: {'delta': float(np.mean(list(points['video'].values())) - np.mean(list(points[k].values()))),
                                       'delta_ci95': np.quantile(v, [.025, .975]).tolist()} for k, v in differences.items()},
            'limits': ['Conditional descriptive intervals on fixed model and reviewed cohort; not retraining or signer-population uncertainty.',
                       'Wrong/blank visual inputs are distribution-shift diagnostics, not causal proof.',
                       'Few recording groups and missing phenomena in bootstrap draws limit coverage.']}
