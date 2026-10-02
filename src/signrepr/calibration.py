"""Categorical probability diagnostics; not boundary/presence calibration."""
import numpy as np


def categorical_scores(logits, gold, temperature=1., bins=5):
    logits, gold = np.asarray(logits, float), np.asarray(gold)
    if logits.ndim != 2 or gold.shape != (len(logits),) or not len(gold):
        raise ValueError('Categorical score dimensions disagree')
    if not np.isfinite(logits).all() or not np.isfinite(temperature) or temperature <= 0 or bins < 1:
        raise ValueError('Invalid logits, temperature or bins')
    if not np.issubdtype(gold.dtype, np.integer) or gold.min() < 0 or gold.max() >= logits.shape[1]:
        raise ValueError('Invalid categorical labels')
    shifted = logits / temperature
    shifted -= shifted.max(1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    probability = np.exp(logp)
    prediction = probability.argmax(1)
    confidence = probability.max(1)
    correct = prediction == gold
    nll = -logp[np.arange(len(gold)), gold]
    brier = ((probability-np.eye(logits.shape[1])[gold])**2).sum(1)
    edges = np.linspace(0, 1, bins+1)
    assignment = np.minimum(np.searchsorted(edges, confidence, side='right')-1, bins-1)
    table, ece = [], 0.
    for i in range(bins):
        keep = assignment == i
        count = int(keep.sum())
        acc, conf = (float(correct[keep].mean()), float(confidence[keep].mean())) if count else (None, None)
        if count:
            ece += count/len(gold)*abs(acc-conf)
        table.append({'lower': float(edges[i]), 'upper': float(edges[i+1]), 'count': count,
                      'accuracy': acc, 'mean_confidence': conf})
    return {'nll': nll, 'brier': brier, 'probability': probability, 'prediction': prediction,
            'metrics': {'mean_nll': float(nll.mean()), 'multiclass_brier': float(brier.mean()),
                        'top1': float(correct.mean()), 'ece': float(ece), 'samples': len(gold),
                        'temperature': float(temperature), 'reliability_bins': table}}


def select_temperature(validation_logits, validation_gold, grid):
    candidates = [{'temperature': float(t), 'validation_mean_nll': categorical_scores(
        validation_logits, validation_gold, t)['metrics']['mean_nll']} for t in grid]
    if not candidates:
        raise ValueError('Temperature grid is empty')
    chosen = min(candidates, key=lambda r: r['validation_mean_nll'])
    return chosen['temperature'], candidates


def paired_mean_bootstrap(groups, values, reference, resamples=2000, seed=42):
    values, reference = np.asarray(values, float), np.asarray(reference, float)
    if values.ndim != 2 or values.shape != reference.shape or values.shape[1] != len(groups):
        raise ValueError('Paired seed/group dimensions disagree')
    if not np.isfinite(values).all() or not np.isfinite(reference).all() or resamples < 2000:
        raise ValueError('Invalid paired values or insufficient resamples')
    _, inverse = np.unique(groups, return_inverse=True)
    n = len(np.unique(inverse))
    if n < 2:
        raise ValueError('Fewer than two source groups')
    rng, deltas = np.random.default_rng(seed), []
    for _ in range(resamples):
        weights = np.bincount(rng.integers(n, size=n), minlength=n)[inverse]
        deltas.append(float(np.average(values-reference, axis=1, weights=weights).mean()))
    return {'point': float(values.mean()), 'reference_point': float(reference.mean()),
            'delta': float((values-reference).mean()), 'delta_ci95': np.quantile(deltas, [.025, .975]).tolist(),
            'groups': n, 'samples': len(groups), 'seeds': values.shape[0], 'resamples': resamples,
            'scope': 'Conditional fixed-head source-group interval; no signer-population or retraining uncertainty.'}
