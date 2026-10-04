"""Ideal-real arithmetic envelopes; no floating-point certification claim."""
import hashlib
import numpy as np


def directional_bounds(gallery, linear_weight, gamma, beta):
    g, w, gamma, beta = map(lambda v: np.asarray(v, dtype=np.float64),
                           (gallery, linear_weight, gamma, beta))
    if (g.ndim != 2 or w.ndim != 2 or g.shape[1] != w.shape[0]
            or gamma.shape != (w.shape[1],) or beta.shape != gamma.shape
            or any(not np.isfinite(v).all() for v in (g, w, gamma, beta))):
        raise ValueError('Invalid LayerNorm/Linear/gallery contract')
    a = g @ w
    v = a * gamma
    center = a @ beta
    radius = np.sqrt(len(gamma)) * np.linalg.norm(v - v.mean(axis=1, keepdims=True), axis=1)
    return center - radius, center + radius


def source_order(source_sha, windows, seed):
    if (len(source_sha) != 64 or any(c not in '0123456789abcdef' for c in source_sha)
            or type(windows) is not int or windows < 1):
        raise ValueError('Invalid source identity or windows')
    digest = hashlib.sha256(f'{seed}:{source_sha}'.encode('ascii')).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'big')).permutation(windows)


def class_bounds(observed, remaining, lower, upper, class_rows):
    if type(remaining) is not int or remaining < 0:
        raise ValueError('Invalid remaining count')
    lo = observed + remaining * np.minimum(lower, 0)
    hi = observed + remaining * np.maximum(upper, 0)
    return (np.asarray([lo[rows].max() for rows in class_rows]),
            np.asarray([hi[rows].max() for rows in class_rows]))


def strict_winner(lower, upper):
    if len(lower) < 2 or len(lower) != len(upper) or np.any(lower > upper):
        raise ValueError('Invalid class intervals')
    candidate = int(np.argmax(lower))
    gap = float(lower[candidate] - np.max(np.delete(upper, candidate)))
    return (candidate if gap > 0 else None), gap


def simulate_prefix(features, weights, gallery, class_rows, lower, upper, order):
    """Stopping reads only acquired rows. Full arrays simulate an offline producer."""
    x = np.asarray(features, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.asarray(order)
    if (x.ndim != 2 or len(x) < 1 or weights.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or np.any(weights > 1)
            or not np.array_equal(np.sort(order), np.arange(len(x)))):
        raise ValueError('Invalid finite window input/order')
    observed = np.zeros(len(gallery), dtype=np.float64)
    weight_sum = 0.0
    for n, index in enumerate(order, 1):
        observed += weights[index] * (gallery @ x[index])
        weight_sum += weights[index]
        lo, hi = class_bounds(observed, len(x) - n, lower, upper, class_rows)
        winner, gap = strict_winner(lo, hi)
        if n < len(x) and weight_sum > 1e-12 and winner is not None:
            return dict(used_windows=n, early=True, class_id=winner,
                        strict_gap=gap, observed_weight_sum=float(weight_sum))
    # The native caller supplies exact pooling including all-zero-weight fallback.
    return dict(used_windows=len(x), early=False, class_id=None,
                strict_gap=None, observed_weight_sum=float(weight_sum))
