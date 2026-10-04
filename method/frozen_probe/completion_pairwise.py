"""Shared-query ideal-real completion envelopes. No learned or fitted bounds."""
import numpy as np


def pair_lower_bounds(gallery, linear_weight, gamma, beta):
    g, w, gamma, beta = map(lambda x: np.asarray(x, dtype=np.float64),
                           (gallery, linear_weight, gamma, beta))
    if (g.ndim != 2 or w.shape != (g.shape[1], len(gamma))
            or beta.shape != gamma.shape
            or any(not np.isfinite(x).all() for x in (g, w, gamma, beta))):
        raise ValueError('Invalid pairwise projection inputs')
    a = g @ w
    v = a * gamma
    center = a @ beta
    lower = np.empty((len(g), len(g)), dtype=np.float64)
    for anchor in range(len(g)):
        difference = v[anchor] - v
        difference -= difference.mean(axis=1, keepdims=True)
        radius = np.sqrt(len(gamma)) * np.linalg.norm(difference, axis=1)
        lower[anchor] = center[anchor] - center - radius
    return lower


def simulate_paired_prefix(features, weights, gallery, class_rows, lower, upper,
                           order, pair_lower):
    # lower/upper are retained for the separate component-envelope verifier.
    x, weights, order = np.asarray(features, dtype=np.float64), np.asarray(weights), np.asarray(order)
    if (x.ndim != 2 or len(x) < 1 or weights.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or np.any(weights > 1)
            or not np.array_equal(np.sort(order), np.arange(len(x)))
            or pair_lower.shape != (len(gallery), len(gallery))):
        raise ValueError('Invalid paired prefix inputs')
    classes = np.full(len(gallery), -1, dtype=int)
    for label, rows in enumerate(class_rows):
        if np.any(classes[rows] != -1):
            raise ValueError('Duplicate gallery class assignment')
        classes[rows] = label
    if np.any(classes < 0) or len(class_rows) < 2:
        raise ValueError('Incomplete gallery classes')
    observed = np.zeros(len(gallery), dtype=np.float64)
    weight_sum = 0.0
    for n, index in enumerate(order, 1):
        observed += weights[index] * (gallery @ x[index])
        weight_sum += weights[index]
        anchor = int(np.argmax(observed))
        competitor = classes != classes[anchor]
        differences = (observed[anchor] - observed[competitor]
                       + (len(x)-n)*np.minimum(0., pair_lower[anchor, competitor]))
        gap = float(differences.min())
        if n < len(x) and weight_sum > 1e-12 and gap > 0:
            return dict(used_windows=n, early=True, class_id=int(classes[anchor]),
                        strict_gap=gap, observed_weight_sum=float(weight_sum))
    return dict(used_windows=len(x), early=False, class_id=None,
                strict_gap=None, observed_weight_sum=float(weight_sum))
