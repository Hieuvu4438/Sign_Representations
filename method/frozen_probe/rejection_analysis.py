"""Conditional signer-cluster rates and compact encoding of validated curves."""
import numpy as np


def cluster_draws(groups, replicates, rng):
    names = sorted(set(groups))
    if not names or type(replicates) is not int or replicates < 1:
        raise ValueError('Invalid cluster resampling input')
    return names, rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=replicates)


def cluster_mean_rate(accepted, masks, groups, draws):
    a, m, w = np.asarray(accepted), np.asarray(masks), np.asarray(draws)
    names = sorted(set(groups))
    if a.ndim != 2 or a.shape != m.shape or a.shape[1] != len(groups) or a.dtype != np.bool_ or m.dtype != np.bool_:
        raise ValueError('Complete Boolean seed/query arrays required')
    if w.ndim != 2 or w.shape[1] != len(names) or not np.issubdtype(w.dtype, np.integer) or (w < 0).any() or not np.all(w.sum(axis=1) == len(names)):
        raise ValueError('Invalid fixed cluster bootstrap multiplicities')
    if not names or a.shape[0] < 1 or w.shape[0] < 1:
        raise ValueError('Empty seed/group/bootstrap input')
    index = np.asarray(groups)
    numerator = np.stack([(a[:, index == group] & m[:, index == group]).sum(axis=1) for group in names], axis=1)
    denominator = np.stack([m[:, index == group].sum(axis=1) for group in names], axis=1)
    n, d = w @ numerator.T, w @ denominator.T
    valid = np.all(d > 0, axis=1)
    ratio = np.divide(n, d, out=np.zeros(n.shape, dtype=np.float64), where=d > 0)
    # Placeholder values in invalid draws may not enter any CI/inference.
    return ratio.mean(axis=1), valid


def conditional_interval(values, valid):
    x, ok = np.asarray(values), np.asarray(valid)
    if x.ndim != 1 or x.shape != ok.shape or ok.dtype != np.bool_ or not len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid bootstrap values/validity')
    if not ok.all():
        return dict(interval95=None, invalid_draws=int((~ok).sum()), replicates=len(x))
    return dict(interval95=np.quantile(x, [.025, .975]).tolist(), invalid_draws=0, replicates=len(x))


def pack_count_curve(points):
    if not points or points[0]['threshold'] != '+inf':
        raise ValueError('Curve must start at reject-all +infinity')
    rows = []
    for i, point in enumerate(points):
        threshold = 0. if i == 0 else float(point['threshold'])
        counts = [point[k] for k in ('known_accept_count', 'known_correct_accept_count', 'unknown_accept_count')]
        if not np.isfinite(threshold) or any(type(x) is not int or x < 0 or x >= 2**53 for x in counts):
            raise ValueError('Invalid compact curve threshold/count')
        if i == 0 and any(counts):
            raise ValueError('Reject-all curve point cannot accept samples')
        rows.append([threshold] + counts)
    return np.asarray(rows, dtype=np.float64)
