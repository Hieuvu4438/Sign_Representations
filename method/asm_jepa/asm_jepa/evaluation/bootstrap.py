"""Percentile bootstrap CIs over queries and paired bootstrap differences between two rank files."""
from __future__ import annotations

import numpy as np

from asm_jepa.evaluation.ranking import metrics_from_ranks

KEYS = ("r1", "r5", "r10", "mrr", "dcg")


def _stats(r):
    m = metrics_from_ranks(r)
    return np.array([m[k] for k in KEYS])


def bootstrap_ci(ranks, n_boot=1000, seed=0, alpha=0.05, cluster=None) -> dict:
    """Resample queries (or clusters, e.g. signers, when `cluster` is given) with replacement."""
    ranks = np.asarray(ranks)
    rng = np.random.default_rng(seed)
    if cluster is None:
        draws = [ranks[rng.integers(0, len(ranks), len(ranks))] for _ in range(n_boot)]
    else:
        cluster = np.asarray(cluster)
        groups = [np.flatnonzero(cluster == c) for c in np.unique(cluster)]
        draws = [ranks[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])]
                 for _ in range(n_boot)]
    vals = np.stack([_stats(d) for d in draws])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2], axis=0)
    return {k: {"lo": float(a), "hi": float(b)} for k, a, b in zip(KEYS, lo, hi)}


def paired_bootstrap(ranks_a, ranks_b, n_boot=1000, seed=0) -> dict:
    a, b = np.asarray(ranks_a), np.asarray(ranks_b)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        diffs.append(_stats(a[i]) - _stats(b[i]))
    diffs = np.stack(diffs)
    point = _stats(a) - _stats(b)
    return {k: {"diff": float(p), "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975)),
                "p_le_0": float(np.mean(d <= 0))} for k, p, d in zip(KEYS, point, diffs.T)}
