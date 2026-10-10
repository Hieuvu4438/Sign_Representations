"""Dictionary retrieval metrics (ASL Citizen style) and pooling for p4_signssl.

Protocol: gallery = labelled train clips, query = val (or test) clips. Class score =
max cosine over gallery clips of that class (equivalent to 1-NN label assignment for
rank 1, as in SignRep Sec 7.2). Rank r of the true class (1 = best; ties count against
the query: r = 1 + #classes with score >= true score, excluding the true class).
DCG = 1/log2(1+r), MRR = 1/r, Rec@K = [r <= K].
"""
from __future__ import annotations

import numpy as np
import torch


def activity_weights(active, clamp=True):
    """SignRep Eq.12: gamma = max(P_LH, P_RH); outputs are regressed [0,1] targets."""
    g = active.max(axis=1)
    return np.clip(g, 0.0, 1.0) if clamp else g


def pool_clips(win, offsets, weights=None):
    """Weighted (or uniform) mean of window vectors per clip. Falls back to uniform when
    the weight mass of a clip is ~0 (zero-activity policy)."""
    n = len(offsets) - 1
    out = np.zeros((n, win.shape[1]), np.float32)
    fallback = 0
    for i in range(n):
        s, e = offsets[i], offsets[i + 1]
        x = win[s:e].astype(np.float32)
        if weights is None:
            out[i] = x.mean(0)
            continue
        w = weights[s:e].astype(np.float32)
        if w.sum() < 1e-6:
            out[i] = x.mean(0)
            fallback += 1
        else:
            out[i] = (w[:, None] * x).sum(0) / w.sum()
    return out, fallback


def l2n(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


@torch.no_grad()
def class_max_scores(q, g, g_labels, n_classes, device="cuda", chunk=2048):
    """q [Nq,D], g [Ng,D] (already normalized), g_labels int [Ng] -> [Nq,C] class-max cos."""
    G = torch.from_numpy(np.ascontiguousarray(g)).to(device)
    gl = torch.from_numpy(np.asarray(g_labels, np.int64)).to(device)
    out = []
    for i in range(0, len(q), chunk):
        Q = torch.from_numpy(np.ascontiguousarray(q[i : i + chunk])).to(device)
        s = Q @ G.T
        cs = torch.full((len(Q), n_classes), -1e9, device=device)
        cs.scatter_reduce_(1, gl.expand(len(Q), -1), s, reduce="amax", include_self=True)
        out.append(cs.cpu())
    return torch.cat(out).numpy()


def ranks_from_scores(scores, q_labels):
    q_labels = np.asarray(q_labels, np.int64)
    true = scores[np.arange(len(scores)), q_labels]
    ge = (scores >= true[:, None]).sum(1)  # includes the true class itself
    return ge.astype(np.int64)  # = 1 + #others with score >= true


def metrics_from_ranks(r, q_labels=None):
    r = np.asarray(r, np.float64)
    m = {
        "DCG": float(np.mean(1.0 / np.log2(1.0 + r))),
        "MRR": float(np.mean(1.0 / r)),
        "R@1": float(np.mean(r <= 1)),
        "R@5": float(np.mean(r <= 5)),
        "R@10": float(np.mean(r <= 10)),
        "n": int(len(r)),
    }
    if q_labels is not None:
        q_labels = np.asarray(q_labels)
        cls = np.unique(q_labels)
        m["R@1_classmacro"] = float(np.mean([np.mean(r[q_labels == c] <= 1) for c in cls]))
    return m


def per_query(r):
    r = np.asarray(r, np.float64)
    return {"DCG": 1.0 / np.log2(1.0 + r), "MRR": 1.0 / r, "R@1": (r <= 1).astype(float),
            "R@5": (r <= 5).astype(float)}


def paired_cluster_bootstrap(a, b, clusters, n_boot=10000, seed=0):
    """Mean difference b-a with cluster (e.g. signer) bootstrap 95% CI, plus per-cluster diffs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    clusters = np.asarray(clusters)
    uc = np.unique(clusters)
    idx = [np.where(clusters == c)[0] for c in uc]
    sums = np.array([(b[i] - a[i]).sum() for i in idx])
    cnts = np.array([len(i) for i in idx])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uc), size=(n_boot, len(uc)))
    boot = sums[draws].sum(1) / cnts[draws].sum(1)
    per = {str(c): float((b[i] - a[i]).mean()) for c, i in zip(uc, idx)}
    return {"diff": float((b - a).mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "per_cluster": per, "n_clusters_positive": int(sum(v > 0 for v in per.values())), "n_clusters": len(uc)}
