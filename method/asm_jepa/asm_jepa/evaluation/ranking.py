"""Rank -> metric conversion shared by frozen retrieval and classifier ranking (§13.2).

Ranks start at 1. Ties are broken deterministically by ascending class id, without looking at y_true:
rank(y) = 1 + #{c : s_c > s_y} + #{c < y : s_c == s_y}, i.e. the position of y in a stable sort by
(-score, class_id).
"""
from __future__ import annotations

import numpy as np
import torch


def ranks_from_scores(scores, y_true) -> np.ndarray:
    """scores [N,C] (numpy or torch), y_true [N] -> int ranks [N]."""
    s = torch.as_tensor(scores)
    y = torch.as_tensor(np.asarray(y_true), device=s.device, dtype=torch.long)
    true = s.gather(1, y[:, None])
    higher = (s > true).sum(1)
    cls = torch.arange(s.shape[1], device=s.device)[None, :]
    tie_before = ((s == true) & (cls < y[:, None])).sum(1)
    return (1 + higher + tie_before).cpu().numpy().astype(np.int64)


def metrics_from_ranks(ranks, ks=(1, 5, 10)) -> dict:
    r = np.asarray(ranks, np.float64)
    if r.size == 0:
        return {"n": 0}
    out = {"n": int(r.size)}
    for k in ks:
        out[f"r{k}"] = float(np.mean(r <= k))
    out["mrr"] = float(np.mean(1.0 / r))
    out["dcg"] = float(np.mean(1.0 / np.log2(r + 1.0)))
    for k in ks:
        out[f"r{k}_pct"] = 100.0 * out[f"r{k}"]
    out["mrr_pct"] = 100.0 * out["mrr"]
    out["dcg_pct"] = 100.0 * out["dcg"]
    out["mean_rank"] = float(np.mean(r))
    out["median_rank"] = float(np.median(r))
    return out


def quality_bins(ranks, quality, n_bins: int = 3, ks=(1, 5, 10)) -> dict:
    """Metrics per pose-quality quantile bin of the queries (quality = mean state-token confidence weight)."""
    r, q = np.asarray(ranks), np.asarray(quality, np.float64)
    if len(r) == 0:
        return {}
    edges = np.quantile(q, np.linspace(0, 1, n_bins + 1))
    out = {}
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (q >= lo) & ((q <= hi) if i == n_bins - 1 else (q < hi))
        out[f"bin{i}"] = {"quality_range": [float(lo), float(hi)], **metrics_from_ranks(r[m], ks)}
    return out
