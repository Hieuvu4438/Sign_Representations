"""Relation tokens (§8.2): LH-wrist - RH-wrist, LH-wrist - nose, RH-wrist - nose, per frame in the body frame.

Per frame: [dx, dy, ||d||, confidence_pair, observed_pair] with dx/dy/norm z-scored by train stats,
confidence_pair = min(c1, c2). 2-D proximity only: never called physical contact.
"""
from __future__ import annotations

import numpy as np

from asm_jepa.data.schema import N_RELATIONS, RELATION_JOINTS


def relation_raw(xy_b: np.ndarray, observed: np.ndarray, time_valid: np.ndarray):
    """-> values [T,3,3] (dx,dy,norm; zeros if not observed), observed_pair [T,3]."""
    T = xy_b.shape[0]
    vals = np.zeros((T, N_RELATIONS, 3), np.float32)
    op = np.zeros((T, N_RELATIONS), bool)
    for r, (a, b) in enumerate(RELATION_JOINTS):
        ok = observed[:, a] & observed[:, b] & time_valid
        d = xy_b[:, a].astype(np.float64) - xy_b[:, b].astype(np.float64)
        v = np.concatenate([d, np.linalg.norm(d, axis=-1, keepdims=True)], -1)
        vals[:, r] = np.where(ok[:, None], v, 0.0)
        op[:, r] = ok
    return vals, op


def relation_frames(xy_b, observed, conf, time_valid, stats) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> features [T,3,5], observed_pair [T,3], confidence_pair [T,3]."""
    vals, op = relation_raw(xy_b, observed, time_valid)
    z = np.clip((vals - stats["rel_mean"]) / stats["rel_std"], -5.0, 5.0)
    cp = np.zeros_like(op, dtype=np.float32)
    for r, (a, b) in enumerate(RELATION_JOINTS):
        cp[:, r] = np.minimum(conf[:, a], conf[:, b])
    cp = np.where(op, cp, 0.0)
    feats = np.concatenate([np.where(op[..., None], z, 0.0), cp[..., None], op[..., None]], -1).astype(np.float32)
    return feats, op, cp


def relation_patch_stats(op: np.ndarray, cp: np.ndarray, time_valid: np.ndarray, P: int):
    """-> coverage [3,n_p], weight [3,n_p] (mean pair confidence over observed x observed fraction)."""
    T = op.shape[0]
    n_p = T // P
    real = time_valid.reshape(n_p, P)
    o = op.reshape(n_p, P, -1) & real[..., None]
    n_real = np.maximum(real.sum(1), 1)[:, None]
    cov = o.sum(1) / n_real
    mean_c = (cp.reshape(n_p, P, -1) * o).sum(1) / np.maximum(o.sum(1), 1)
    return cov.T.astype(np.float32), np.clip(mean_c * cov, 0.0, 1.0).T.astype(np.float32)
