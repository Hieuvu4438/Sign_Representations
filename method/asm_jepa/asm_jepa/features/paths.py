"""Motion tokens (§8.3): multi-scale level-2 log-signatures of wrist paths over real time intervals.

Families: left wrist, right wrist, left-minus-right wrist (body frame). Centres: patches at quantiles
1/8, 3/8, 5/8, 7/8 of the valid patch timeline (= patches 2,6,10,14 for 16 valid patches). Widths are seconds
of real time (timestamps of the selected frames), clipped to the clip's valid span. Two views:
geometry = logsig2([x,y]) (3-d); timing = logsig2([x,y,t/width]) (6-d) + log1p(actual interval seconds) (7-d).
Strict gap policy: an interval needs >= path_min_points distinct timestamps and every path landmark observed
at every selected sample, otherwise the token is invalid (no bridging gaps with zeros).
"""
from __future__ import annotations

import numpy as np

from asm_jepa.data.schema import LH_WRIST, PATH_FAMILIES, RH_WRIST
from asm_jepa.data.sampling import patch_times
from asm_jepa.features.logsig2 import logsig2_np

FAMILY_JOINTS = {"left_wrist": (LH_WRIST,), "right_wrist": (RH_WRIST,), "left_minus_right_wrist": (LH_WRIST, RH_WRIST)}


def family_path(name: str, xy_b: np.ndarray, observed: np.ndarray, conf: np.ndarray):
    joints = FAMILY_JOINTS[name]
    if len(joints) == 1:
        j = joints[0]
        return xy_b[:, j].astype(np.float64), observed[:, j], conf[:, j]
    a, b = joints
    return (xy_b[:, a].astype(np.float64) - xy_b[:, b].astype(np.float64), observed[:, a] & observed[:, b],
            0.5 * (conf[:, a] + conf[:, b]))


def interval_supports(timestamps, time_valid, P, widths, centers_per_clip):
    """-> support [C,S,T] bool, center_time [C], lo/hi [C,S] seconds, center_patch [C]."""
    T = len(timestamps)
    pt, pvalid = patch_times(timestamps, time_valid, P)
    valid_patches = np.flatnonzero(pvalid)
    C, S = centers_per_clip, len(widths)
    support = np.zeros((C, S, T), bool)
    lo = np.zeros((C, S))
    hi = np.zeros((C, S))
    center_time = np.zeros(C)
    center_patch = np.full(C, -1, np.int64)
    if len(valid_patches) == 0:
        return support, center_time, lo, hi, center_patch
    real_t = timestamps[time_valid]
    t_first, t_last = float(real_t.min()), float(real_t.max())
    qs = (2 * np.arange(C) + 1) / (2 * C)
    for c, q in enumerate(qs):
        p = valid_patches[min(int(np.floor(q * len(valid_patches))), len(valid_patches) - 1)]
        center_patch[c] = p
        center_time[c] = pt[p]
        for s, w in enumerate(widths):
            lo[c, s] = max(pt[p] - w / 2.0, t_first)
            hi[c, s] = min(pt[p] + w / 2.0, t_last)
            support[c, s] = time_valid & (timestamps >= lo[c, s] - 1e-9) & (timestamps <= hi[c, s] + 1e-9)
    return support, center_time, lo, hi, center_patch


def path_tokens(xy_b, observed, conf, time_valid, timestamps, P, widths, centers_per_clip, min_points, stats,
                families=PATH_FAMILIES):
    """Returns dict with geom [F,C,S,3], timing [F,C,S,7] (z-scored, zero if invalid), valid [F,C,S],
    weight [F,C,S], center_time [C], support_patches [F,C,S,n_p] (raw dependency patches), raw_geom/raw_timing."""
    T = len(timestamps)
    n_p = T // P
    F, C, S = len(families), centers_per_clip, len(widths)
    support, center_time, lo, hi, _ = interval_supports(timestamps, time_valid, P, widths, C)
    inc_support = support[..., 1:] & support[..., :-1]  # [C,S,T-1]
    n_points = support.sum(-1)  # timestamps are strictly increasing on real frames -> distinct
    widths_arr = np.asarray(widths, np.float64)
    raw_geom = np.zeros((F, C, S, 3))
    raw_time = np.zeros((F, C, S, 7))
    valid = np.zeros((F, C, S), bool)
    weight = np.zeros((F, C, S), np.float32)
    for f, name in enumerate(families):
        path, obs, cf = family_path(name, xy_b, observed, conf)
        all_obs = ~np.any(support & ~obs[None, None, :], axis=-1)
        valid[f] = (n_points >= min_points) & all_obs
        p2 = np.broadcast_to(path, (C, S, T, 2))
        raw_geom[f] = logsig2_np(p2, inc_support)
        tnorm = (timestamps[None, None, :] - lo[..., None]) / widths_arr[None, :, None]
        p3 = np.concatenate([p2, tnorm[..., None]], -1)
        raw_time[f, ..., :6] = logsig2_np(p3, inc_support)
        raw_time[f, ..., 6] = np.log1p(np.maximum(hi - lo, 0.0))
        mean_conf = (support * cf[None, None, :]).sum(-1) / np.maximum(n_points, 1)
        weight[f] = np.where(valid[f], np.clip(mean_conf, 0.0, 1.0), 0.0)
    geom = np.clip((raw_geom - stats["geom_mean"][:, None]) / stats["geom_std"][:, None], -5, 5)
    timing = np.clip((raw_time - stats["time_mean"][:, None]) / stats["time_std"][:, None], -5, 5)
    geom = np.where(valid[..., None], geom, 0.0).astype(np.float32)
    timing = np.where(valid[..., None], timing, 0.0).astype(np.float32)
    patch_of = np.arange(T) // P
    support_patches = np.zeros((C, S, n_p), bool)
    for p in range(n_p):
        support_patches[..., p] = support[..., patch_of == p].any(-1)
    return {"geom": geom, "timing": timing, "valid": valid, "weight": weight, "center_time": center_time,
            "support_patches": np.broadcast_to(support_patches, (F, C, S, n_p)).copy(),
            "raw_geom": raw_geom, "raw_timing": raw_time}
