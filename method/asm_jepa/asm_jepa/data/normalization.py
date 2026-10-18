"""Coordinate normalisation (§7.3).

body frame : xy_body = (xy_pixel - root_t) / s_B, root_t = shoulder midpoint (public anchors),
             s_B = median shoulder distance over the clip's valid shoulder frames (isotropic, pixel space).
hand frame : xy_local = (xy - wrist_t) / s_H(t), s_H(t) = median ||MCP_i - wrist|| over >=2 observed MCPs of
             the same hand at the same frame (never a clip-level hand scale -> no hidden-hand leakage).
Missing values are zero-filled after normalisation and always accompanied by a 0 flag.
"""
from __future__ import annotations

import numpy as np

from asm_jepa.data.schema import HAND_MCP_LOCAL, L_SHOULDER, N_JOINTS, PART_SLICES, R_SHOULDER

EPS = 1e-6


def body_normalize(xy: np.ndarray, observed: np.ndarray, time_valid: np.ndarray) -> dict:
    """xy [T,J,2] pixels, observed [T,J] -> body-frame coordinates (zeros where unobserved)."""
    T = xy.shape[0]
    sh_ok = observed[:, L_SHOULDER] & observed[:, R_SHOULDER] & time_valid
    out = {"xy": np.zeros_like(xy, dtype=np.float32), "anchor_fallback": np.zeros(T, bool),
           "unusable_pose": False, "body_scale_px": 0.0}
    if not sh_ok.any():
        out["unusable_pose"] = True
        return out
    left, right = xy[:, L_SHOULDER].astype(np.float64), xy[:, R_SHOULDER].astype(np.float64)
    s_b = float(np.median(np.linalg.norm(left[sh_ok] - right[sh_ok], axis=-1)))
    if not np.isfinite(s_b) or s_b <= EPS:
        out["unusable_pose"] = True
        return out
    root = np.where(sh_ok[:, None], (left + right) / 2.0, 0.0)
    fallback_root = np.median(((left + right) / 2.0)[sh_ok], axis=0)
    root[~sh_ok] = fallback_root
    xb = (xy.astype(np.float64) - root[:, None, :]) / s_b
    obs = observed & time_valid[:, None]
    xb = np.where(obs[..., None], xb, 0.0)
    out.update(xy=xb.astype(np.float32), anchor_fallback=~sh_ok & time_valid, body_scale_px=s_b)
    return out


def _median_of_valid(d: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise median of d[ok] for up to 4 values; returns (median, count)."""
    cnt = ok.sum(1)
    s = np.sort(np.where(ok, d, np.inf), axis=1)
    med = np.zeros(len(d))
    for c in (1, 2, 3, 4):
        rows = cnt == c
        if not rows.any():
            continue
        if c % 2:
            med[rows] = s[rows, c // 2]
        else:
            med[rows] = 0.5 * (s[rows, c // 2 - 1] + s[rows, c // 2])
    return med, cnt


def hand_local(xy_b: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame wrist-centred, MCP-scaled hand coordinates. Body/face rows stay zero with local_valid=0."""
    T = xy_b.shape[0]
    local = np.zeros((T, N_JOINTS, 2), np.float32)
    local_valid = np.zeros((T, N_JOINTS), bool)
    for part in ("lh", "rh"):
        sl = PART_SLICES[part]
        w = sl.start
        mcps = [w + k for k in HAND_MCP_LOCAL]
        wrist_ok = observed[:, w]
        ok = observed[:, mcps] & wrist_ok[:, None]
        d = np.linalg.norm(xy_b[:, mcps].astype(np.float64) - xy_b[:, w:w + 1].astype(np.float64), axis=-1)
        s_h, cnt = _median_of_valid(d, ok)
        frame_ok = (cnt >= 2) & (s_h > EPS)
        scale = np.where(frame_ok, s_h, 1.0)
        loc = (xy_b[:, sl].astype(np.float64) - xy_b[:, w:w + 1].astype(np.float64)) / scale[:, None, None]
        lv = observed[:, sl] & frame_ok[:, None]
        local[:, sl] = np.where(lv[..., None], loc, 0.0)
        local_valid[:, sl] = lv
    return local, local_valid


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray, valid: np.ndarray, clip: float = 5.0) -> np.ndarray:
    z = np.clip((x - mean) / std, -clip, clip)
    return np.where(valid[..., None] if valid.ndim == x.ndim - 1 else valid, z, 0.0).astype(np.float32)


def joint_channels(xy_b, observed, conf, local, local_valid, time_valid, stats) -> np.ndarray:
    """-> [T,114,7]: body_x, body_y, hand_local_x, hand_local_y, confidence, observed, local_valid.

    Coordinates are z-scored with fixed train-only statistics; confidence and flags are not.
    """
    obs = observed & time_valid[:, None]
    lv = local_valid & obs
    out = np.zeros(xy_b.shape[:2] + (7,), np.float32)
    out[..., 0:2] = standardize(xy_b, stats["body_mean"], stats["body_std"], obs)
    out[..., 2:4] = standardize(local, stats["local_mean"], stats["local_std"], lv)
    out[..., 4] = np.where(obs, conf, 0.0)
    out[..., 5] = obs
    out[..., 6] = lv
    return out


def saturation(xy_b, local, observed, local_valid, bound: float) -> dict:
    o = observed.sum()
    lv = local_valid.sum()
    return {"body_saturation": float((np.abs(xy_b[observed]) > bound).any(-1).sum() / max(o, 1)),
            "local_saturation": float((np.abs(local[local_valid]) > bound).any(-1).sum() / max(lv, 1))}
