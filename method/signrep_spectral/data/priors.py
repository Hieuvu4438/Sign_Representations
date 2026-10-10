"""Canonical targets ``hand_image_canonical_v1_xy`` for a 16-frame clip.

Deviation from spec 6.3: the local pose has no z, so per-hand targets are
21 joints x (x, y) = 42 dims (joint-major), and spectral hand groups use joints
1..20 -> indices 2..41 (D=40). Everything else follows the spec.
"""
from __future__ import annotations

import numpy as np

TARGET_SCHEMA = "hand_image_canonical_v1_xy_r2"
HEADS = {"left_pose": 42, "right_pose": 42, "wrist_image_xy": 4}
SPECTRAL_GROUPS = {
    "left_hand_motion": ("left_pose", slice(2, 42)),
    "right_hand_motion": ("right_pose", slice(2, 42)),
    "wrist_translation": ("wrist_image_xy", slice(0, 4)),
}


def build_canonical_targets(raw: dict, start: int, length: int, width: int, height: int,
                            scale_min_obs: int = 4, scale_floor: float = 1e-4,
                            max_rel_extent: float = float("inf")) -> dict:
    """r2 policy: joints whose wrist-relative coordinate exceeds ``max_rel_extent``
    palm lengths are marked unobserved (anatomically implausible detections)."""
    sl = slice(start, start + length)
    xy = raw["hand_xy"][sl].astype(np.float64)  # [T,2,21,2]
    obs = raw["joint_observed"][sl]  # [T,2,21]
    if xy.shape[0] != length:
        raise ValueError("clip shorter than requested length")
    u = xy.copy()
    u[..., 1] *= height / width  # y in width units
    targets, masks, info = {}, {}, {}
    wrist_xy = np.zeros((length, 4), np.float64)
    wrist_obs = np.zeros((length, 4), bool)
    for h, name in enumerate(("left_pose", "right_pose")):
        valid_scale = obs[:, h, 0] & obs[:, h, 9]
        d = np.linalg.norm(u[:, h, 9] - u[:, h, 0], axis=-1)
        n_scale = int(valid_scale.sum())
        s = float(np.median(d[valid_scale])) if n_scale else 0.0
        scale_ok = n_scale >= scale_min_obs and s >= scale_floor
        rel = (u[:, h] - u[:, h, :1]) / (s if scale_ok else 1.0)  # [T,21,2]
        m = obs[:, h] & obs[:, h, :1] & scale_ok  # joint & wrist observed
        m &= (np.abs(rel) <= max_rel_extent).all(-1)
        targets[name] = np.where(m[..., None], rel, 0.0).reshape(length, 42)
        masks[name] = np.repeat(m, 2, axis=1)  # joint-major xy
        info[f"{name}_scale"] = s
        info[f"{name}_scale_ok"] = bool(scale_ok)
        wrist_xy[:, 2 * h : 2 * h + 2] = np.where(obs[:, h, :1], u[:, h, 0], 0.0)
        wrist_obs[:, 2 * h : 2 * h + 2] = obs[:, h, :1]
    targets["wrist_image_xy"] = wrist_xy
    masks["wrist_image_xy"] = wrist_obs
    return {
        "targets": {k: v.astype(np.float32) for k, v in targets.items()},
        "masks": masks,
        "info": info,
    }
