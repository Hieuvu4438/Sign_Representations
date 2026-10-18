"""State-token support (§8.1): part/patch validity, confidence weights, anchor and MAE descriptors.

All functions read the [T,114,7] joint-channel tensor of one view (channels: body xy z-scored, hand-local xy
z-scored, confidence, observed, local_valid) so every descriptor of patch p only reads frames of patch p
and joints of its own part (plus public shoulder anchors already folded into the body frame).
"""
from __future__ import annotations

import numpy as np

from asm_jepa.data.schema import FACE_ANCHOR_LOCAL, PART_ORDER, PART_SLICES

ANCHOR_OFFSETS_FN = lambda P: [0, P // 2, P - 1]  # noqa: E731


def part_patch_coverage(joints: np.ndarray, time_valid: np.ndarray, P: int):
    """-> coverage [4,n_p] (observed fraction over the part's joints x real frames), real_frames [n_p],
    weight [4,n_p] = mean confidence over observed * observed fraction (clipped to [0,1])."""
    T = joints.shape[0]
    n_p = T // P
    obs = joints[..., 5].reshape(n_p, P, -1)
    conf = joints[..., 4].reshape(n_p, P, -1)
    real = time_valid.reshape(n_p, P)
    real_cnt = real.sum(1)
    cov = np.zeros((len(PART_ORDER), n_p), np.float32)
    weight = np.zeros_like(cov)
    for a, part in enumerate(PART_ORDER):
        sl = PART_SLICES[part]
        o = obs[:, :, sl] * real[:, :, None]
        denom = np.maximum(real_cnt * (sl.stop - sl.start), 1)
        n_obs = o.sum((1, 2))
        cov[a] = n_obs / denom
        mean_conf = (conf[:, :, sl] * o).sum((1, 2)) / np.maximum(n_obs, 1)
        weight[a] = np.clip(mean_conf * cov[a], 0.0, 1.0)
    return cov, real_cnt, weight


def anchor_targets(joints: np.ndarray, time_valid: np.ndarray, P: int) -> dict:
    """State-anchor descriptors at offsets [0, P//2, P-1] of every patch (§10.2).

    hands: 21 hand-local xy + wrist body xy (44/offset); body: 4 body xy (8/offset);
    face: mouth 48-67 + brows 17-26 body xy (60/offset). Returns {part: (values [n_p,D], valid [n_p,D])}.
    """
    T = joints.shape[0]
    n_p = T // P
    offs = np.array(ANCHOR_OFFSETS_FN(P))
    frames = (np.arange(n_p)[:, None] * P + offs[None, :])  # [n_p, 3]
    real = time_valid[frames]  # [n_p,3]
    out = {}
    for part in PART_ORDER:
        sl = PART_SLICES[part]
        if part in ("lh", "rh"):
            loc = joints[frames, sl, 2:4]  # [n_p,3,21,2]
            loc_ok = joints[frames, sl, 6] > 0.5
            wrist = joints[frames, sl.start, 0:2][:, :, None, :]  # [n_p,3,1,2]
            wrist_ok = (joints[frames, sl.start, 5] > 0.5)[:, :, None]
            vals = np.concatenate([loc.reshape(n_p, 3, -1), wrist.reshape(n_p, 3, -1)], -1)
            ok = np.concatenate([np.repeat(loc_ok, 2, -1), np.repeat(wrist_ok, 2, -1)], -1)
        else:
            idx = np.arange(sl.start, sl.stop) if part == "body" else sl.start + np.array(FACE_ANCHOR_LOCAL)
            vals = joints[frames][:, :, idx, 0:2].reshape(n_p, 3, -1)
            ok = np.repeat(joints[frames][:, :, idx, 5] > 0.5, 2, -1)
        ok = ok & real[:, :, None]
        out[part] = (np.where(ok, vals, 0.0).reshape(n_p, -1).astype(np.float32), ok.reshape(n_p, -1))
    return out


MAE_DIMS = {"body": 4 * 8, "lh": 4 * 44, "rh": 4 * 44, "face": 4 * 136}


def mae_targets(joints: np.ndarray, time_valid: np.ndarray, P: int) -> dict:
    """R1 coordinate-MAE targets: raw normalised coordinates of all P frames of each state patch.
    hands: local 21x2 + global wrist 2 per frame; body 4x2; face 68x2 (body frame, z-scored)."""
    T = joints.shape[0]
    n_p = T // P
    real = time_valid.reshape(n_p, P)
    out = {}
    for part in PART_ORDER:
        sl = PART_SLICES[part]
        j = joints[:, sl].reshape(n_p, P, sl.stop - sl.start, 7)
        if part in ("lh", "rh"):
            vals = np.concatenate([j[..., 2:4].reshape(n_p, P, -1), j[:, :, 0, 0:2]], -1)
            ok = np.concatenate([np.repeat(j[..., 6] > 0.5, 2, -1), np.repeat(j[:, :, 0:1, 5] > 0.5, 2, -1)], -1)
        else:
            vals = j[..., 0:2].reshape(n_p, P, -1)
            ok = np.repeat(j[..., 5] > 0.5, 2, -1)
        ok = ok & real[:, :, None]
        out[part] = (np.where(ok, vals, 0.0).reshape(n_p, -1).astype(np.float32), ok.reshape(n_p, -1))
    return out
