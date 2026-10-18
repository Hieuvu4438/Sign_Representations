"""Leak-free window pipeline on raw keypoints (numpy, CPU workers) — proposal12 v2.1 §6.3 steps 2–4 and §6.10.

Order (the end-to-end perturbation test checks exactly this order):
  1. handedness decision on the whole clip (declared 1-bit leak) + mirroring
  2. content-independent window placement: start time and speed come from the RNG and the clip length only
  3. speed resampling by linear interpolation with explicit source support per output frame
  4. (dataset) mask sampling on the bin grid, propagation of hidden source frames, scrubbing of the student source
  5. augmentation of the resampled frames (camera similarity, morphology, keypoint noise, hand dropout)
Coordinates are source pixels; NaN = not observed (or scrubbed).
"""
from __future__ import annotations

import numpy as np

from fp.joints import (ANCHORS, EYE_CORNERS, LELB, LH0, LSHO, LWRI, MIRROR_PERM, RELB, RH0, RSHO, RWRI, F0, NF)

TARGET_FPS = 25.0


# ------------------------------------------------------------------ handedness
def left_dominant(xy: np.ndarray) -> bool:
    """dominant side = larger summed body-wrist motion over frames where the wrist is observed in consecutive frames."""
    def motion(j):
        p = xy[:, j]
        ok = np.isfinite(p).all(-1)
        both = ok[1:] & ok[:-1]
        return float(np.abs(np.diff(p, axis=0))[both].sum()) if both.any() else 0.0
    return motion(LWRI) > motion(RWRI)


def shoulder_center_x(xy):
    c = (xy[:, LSHO, 0] + xy[:, RSHO, 0]) / 2
    c = c[np.isfinite(c)]
    return float(np.median(c)) if len(c) else 0.0


def mirror(xy, conf):
    cx = shoulder_center_x(xy)
    out = xy.copy()
    out[..., 0] = 2 * cx - out[..., 0]
    return out[:, MIRROR_PERM], conf[:, MIRROR_PERM]


def canonicalise(xy, conf):
    """returns (xy, conf, flipped) with the dominant hand on the RIGHT-side indices."""
    if left_dominant(xy):
        a, b = mirror(xy, conf)
        return a, b, True
    return xy, conf, False


# ------------------------------------------------------------------ window placement and resampling
def window_positions(n_frames, fps, T=64, start_s=0.0, speed=1.0, target_fps=TARGET_FPS):
    """output sample k is at time start_s + k * speed / eff (eff = min(fps, target_fps)).
    returns i0, i1 (int source frames), w (weight of i1), valid (bool: inside the clip)."""
    eff = min(fps, target_fps)
    u = (start_s + np.arange(T) * speed / eff) * fps
    i0 = np.floor(u).astype(np.int64)
    w = (u - i0).astype(np.float32)
    valid = (i0 >= 0) & (i0 <= n_frames - 1)
    i1 = np.minimum(i0 + 1, n_frames - 1)
    w = np.where((i1 == i0) | ~valid, 0.0, w).astype(np.float32)
    w = np.where(w < 1e-6, 0.0, w)
    i0 = np.clip(i0, 0, max(n_frames - 1, 0))
    return i0, i1, w, valid


def random_start(n_frames, fps, T, speed, rng, target_fps=TARGET_FPS):
    dur = n_frames / fps
    need = T * speed / min(fps, target_fps)
    return float(rng.uniform(0.0, max(0.0, dur - need)))


def support(i0, i1, w, valid):
    """list of source frames each output frame reads (empty for padding)."""
    return [([] if not v else ([a] if ww == 0 else [a, b])) for a, b, ww, v in zip(i0, i1, w, valid)]


def resample(xy, conf, i0, i1, w, valid, flag=None):
    """linear interpolation; a joint is observed only if every source frame it reads (weight > 0) is observed.
    flag (optional, [F, J] bool) is OR-propagated through the support (used for scrubbed / hidden joints)."""
    a, b = xy[i0], xy[i1]
    ww = w[:, None, None]
    b_used = np.where(ww > 0, b, a)
    out = (1 - ww) * a + ww * b_used                               # NaN propagates when a used frame is NaN
    need_b = (w > 0)[:, None]
    ok = np.isfinite(a).all(-1) & (np.isfinite(b).all(-1) | ~need_b)
    out = np.where(ok[..., None], out, np.nan)
    c = (1 - w[:, None]) * conf[i0] + w[:, None] * conf[i1]
    c = np.where(ok, c, 0.0)
    out[~valid] = np.nan
    c[~valid] = 0.0
    f = None
    if flag is not None:
        f = flag[i0] | (flag[i1] & need_b)
        f[~valid] = False
    return out.astype(np.float32), c.astype(np.float32), f


# ------------------------------------------------------------------ augmentation (applied per view on output frames)
def anchor_frame(xy):
    """clip-median shoulder centre, width and angle (anchors are never masked)."""
    L, R = xy[:, LSHO], xy[:, RSHO]
    ok = np.isfinite(L).all(-1) & np.isfinite(R).all(-1)
    if not ok.any():
        return np.zeros(2, np.float32), 100.0, 0.0
    v = (L - R)[ok]
    c = np.median((L + R)[ok] / 2, 0)
    sig = float(np.median(np.linalg.norm(v, axis=-1))) + 1e-6
    ang = float(np.arctan2(np.median(v[:, 1]), np.median(v[:, 0])))
    return c.astype(np.float32), sig, ang


def camera_affine(rng, rot_deg, scale, shear):
    """content-independent 2x2 augmentation applied in the BODY frame by pj.features (a pixel-space similarity around the
    shoulder centre would be undone by the anchor normalisation — code review finding #3)."""
    th = np.deg2rad(rng.uniform(-rot_deg, rot_deg)) if rot_deg else 0.0
    s = 1 + rng.uniform(-scale, scale) if scale else 1.0
    sh = rng.uniform(-shear, shear) if shear else 0.0
    A = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.array([[1, sh], [0, 1]])
    return A.astype(np.float32)


def _ik_elbow(s, e, w, k_up, k_fo):
    """re-solve the elbow for scaled segment lengths keeping shoulder s and wrist w fixed (2-link IK, 2-D)."""
    a = np.linalg.norm(e - s, axis=-1) * k_up
    b = np.linalg.norm(w - e, axis=-1) * k_fo
    d = np.linalg.norm(w - s, axis=-1) + 1e-6
    tot = a + b
    scale = np.where(d > 0.999 * tot, d / (0.999 * tot), 1.0)          # make the wrist reachable
    a, b = a * scale, b * scale
    d_eff = np.maximum(d, 1.001 * np.abs(a - b))                        # wrist too close: clamp the triangle
    x = (a ** 2 - b ** 2 + d_eff ** 2) / (2 * d_eff)
    h = np.sqrt(np.maximum(a ** 2 - x ** 2, 0.0))
    u = (w - s) / d[:, None]
    n = np.stack([-u[:, 1], u[:, 0]], -1)
    side = np.sign(((e - s) * n).sum(-1)); side = np.where(side == 0, 1.0, side)
    return s + x[:, None] * u + (side * h)[:, None] * n


def morphology(xy, rng, arm=0.10, shoulder=0.05, finger=0.10, face=0.05):
    """end-effector-preserving morphology change: elbows re-solved with wrists fixed, shoulder width around the
    shoulder centre with wrists fixed, finger segments scaled from the MCP, face scaled around the eye centre."""
    out = xy.copy()
    c, _, _ = anchor_frame(xy)
    ksh = 1 + rng.uniform(-shoulder, shoulder)
    for j in (LSHO, RSHO):
        out[:, j] = c + ksh * (xy[:, j] - c)
    for sh, el, wr in ((LSHO, LELB, LWRI), (RSHO, RELB, RWRI)):
        k1, k2 = 1 + rng.uniform(-arm, arm), 1 + rng.uniform(-arm, arm)      # draw unconditionally: RNG use is content-independent
        s, e, w = out[:, sh], xy[:, el], xy[:, wr]
        ok = np.isfinite(s).all(-1) & np.isfinite(e).all(-1) & np.isfinite(w).all(-1)
        if ok.any():
            out[ok, el] = _ik_elbow(s[ok], e[ok], w[ok], k1, k2)
    chains = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
    for base in (LH0, RH0):
        for ch in chains:
            k = 1 + rng.uniform(-finger, finger)
            idx = [base + i for i in ch]
            seg = xy[:, idx]
            ok = np.isfinite(seg).all((-1, -2))                         # skip frames with a gap inside the chain
            if ok.any():
                p = seg[ok]
                out_idx = p.copy()
                for m in range(1, len(ch)):
                    out_idx[:, m] = out_idx[:, m - 1] + k * (p[:, m] - p[:, m - 1])
                out[np.flatnonzero(ok)[:, None], np.array(idx)[None, :]] = out_idx
    eye = xy[:, EYE_CORNERS].mean(1)
    kx, ky = 1 + rng.uniform(-face, face), 1 + rng.uniform(-face, face)
    fi = np.arange(F0, F0 + NF)
    out[:, fi] = eye[:, None] + (xy[:, fi] - eye[:, None]) * np.array([kx, ky], np.float32)
    return out.astype(np.float32)


def keypoint_noise(xy, conf, rng, sigma_rel=0.02):
    _, sig, _ = anchor_frame(xy)
    out = xy.copy()
    hands = np.r_[LH0:LH0 + 21, RH0:RH0 + 21]
    out[:, hands] += (rng.standard_normal(out[:, hands].shape) * (sigma_rel * sig * (1 - conf[:, hands]))[..., None]).astype(np.float32)
    return out


def hand_dropout(xy, conf, rng, p=0.05):
    out, c = xy.copy(), conf.copy()
    for base in (LH0, RH0):
        drop = rng.random(len(xy)) < p
        out[drop, base:base + 21] = np.nan; c[drop, base:base + 21] = 0.0
    return out, c


def assert_anchors_untouched(idx):
    assert not set(idx) & set(ANCHORS)
