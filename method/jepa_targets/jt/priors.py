"""Frame-level target priors from local COCO-WholeBody 2-D keypoints (133 joints, normalised xy).

Declared adaptation: SignRep regresses priors from a 3-D sign-specific pose lifter (61 kpts, angles,
distances) that is not available locally; here every prior is 2-D. Body-centred frame: origin =
shoulder midpoint, unit = shoulder width, y up. Every group returns (value [F, D], mask [F, D]).

Groups
  POSE  SignRep-like: upper-body keypoints, per-hand joints rel. wrist / palm size, finger bend angles
        (cos, sin), intra-hand fingertip distances, elbow angles.
  REL   inter-part relations (phonological "location"/contact): wrist & index tip rel. nose and rel.
        chest, hand->face minimum distance, hand-hand vector and minimum distance.
  NM    non-manual: mouth (20 pts) and brows (10 pts) in a face frame, eye/mouth openness, head roll/yaw proxy.
  MOT   movement: wrist and index-tip velocities (+ wrist acceleration), hand-shape change speed.
"""
from __future__ import annotations

import numpy as np

LH, RH = np.arange(91, 112), np.arange(112, 133)
FACE0 = 23  # face landmark k (0..67, iBUG order) -> index FACE0 + k
UPPER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# score thresholds (unnormalised scores) as calibrated in asc_jepa/roi.py
THR = {"hand": 1.692, "face": 8.033, "body": 5.876}
FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
TIPS = [4, 8, 12, 16, 20]

DIMS = {"POSE": 26 + 2 * (42 + 30 + 10) + 4, "REL": 2 * 7 + 3, "NM": 40 + 20 + 3 + 2, "MOT": 2 * 7}


def _yup(v):
    v = v.copy(); v[..., 1] *= -1; return v


def _frame(K, Sc):
    sh = (K[:, 5] + K[:, 6]) / 2
    sw = np.linalg.norm(K[:, 5] - K[:, 6], axis=-1)
    okb = (Sc[:, UPPER].mean(1) >= THR["body"]) & np.isfinite(sw) & (sw > 1e-3)
    return sh, np.where(okb, sw, 1.0), okb


def _hand_ok(K, Sc, idx):
    h = K[:, idx]
    palm = np.linalg.norm(h[:, 5] - h[:, 17], axis=-1)
    ok = np.isfinite(h).all((1, 2)) & (Sc[:, idx].mean(1) >= THR["hand"]) & (palm > 1e-3)
    return h, np.where(ok, palm, 1.0), ok


def _angle(a, b, c):
    """bend angle at b for segments a-b-c, returned as (cos, sin) of the signed 2-D angle."""
    u, v = a - b, c - b
    nu = np.linalg.norm(u, axis=-1) + 1e-8; nv = np.linalg.norm(v, axis=-1) + 1e-8
    cos = (u * v).sum(-1) / (nu * nv)
    sin = (u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]) / (nu * nv)
    return np.stack([cos, sin], -1)


def pose_group(K, Sc):
    F = len(K); out = np.zeros((F, DIMS["POSE"]), np.float32); m = np.zeros_like(out, bool)
    sh, sw, okb = _frame(K, Sc)
    up = _yup((K[:, UPPER] - sh[:, None]) / sw[:, None, None]).reshape(F, -1)
    out[:, :26] = np.nan_to_num(up); m[:, :26] = okb[:, None] & np.isfinite(up).all(1)[:, None]
    o = 26
    for idx in (LH, RH):
        h, palm, ok = _hand_ok(K, Sc, idx)
        rel = _yup((h - h[:, :1]) / palm[:, None, None]).reshape(F, -1)
        out[:, o:o + 42] = np.nan_to_num(rel); m[:, o:o + 42] = ok[:, None]; o += 42
        angs = [_angle(h[:, a], h[:, b], h[:, c]) for f in FINGERS for a, b, c in ((0, f[0], f[1]), (f[0], f[1], f[2]), (f[1], f[2], f[3]))]
        angs = np.concatenate(angs, -1)
        out[:, o:o + 30] = np.nan_to_num(angs); m[:, o:o + 30] = ok[:, None]; o += 30
        tips = h[:, TIPS]
        d = [np.linalg.norm(tips[:, i] - tips[:, j], axis=-1) for i in range(5) for j in range(i + 1, 5)]
        out[:, o:o + 10] = np.nan_to_num(np.stack(d, -1) / palm[:, None]); m[:, o:o + 10] = ok[:, None]; o += 10
    el = np.concatenate([_angle(K[:, 5], K[:, 7], K[:, 9]), _angle(K[:, 6], K[:, 8], K[:, 10])], -1)
    out[:, o:o + 4] = np.nan_to_num(el); m[:, o:o + 4] = okb[:, None]
    return out, m


def _face(K, Sc):
    f = K[:, FACE0:FACE0 + 68]
    okf = np.isfinite(f).all((1, 2)) & (Sc[:, FACE0:FACE0 + 68].mean(1) >= THR["face"])
    fw = np.linalg.norm(f[:, 16] - f[:, 0], axis=-1)
    okf &= fw > 1e-3
    return f, np.where(okf, fw, 1.0), okf


def rel_group(K, Sc):
    F = len(K); out = np.zeros((F, DIMS["REL"]), np.float32); m = np.zeros_like(out, bool)
    sh, sw, okb = _frame(K, Sc)
    f, fw, okf = _face(K, Sc)
    nose = K[:, 0]
    hands = [_hand_ok(K, Sc, idx) for idx in (LH, RH)]
    o = 0
    for h, palm, ok in hands:
        okn = ok & okb
        for p in (h[:, 0], h[:, 8]):
            out[:, o:o + 2] = np.nan_to_num(_yup((p - nose) / sw[:, None])); m[:, o:o + 2] = okn[:, None]; o += 2
        out[:, o:o + 2] = np.nan_to_num(_yup((h[:, 0] - sh) / sw[:, None])); m[:, o:o + 2] = okn[:, None]; o += 2
        dmin = np.linalg.norm(h[:, :, None] - f[:, None], axis=-1).min((1, 2)) / sw
        out[:, o] = np.nan_to_num(dmin); m[:, o] = okn & okf; o += 1
    (hl, _, okl), (hr, _, okr) = hands
    ok2 = okl & okr & okb
    out[:, o:o + 2] = np.nan_to_num(_yup((hr[:, 0] - hl[:, 0]) / sw[:, None])); m[:, o:o + 2] = ok2[:, None]; o += 2
    out[:, o] = np.nan_to_num(np.linalg.norm(hl[:, :, None] - hr[:, None], axis=-1).min((1, 2)) / sw); m[:, o] = ok2
    return out, m


def nm_group(K, Sc):
    F = len(K); out = np.zeros((F, DIMS["NM"]), np.float32); m = np.zeros_like(out, bool)
    f, fw, okf = _face(K, Sc)
    mouth, brows = f[:, 48:68], f[:, 17:27]
    eyes_c = (f[:, 36:42].mean(1) + f[:, 42:48].mean(1)) / 2
    out[:, :40] = np.nan_to_num(_yup((mouth - mouth.mean(1, keepdims=True)) / fw[:, None, None]).reshape(F, -1))
    out[:, 40:60] = np.nan_to_num(_yup((brows - eyes_c[:, None]) / fw[:, None, None]).reshape(F, -1))
    eo = [np.linalg.norm(f[:, a] - f[:, b], axis=-1) for a, b in ((37, 41), (44, 46), (62, 66))]
    out[:, 60:63] = np.nan_to_num(np.stack(eo, -1) / fw[:, None])
    roll = f[:, 45] - f[:, 36]
    out[:, 63] = np.nan_to_num(np.arctan2(-roll[:, 1], roll[:, 0]))
    out[:, 64] = np.nan_to_num(((f[:, 30] - eyes_c)[:, 0]) / fw)
    m[:] = okf[:, None]
    return out, m


def mot_group(K, Sc):
    F = len(K); out = np.zeros((F, DIMS["MOT"]), np.float32); m = np.zeros_like(out, bool)
    sh, sw, okb = _frame(K, Sc)
    o = 0
    for idx in (LH, RH):
        h, palm, ok = _hand_ok(K, Sc, idx)
        ok = ok & okb
        w = _yup((h[:, 0] - sh) / sw[:, None]); t = _yup((h[:, 8] - sh) / sw[:, None])
        shape = (h - h[:, :1]) / palm[:, None, None]
        vw = np.zeros_like(w); vw[1:] = w[1:] - w[:-1]
        aw = np.zeros_like(w); aw[1:-1] = w[2:] - 2 * w[1:-1] + w[:-2]
        vt = np.zeros_like(t); vt[1:] = t[1:] - t[:-1]
        vs = np.zeros(F, np.float32); vs[1:] = np.linalg.norm(shape[1:] - shape[:-1], axis=-1).mean(-1)
        okv = np.zeros(F, bool); okv[1:] = ok[1:] & ok[:-1]
        oka = np.zeros(F, bool); oka[1:-1] = ok[2:] & ok[1:-1] & ok[:-2]
        for v, mk in ((vw, okv), (aw, oka), (vt, okv)):
            out[:, o:o + 2] = np.nan_to_num(v); m[:, o:o + 2] = mk[:, None]; o += 2
        out[:, o] = np.nan_to_num(vs); m[:, o] = okv; o += 1
    return out, m


GROUPS = {"POSE": pose_group, "REL": rel_group, "NM": nm_group, "MOT": mot_group}


def compute(K, Sc, groups):
    """K [F,133,2], Sc [F,133] -> {group: (value [F,D], mask [F,D])}."""
    return {g: GROUPS[g](K, Sc) for g in groups}
