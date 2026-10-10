"""Geometry-only teacher states for the CIP-Sign gate test (proposal5 §6.5, adapted).

DEVIATIONS from proposal5 (declared): the local ASL Citizen teacher is 2-D COCO-WholeBody
(133 kpts, unnormalised scores, unknown estimator) instead of MediaPipe Tasks 3-D; no
appearance (DINO) block; face block uses landmark geometry instead of blendshapes; native
frame rate (~30 fps) instead of a resampled 30 Hz grid. Parts fixed LH=0, RH=1, FACE=2, BODY=3.

State per part (D = 102, zero-padded; validity mask per dimension):
  LH/RH : 20 joints rel. wrist / sH (40) + wrist rel. torso / sT (2) + log(sH/sT) (1)  = 43
  FACE  : 51 points (brows, nose, eyes, mouth) rel. face centre / face scale (102)     = 102
          (face centre rel. torso is NOT included: it would duplicate BODY head info)
  BODY  : nose, ears, shoulders, hips rel. torso / sT (14) + shoulder dir (2) + log(sT) (1) = 17
Hands/wrists never enter BODY or FACE.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

KP = Path("/home/dongvk/datasets/ASL_Citizen/keypoints")
D = 102
LH, RH = np.arange(91, 112), np.arange(112, 133)
FACE_PTS = np.r_[23 + 17 : 23 + 36, 23 + 36 : 23 + 48, 23 + 48 : 23 + 68]  # brows+nose, eyes, mouth = 19+12+20
assert len(FACE_PTS) == 51
BODY_PTS = [0, 3, 4, 5, 6, 11, 12]
PART_DIMS = [43, 43, 102, 17]


def _hand(k, s, sh, sT, thr):
    w = k[0]
    sH = np.linalg.norm(k[9] - k[0])
    v = np.zeros(D, np.float32)
    m = np.zeros(D, bool)
    if sH < 1e-3 or s.mean() < thr:
        return v, m
    v[:40] = ((k[1:] - w) / sH).ravel()
    m[:40] = True
    if sT > 1e-3:
        v[40:42] = (w - sh) / sT
        v[42] = np.log(sH / sT)
        m[40:43] = True
    return v, m


def frame_states(k, s, thr_hand, thr_face, thr_body):
    """k [133,2], s [133] for one detected person -> states [4,D], valid [4,D]."""
    st = np.zeros((4, D), np.float32)
    va = np.zeros((4, D), bool)
    sh = (k[5] + k[6]) / 2
    sT = np.linalg.norm(k[5] - k[6])
    body_ok = s[[5, 6]].min() >= thr_body and sT > 1e-3
    st[0], va[0] = _hand(k[LH], s[LH], sh, sT if body_ok else 0.0, thr_hand)
    st[1], va[1] = _hand(k[RH], s[RH], sh, sT if body_ok else 0.0, thr_hand)
    f = k[FACE_PTS]
    if s[FACE_PTS].mean() >= thr_face:
        c = f.mean(0)
        sc = np.sqrt(((f - c) ** 2).sum(1).mean())
        if sc > 1e-4:
            st[2] = ((f - c) / sc).ravel()
            va[2] = True
    if body_ok:
        b = (k[BODY_PTS] - sh) / sT
        vis = s[BODY_PTS] >= thr_body
        st[3, :14] = b.ravel()
        va[3, :14] = np.repeat(vis, 2)
        d = (k[6] - k[5]) / sT
        st[3, 14:16] = d
        st[3, 16] = np.log(sT)
        va[3, 14:17] = True
    return st, va


def video_states(sample_id, thr, kp_dir=None):
    """-> states [T,4,D] float32, valid [T,4,D] bool (frames with != 1 person invalid)."""
    d = pickle.load(open((kp_dir or KP) / f"{sample_id}.pkl", "rb"))
    T = len(d["keypoints"])
    S = np.zeros((T, 4, D), np.float32)
    V = np.zeros((T, 4, D), bool)
    for t, (k, s) in enumerate(zip(d["keypoints"], d["scores"])):
        k, s = np.asarray(k), np.asarray(s)
        if k.ndim == 3 and k.shape[0] == 1:
            S[t], V[t] = frame_states(k[0], s[0], *thr)
    return S, V


def score_quantiles(sample_ids, q=0.2, kp_dir=None):
    """Calibration of unnormalised detector scores: per-part q-quantile of mean scores
    over calibration videos (scores are not probabilities; threshold is a coverage rule)."""
    hs, fs, bs = [], [], []
    for sid in sample_ids:
        d = pickle.load(open((kp_dir or KP) / f"{sid}.pkl", "rb"))
        for s in d["scores"]:
            s = np.asarray(s)
            if s.ndim == 2 and s.shape[0] == 1:
                s = s[0]
                hs += [s[LH].mean(), s[RH].mean()]
                fs.append(s[FACE_PTS].mean())
                bs += list(s[[5, 6]])
    return float(np.quantile(hs, q)), float(np.quantile(fs, q)), float(np.quantile(bs, q))
