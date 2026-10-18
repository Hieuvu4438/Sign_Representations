"""J = 67 landmark schema (proposal11 §8.1) built from the local COCO-WholeBody 2-D keypoints.

Declared adaptation: the proposal writes the schema with MediaPipe indices; ASL Citizen ships COCO-WholeBody
(133 joints, iBUG-68 face). The mapping below keeps the same anatomy:
  body 7  : nose, L/R shoulder, L/R elbow, L/R wrist      (COCO 0, 5, 6, 7, 8, 9, 10)
  LH 21   : COCO 91..111  (subject's left hand)
  RH 21   : COCO 112..132 (subject's right hand)
  face 18 : brows iBUG 17..26 (10), mouth corners + inner lips iBUG 48, 54, 62, 66 (4),
            eye corners iBUG 36, 39, 42, 45 (4)   [= MediaPipe 70..300 / 61,291,13,14 / 33,133,362,263]
Coordinates: COCO-WholeBody stores x/W and y/H separately -> converted to source pixels first (§6.4).
Per-joint interface (p, a, q, r) of §6.5: pixel xy, observed flag, available quality, reason code.
Quality = detector heat-map score scaled by the part threshold; it is *uncalibrated* (not a probability).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

from oj.paths import ASLC_KP

J = 67
BODY = np.arange(0, 7); LH = np.arange(7, 28); RH = np.arange(28, 49); FACE = np.arange(49, 67)
PARTS = {"body": BODY, "lh": LH, "rh": RH, "face": FACE}
PART_NAMES = ("body", "face", "lh", "rh")
PART_OF = np.zeros(J, np.int64)
for _i, _n in enumerate(PART_NAMES):
    PART_OF[PARTS[_n]] = _i

_FACE_IBUG = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 48, 54, 62, 66, 36, 39, 42, 45]
COCO_INDEX = np.array([0, 5, 6, 7, 8, 9, 10] + list(range(91, 112)) + list(range(112, 133)) + [23 + k for k in _FACE_IBUG])
assert len(COCO_INDEX) == J

# body node ids
NOSE, LSH, RSH, LEL, REL, LWR, RWR = range(7)
LH_ROOT, RH_ROOT = int(LH[0]), int(RH[0])
DUPLICATE_WRISTS = ((LWR, LH_ROOT), (RWR, RH_ROOT))  # body wrist <-> hand root (§8.1, tie masks)

# parent of every node for directed parent-bone features (-1 = root, no bone)
PARENT = np.full(J, -1, np.int64)
PARENT[[LSH, RSH]] = NOSE; PARENT[LEL] = LSH; PARENT[REL] = RSH; PARENT[LWR] = LEL; PARENT[RWR] = REL
HAND_CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
for base in (LH[0], RH[0]):
    for ch in HAND_CHAINS:
        for a, b in zip(ch[:-1], ch[1:]):
            PARENT[base + b] = base + a
_f = {k: 49 + i for i, k in enumerate(_FACE_IBUG)}
for chain in ([17, 18, 19, 20, 21], [22, 23, 24, 25, 26]):
    for a, b in zip(chain[:-1], chain[1:]):
        PARENT[_f[b]] = _f[a]
PARENT[_f[39]] = _f[36]; PARENT[_f[42]] = _f[45]
PARENT[_f[62]] = _f[48]; PARENT[_f[54]] = _f[62]; PARENT[_f[66]] = _f[54]
FACE_EYE_ANCHORS = (_f[36], _f[45])  # face centre/scale for local face branch
MOUTH = (_f[48], _f[54], _f[62], _f[66]); BROWS = tuple(_f[k] for k in range(17, 27))

# undirected anatomical edges (spatial graph); mouth ring closes 66-48
EDGES = [(int(PARENT[j]), j) for j in range(J) if PARENT[j] >= 0] + [(_f[66], _f[48])]

# score thresholds (unnormalised COCO-WholeBody scores), 20%-quantile part means calibrated on two held-out
# ASL Citizen train signers in an earlier project audit (asc_jepa/roi.py); joint threshold = 0.5 x part.
PART_THR = {"body": 5.876, "face": 8.033, "lh": 1.692, "rh": 1.692}

REASON_MEASURED, REASON_MISSING, REASON_SYNTHETIC, REASON_PADDING = 0, 1, 2, 3


@dataclass
class Landmarks:
    xy: np.ndarray        # [F, J, 2] source pixels (nan when missing)
    observed: np.ndarray  # [F, J] bool
    quality: np.ndarray   # [F, J] float in [0, 1], uncalibrated
    reason: np.ndarray    # [F, J] int
    width: int
    height: int


def load_landmarks(sample_id: str, width: int, height: int, root=None) -> Landmarks:
    d = pickle.load(open((ASLC_KP if root is None else root) / f"{sample_id}.pkl", "rb"))
    F = len(d["keypoints"])
    xy = np.full((F, J, 2), np.nan, np.float32); sc = np.zeros((F, J), np.float32)
    for t, (k, s) in enumerate(zip(d["keypoints"], d["scores"])):
        k, s = np.asarray(k), np.asarray(s)
        if k.ndim == 3 and k.shape[0] == 1:  # single detected person only
            xy[t] = k[0, COCO_INDEX] * np.array([width, height], np.float32)
            sc[t] = s[0, COCO_INDEX]
    obs = np.isfinite(xy).all(-1)
    qual = np.zeros((F, J), np.float32)
    for name, idx in PARTS.items():
        thr = PART_THR[name]
        part_ok = sc[:, idx].mean(1) >= thr
        joint_ok = sc[:, idx] >= 0.5 * thr
        obs[:, idx] &= part_ok[:, None] & joint_ok
        qual[:, idx] = np.clip(sc[:, idx] / (2.0 * thr), 0.0, 1.0)
    reason = np.where(obs, REASON_MEASURED, REASON_MISSING).astype(np.int64)
    xy[~obs] = np.nan
    return Landmarks(xy, obs, np.where(obs, qual, 0.0).astype(np.float32), reason, width, height)


def handedness_check(lm: Landmarks, max_px: float = 40.0) -> float:
    """Fraction of frames where body wrists and hand roots agree (chirality/mapping overlay proxy)."""
    ok, n = 0, 0
    for bw, hr in DUPLICATE_WRISTS:
        m = lm.observed[:, bw] & lm.observed[:, hr]
        n += int(m.sum()); ok += int((np.linalg.norm(lm.xy[m, bw] - lm.xy[m, hr], axis=-1) < max_px).sum())
    return ok / max(n, 1)
