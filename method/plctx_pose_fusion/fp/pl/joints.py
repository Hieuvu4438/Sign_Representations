"""Joint layout, channel ownership and mirroring (proposal12 v2.1 §6.3–§6.4).

103 joints selected from COCO-WholeBody (133):
  body  9 : nose, L/R shoulder, L/R elbow, L/R wrist, L/R hip            (COCO 0, 5, 6, 7, 8, 9, 10, 11, 12)
  face 52 : iBUG brows 17-26 (10), eyelids 37,38,40,41,43,44,46,47 (8), eye corners 36,39,42,45 (4),
            nose 27-35 (9), mouth 48-67 (20), chin 8 (1)                  (COCO 23 + iBUG index)
  LH   21 : COCO 91..111 (signer's left hand);   RH 21 : COCO 112..132
After handedness canonicalisation (mirroring left-dominant clips) the RIGHT side is always the dominant (D) side.

Channels (token types) and joint ownership — every joint belongs to at most one channel; shoulders are anchors
(never masked, never targets); body nose and hips are unused.
  0 D_CONF  dominant hand 21        1 D_LOC  dominant body wrist + elbow
  2 N_CONF  non-dominant hand 21    3 N_LOC  non-dominant body wrist + elbow
  4 F_UP    brows + eyelids (18)    5 F_MO   mouth + chin (21)          6 F_HD  eye corners + nose (13)
Features of F_UP / F_MO are expressed in the eye frame (owned by F_HD)  ->  DEPENDS[F_UP] = DEPENDS[F_MO] = {self, F_HD}
and the mask closure "F_HD hidden => F_UP, F_MO hidden" (§6.3).
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ layout
BODY_COCO = [0, 5, 6, 7, 8, 9, 10, 11, 12]
NOSE_B, LSHO, RSHO, LELB, RELB, LWRI, RWRI, LHIP, RHIP = range(9)
FACE_IBUG = (list(range(17, 27)) + [37, 38, 40, 41, 43, 44, 46, 47] + [36, 39, 42, 45] + list(range(27, 36))
             + list(range(48, 68)) + [8])
F0, NF = 9, len(FACE_IBUG)                      # face block start, 52
LH0, RH0 = F0 + NF, F0 + NF + 21                # 61, 82
J = RH0 + 21                                    # 103
COCO_INDEX = np.array(BODY_COCO + [23 + i for i in FACE_IBUG] + list(range(91, 112)) + list(range(112, 133)))
assert len(COCO_INDEX) == J == 103


def _face(ibug_list):
    return [F0 + FACE_IBUG.index(i) for i in ibug_list]


BROWS = _face(range(17, 27))
EYELIDS = _face([37, 38, 40, 41, 43, 44, 46, 47])
EYE_CORNERS = _face([36, 39, 42, 45])           # order: R-outer 36, R-inner 39, L-inner 42, L-outer 45 (iBUG)
NOSE = _face(range(27, 36))
NOSE_TIP = _face([30])[0]
MOUTH = _face(range(48, 68))
CHIN = _face([8])[0]
LH = list(range(LH0, LH0 + 21))
RH = list(range(RH0, RH0 + 21))

# ------------------------------------------------------------------ channels
CHANNELS = ("D_CONF", "D_LOC", "N_CONF", "N_LOC", "F_UP", "F_MO", "F_HD")
NC = len(CHANNELS)
D_CONF, D_LOC, N_CONF, N_LOC, F_UP, F_MO, F_HD = range(NC)
HAND_CH, FACE_CH = (D_CONF, D_LOC, N_CONF, N_LOC), (F_UP, F_MO, F_HD)
OWNED = {
    D_CONF: RH, D_LOC: [RWRI, RELB],
    N_CONF: LH, N_LOC: [LWRI, LELB],
    F_UP: BROWS + EYELIDS, F_MO: MOUTH + [CHIN], F_HD: EYE_CORNERS + NOSE,
}
ANCHORS = [LSHO, RSHO]
DEPENDS = {c: {c} for c in range(NC)}
DEPENDS[F_UP] = {F_UP, F_HD}
DEPENDS[F_MO] = {F_MO, F_HD}
# tokenizer type (CONF shared by D/N, LOC shared by D/N) and side embedding id (0 = dominant / face, 1 = non-dominant)
TOKEN_TYPE = np.array([0, 1, 0, 1, 2, 3, 4])
SIDE = np.array([0, 0, 1, 1, 0, 0, 0])
GROUP = np.array([0, 0, 0, 0, 1, 1, 1])        # 0 = hands, 1 = face (loss weight gamma)

_seen = sorted(j for c in OWNED for j in OWNED[c])
assert len(_seen) == len(set(_seen)), "a joint is owned by two channels"
assert not set(_seen) & set(ANCHORS)

# ------------------------------------------------------------------ mirroring (left-dominant -> right-dominant)
_IBUG_MIRROR = (list(range(16, -1, -1)) + list(range(26, 16, -1)) + [27, 28, 29, 30] + [35, 34, 33, 32, 31]
                + [45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40] + [54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55]
                + [64, 63, 62, 61, 60, 67, 66, 65])
assert sorted(_IBUG_MIRROR) == list(range(68))
MIRROR_PERM = np.arange(J)
for a, b in [(LSHO, RSHO), (LELB, RELB), (LWRI, RWRI), (LHIP, RHIP)]:
    MIRROR_PERM[a], MIRROR_PERM[b] = b, a
for k, i in enumerate(FACE_IBUG):
    MIRROR_PERM[F0 + k] = F0 + FACE_IBUG.index(_IBUG_MIRROR[i])
MIRROR_PERM[LH0:LH0 + 21] = np.arange(RH0, RH0 + 21)
MIRROR_PERM[RH0:RH0 + 21] = np.arange(LH0, LH0 + 21)
assert sorted(MIRROR_PERM.tolist()) == list(range(J))
assert (MIRROR_PERM[MIRROR_PERM] == np.arange(J)).all()

# ------------------------------------------------------------------ observation thresholds (shared with obs_jepa B)
# unnormalised COCO-WholeBody scores; 20%-quantile part means calibrated in an earlier project audit
# (method/obs_jepa/oj/data/landmarks.py PART_THR); joint threshold = 0.5 x part threshold.
PART_THR = {"body": 5.876, "face": 8.033, "hand": 1.692}
PART_OF_JOINT = np.array(["body"] * 9 + ["face"] * NF + ["lhand"] * 21 + ["rhand"] * 21)
PARTS = {"body": list(range(9)), "face": list(range(F0, F0 + NF)), "lhand": LH, "rhand": RH}
THR_OF_PART = {"body": PART_THR["body"], "face": PART_THR["face"], "lhand": PART_THR["hand"], "rhand": PART_THR["hand"]}

# hand skeleton (local hand indices 0..20)
FINGER_BONES = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
                (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)]
PALM_POLY = [1, 5, 13, 17]                      # thumb CMC + index/ring/pinky MCP (middle MCP h9 defines the axis)
FINGER_GROUPS = {                               # R1 hides the finger joints beyond the palm polygon (PIP/DIP/TIP; thumb MCP/IP/TIP)
    "thumb": [2, 3, 4], "index": [6, 7, 8], "middle": [10, 11, 12], "ring": [14, 15, 16], "pinky": [18, 19, 20],
    "index+middle": [6, 7, 8, 10, 11, 12], "middle+ring+pinky": [10, 11, 12, 14, 15, 16, 18, 19, 20],
    "thumb+index": [2, 3, 4, 6, 7, 8],
}
