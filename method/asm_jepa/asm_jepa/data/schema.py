"""Canonical joint schema: COCO-WholeBody 133 raw -> 114 model joints in four streams.

Raw indexing (COCO-WholeBody): body 0-16, feet 17-22, face 23-90 (68-point iBUG order), left hand 91-111,
right hand 112-132 (left/right = the signer's anatomical side under the COCO convention; verify by overlay).
The model drops body nose/eyes/ears and the body wrists to avoid aliases of face/hand landmarks.
"""
from __future__ import annotations

import numpy as np

SCHEMA_NAME = "coco133_to_aslc114_v1"
N_RAW = 133

BODY = [5, 6, 7, 8]  # L shoulder, R shoulder, L elbow, R elbow
LH = list(range(91, 112))  # 21, wrist = local 0
RH = list(range(112, 133))  # 21, wrist = local 0
FACE = list(range(23, 91))  # 68
PART_ORDER = ["body", "lh", "rh", "face"]
PART_RAW = {"body": BODY, "lh": LH, "rh": RH, "face": FACE}
MODEL_TO_RAW = np.array(BODY + LH + RH + FACE, dtype=np.int64)
N_JOINTS = len(MODEL_TO_RAW)  # 114

PART_SLICES = {}
_start = 0
for _p in PART_ORDER:
    PART_SLICES[_p] = slice(_start, _start + len(PART_RAW[_p]))
    _start += len(PART_RAW[_p])
PART_SIZES = {p: len(PART_RAW[p]) for p in PART_ORDER}
N_PARTS = len(PART_ORDER)

# model-space indices of semantic landmarks
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 0, 1, 2, 3
SHOULDERS = [L_SHOULDER, R_SHOULDER]
ELBOWS = [L_ELBOW, R_ELBOW]
LH_WRIST = PART_SLICES["lh"].start
RH_WRIST = PART_SLICES["rh"].start
HAND_MCP_LOCAL = [5, 9, 13, 17]
FACE_NOSE_TIP_LOCAL = 30
NOSE_TIP = PART_SLICES["face"].start + FACE_NOSE_TIP_LOCAL
FACE_MOUTH_LOCAL = list(range(48, 68))
FACE_BROW_LOCAL = list(range(17, 27))
FACE_ANCHOR_LOCAL = FACE_MOUTH_LOCAL + FACE_BROW_LOCAL  # 30 landmarks for the face state anchor

# joints hidden by a raw mask of each part unit; shoulders are public normalisation anchors
MASKABLE = {
    "body": np.array(ELBOWS),
    "lh": np.arange(PART_SLICES["lh"].start, PART_SLICES["lh"].stop),
    "rh": np.arange(PART_SLICES["rh"].start, PART_SLICES["rh"].stop),
    "face": np.arange(PART_SLICES["face"].start, PART_SLICES["face"].stop),
}

CHANNELS = ["body_x", "body_y", "hand_local_x", "hand_local_y", "confidence", "observed", "local_valid"]
N_CHANNELS = len(CHANNELS)

RELATIONS = [("lh_wrist", "rh_wrist"), ("lh_wrist", "nose"), ("rh_wrist", "nose")]
RELATION_JOINTS = [(LH_WRIST, RH_WRIST), (LH_WRIST, NOSE_TIP), (RH_WRIST, NOSE_TIP)]
RELATION_PARTS = [(1, 2), (1, 3), (2, 3)]  # part indices in PART_ORDER that each relation depends on
N_RELATIONS = len(RELATIONS)
RELATION_CHANNELS = 5  # dx, dy, norm, confidence_pair, observed_pair

PATH_FAMILIES = ["left_wrist", "right_wrist", "left_minus_right_wrist"]
PATH_FAMILY_PARTS = {"left_wrist": (1,), "right_wrist": (2,), "left_minus_right_wrist": (1, 2)}

# state-anchor descriptor layout per part at one frame offset
ANCHOR_DIMS_PER_OFFSET = {"body": 8, "lh": 44, "rh": 44, "face": 60}


def raw_to_model(xy133: np.ndarray, score133: np.ndarray, observed133: np.ndarray):
    """Select the 114 model joints from [T,133,*] raw arrays (raw cache keeps all 133)."""
    if xy133.shape[1] != N_RAW or score133.shape[1] != N_RAW or observed133.shape[1] != N_RAW:
        raise ValueError(f"expected 133 raw joints, got {xy133.shape}, {score133.shape}, {observed133.shape}")
    return xy133[:, MODEL_TO_RAW], score133[:, MODEL_TO_RAW], observed133[:, MODEL_TO_RAW]


def joint_part_index() -> np.ndarray:
    out = np.zeros(N_JOINTS, dtype=np.int64)
    for i, p in enumerate(PART_ORDER):
        out[PART_SLICES[p]] = i
    return out
