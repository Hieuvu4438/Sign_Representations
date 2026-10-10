"""Raw pose adapter: ASL Citizen local COCO-WholeBody (133 kpt, 2-D) pickles -> internal contract.

Observed schema (audited on this machine, see records/discovery.md):
  dict(keypoints=list[F] of float64 [P,133,2], scores=list[F] of float32 [P,133])
  P = detected persons per frame (mostly 1; 0/2/3 occur). Coordinates are image
  normalized (x/W, y/H; values slightly outside [0,1] occur). Scores are
  unnormalized backend scores (observed range ~0.3..10), NOT probabilities.
COCO-WholeBody order: body 0-16, feet 17-22, face 23-90, left hand 91-111,
right hand 112-132; hand joints follow wrist/thumb/index/middle/ring/pinky order
(0 wrist, 9 middle-finger MCP), same topology as MediaPipe's 21-joint hand.
No z coordinate exists, so the canonical targets here are 2-D (xy).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

SCHEMA_ID = "asl_citizen_coco_wholebody133_xy_v1"
LEFT_HAND = slice(91, 112)
RIGHT_HAND = slice(112, 133)


def read_raw_pose(pkl_path: str | Path, score_threshold: float, coord_margin: float = 0.05) -> dict:
    """Trusted local pickle (dataset owner's keypoint cache), read-only, schema-checked."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict) or set(d) != {"keypoints", "scores"}:
        raise ValueError(f"unexpected pose schema keys in {pkl_path}")
    kps, scs = d["keypoints"], d["scores"]
    if len(kps) != len(scs):
        raise ValueError("keypoints/scores length mismatch")
    F = len(kps)
    hand_xy = np.zeros((F, 2, 21, 2), np.float32)
    hand_score = np.zeros((F, 2, 21), np.float32)
    person_count = np.zeros((F,), np.int64)
    for t, (k, s) in enumerate(zip(kps, scs)):
        k = np.asarray(k)
        s = np.asarray(s)
        if k.ndim != 3 or k.shape[1:] != (133, 2) or s.shape != k.shape[:2]:
            raise ValueError(f"frame {t}: bad shapes {k.shape} {s.shape}")
        person_count[t] = k.shape[0]
        if k.shape[0] == 1:
            hand_xy[t, 0], hand_xy[t, 1] = k[0, LEFT_HAND], k[0, RIGHT_HAND]
            hand_score[t, 0], hand_score[t, 1] = s[0, LEFT_HAND], s[0, RIGHT_HAND]
    single = person_count == 1  # 0 or >1 persons: identity/handedness ambiguous -> invalid
    finite = np.isfinite(hand_xy).all(-1)
    in_bounds = (
        (hand_xy >= -coord_margin) & (hand_xy <= 1 + coord_margin)
    ).all(-1)
    hand_present = single[:, None] & (hand_score.mean(-1) >= score_threshold)
    track_valid = np.repeat(single[:, None], 2, axis=1)
    joint_observed = hand_present[:, :, None] & track_valid[:, :, None] & finite & in_bounds & (hand_score >= score_threshold)
    hand_xy = np.where(joint_observed[..., None], hand_xy, 0.0).astype(np.float32)
    return {
        "hand_xy": hand_xy,
        "hand_score": hand_score,
        "hand_present": hand_present,
        "track_valid": track_valid,
        "joint_observed": joint_observed,
        "person_count": person_count,
        "num_frames": F,
        "pose_schema_id": SCHEMA_ID,
        "coordinate_system": "image_normalized_xy(x/W,y/H)",
        "confidence_mode": f"backend_score_unnormalized>={score_threshold}",
    }
