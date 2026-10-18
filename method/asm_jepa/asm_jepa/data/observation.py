"""Audit-driven observation filters applied to the raw 133-joint arrays before the 133 -> 114 selection.

Why (pose audit, 2026-10-16, 300-400 train clips): raw RTMW SimCC scores (~0.8-9.5) do not separate valid from
invalid hands, so a score threshold cannot mark them unobserved:
- ~50 % of frames have a body wrist at the bottom image edge; the resting hand there is partly or fully outside
  the image (42 % of its joints on average) yet keeps a score of ~1-5.
- ~14 % of frames put both hands on the same physical hand. The wrongly placed hand's wrist lies far from the
  COCO *body* wrist of its own side (median 0.96 shoulder widths) while correctly placed hands are within
  0.08 shoulder widths in 75 % of frames.

Filters (each off by default; enabled in the final config):
- in_image_only: a joint whose pixel position is outside the image (+ margin x image size) is unobserved.
- hand_wrist_max_dist (> 0): a hand is unobserved in a frame when its wrist is further than this many shoulder
  widths from the body wrist of the same side (when that body wrist is observed).

Leakage: each decision reads only the joint itself, or the hand itself plus raw body wrists/shoulders. Body
wrists are not model joints and shoulders are public anchors, so a masked unit never changes the observation
status of a visible one (no cross-part dependency; see tests/test_observation_filter.py).
"""
from __future__ import annotations

import numpy as np

from asm_jepa.data.schema import LH, RH

COCO_L_SHOULDER, COCO_R_SHOULDER, COCO_L_WRIST, COCO_R_WRIST = 5, 6, 9, 10


def shoulder_width(xy: np.ndarray, observed: np.ndarray) -> float | None:
    ok = observed[:, COCO_L_SHOULDER] & observed[:, COCO_R_SHOULDER]
    if not ok.any():
        return None
    sw = float(np.median(np.linalg.norm(xy[ok, COCO_L_SHOULDER] - xy[ok, COCO_R_SHOULDER], axis=-1)))
    return sw if sw > 1e-6 else None


def observation_filter(xy: np.ndarray, observed: np.ndarray, image_wh: np.ndarray, data_cfg: dict) -> np.ndarray:
    """xy [T,133,2] pixels, observed [T,133] -> filtered observed (new array; inputs untouched)."""
    ob = observed.copy()
    if data_cfg.get("in_image_only", False):
        wh = np.asarray(image_wh, np.float64)
        wh = np.broadcast_to(wh if wh.ndim == 2 else wh[None], (len(xy), 2))[:, None, :]
        m = float(data_cfg.get("in_image_margin", 0.0)) * wh
        with np.errstate(invalid="ignore"):
            inside = ((xy >= -m) & (xy <= wh + m)).all(-1)
        ob &= inside
    tau = float(data_cfg.get("hand_wrist_max_dist", 0.0))
    if tau > 0:
        sw = shoulder_width(xy, ob)
        if sw is not None:
            for hand, body_wrist in ((LH, COCO_L_WRIST), (RH, COCO_R_WRIST)):
                w = hand[0]
                ref = ob[:, w] & ob[:, body_wrist]
                with np.errstate(invalid="ignore"):
                    far = ref & (np.linalg.norm(xy[:, w] - xy[:, body_wrist], axis=-1) > tau * sw)
                ob[np.ix_(far, hand)] = False
    return ob
