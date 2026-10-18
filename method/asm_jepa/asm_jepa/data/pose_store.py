"""Per-clip pose cache (`.npz`, allow_pickle=False) — contract of proposal15 §6.3.

Keys: xy float32 [T,133,2] source pixels; score float32 [T,133] raw extractor score (not calibrated);
observed bool [T,133]; timestamps_sec float64 [T]; frame_valid bool [T]; image_wh int32 [2] or [T,2];
source_frame_index int32 [T]; meta_json (0-d unicode) with schema/extractor/timestamp provenance.
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from asm_jepa import CACHE_VERSION

REQUIRED_KEYS = ("xy", "score", "observed", "timestamps_sec", "frame_valid", "image_wh", "source_frame_index")


@dataclass
class PoseClip:
    xy: np.ndarray
    score: np.ndarray
    observed: np.ndarray
    timestamps_sec: np.ndarray
    frame_valid: np.ndarray
    image_wh: np.ndarray
    source_frame_index: np.ndarray
    meta: dict

    @property
    def num_frames(self) -> int:
        return int(self.xy.shape[0])


def validate_arrays(c: PoseClip, n_joints: int = 133) -> None:
    T = c.xy.shape[0]
    checks = [
        (c.xy.dtype == np.float32 and c.xy.shape == (T, n_joints, 2), f"xy {c.xy.dtype} {c.xy.shape}"),
        (c.score.dtype == np.float32 and c.score.shape == (T, n_joints), f"score {c.score.dtype} {c.score.shape}"),
        (c.observed.dtype == np.bool_ and c.observed.shape == (T, n_joints), f"observed {c.observed.shape}"),
        (c.timestamps_sec.dtype == np.float64 and c.timestamps_sec.shape == (T,), "timestamps_sec"),
        (c.frame_valid.dtype == np.bool_ and c.frame_valid.shape == (T,), "frame_valid"),
        (c.image_wh.dtype == np.int32 and c.image_wh.shape in ((2,), (T, 2)), f"image_wh {c.image_wh.shape}"),
        (c.source_frame_index.dtype == np.int32 and c.source_frame_index.shape == (T,), "source_frame_index"),
    ]
    for ok, what in checks:
        if not ok:
            raise ValueError(f"pose cache contract violated: {what}")
    if c.observed.any() and not np.isfinite(c.xy[c.observed]).all():
        raise ValueError("observed joints must have finite coordinates")
    if T > 1 and np.any(np.diff(c.timestamps_sec) <= 0):
        raise ValueError("timestamps must be strictly increasing")


def write_clip(path, clip: PoseClip, compress: bool = True) -> None:
    validate_arrays(clip)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(clip.meta)
    meta.setdefault("cache_version", CACHE_VERSION)
    buf = io.BytesIO()
    saver = np.savez_compressed if compress else np.savez
    saver(buf, xy=clip.xy, score=clip.score, observed=clip.observed, timestamps_sec=clip.timestamps_sec,
          frame_valid=clip.frame_valid, image_wh=clip.image_wh, source_frame_index=clip.source_frame_index,
          meta_json=np.array(json.dumps(meta, sort_keys=True)))
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
    os.replace(tmp, path)


def read_clip(path) -> PoseClip:
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in REQUIRED_KEYS if k not in z.files]
        if missing:
            raise ValueError(f"{path}: missing keys {missing}")
        meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
        return PoseClip(z["xy"], z["score"], z["observed"], z["timestamps_sec"], z["frame_valid"],
                        z["image_wh"], z["source_frame_index"], meta)


def is_valid_cache(path) -> bool:
    try:
        validate_arrays(read_clip(path))
        return True
    except Exception:  # noqa: BLE001
        return False
