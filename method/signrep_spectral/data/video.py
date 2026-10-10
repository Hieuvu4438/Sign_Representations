"""Video decode (PyAV, display order, explicit PTS) and upstream-equivalent preprocessing."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIZE = 224


def probe_video(path: str) -> dict:
    """Decode every frame (no pixel conversion) to get the true frame count and PTS."""
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        tb = float(s.time_base)
        pts = [f.pts for f in c.decode(s)]
        w, h = s.codec_context.width, s.codec_context.height
        rate = float(s.average_rate) if s.average_rate else None
    pts_sec = np.array([np.nan if p is None else p * tb for p in pts], dtype=np.float64)
    return {"num_frames": len(pts), "pts_sec": pts_sec, "width": w, "height": h, "average_rate": rate}


def decode_frames(path: str, start: int = 0, stop: Optional[int] = None, size: int = SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """RGB uint8 frames [n,size,size,3] for decoded ordinals [start, stop), exactly like upstream.

    Upstream example: cv2.VideoCapture -> BGR2RGB -> albumentations Resize(224,224)
    (cv2.INTER_LINEAR on uint8) -> Normalize. Pixels from cv2.VideoCapture equal
    PyAV ``to_ndarray('rgb24')`` on this machine (tested) and decode ~3-4x faster.
    Frame order/PTS validity was established with PyAV at audit time; the returned
    pts_sec are cv2 CAP_PROP_POS_MSEC values (diagnostic only).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    out, pts = [], []
    i = 0
    try:
        while stop is None or i < stop:
            if i < start:
                if not cap.grab():
                    break
                i += 1
                continue
            ok, bgr = cap.read()
            if not ok:
                break
            pts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out.append(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR))
            i += 1
    finally:
        cap.release()
    if not out:
        return np.zeros((0, size, size, 3), np.uint8), np.zeros((0,), np.float64)
    return np.stack(out), np.asarray(pts, dtype=np.float64)


def worker_init(_worker_id: int) -> None:
    """DataLoader workers: one cv2 thread each (avoids oversubscription with many workers)."""
    cv2.setNumThreads(1)


def normalize_frames(frames_uint8: torch.Tensor) -> torch.Tensor:
    """[...,T,H,W,3] uint8 -> [...,3,T,H,W] float (BCTHW for batched input)."""
    x = frames_uint8.float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype)
    x = (x - mean) / std
    return x.movedim(-1, -4)


def uniform_clock_ok(pts_sec: np.ndarray, max_rel_dev: float) -> bool:
    if len(pts_sec) < 2 or not np.isfinite(pts_sec).all():
        return False
    dt = np.diff(pts_sec)
    if (dt <= 0).any():
        return False
    med = float(np.median(dt))
    return med > 0 and float(np.max(np.abs(dt - med)) / med) <= max_rel_dev


def valid_clip_starts(pts_sec: np.ndarray, length: int, max_rel_dev: float) -> np.ndarray:
    """All starts whose `length` consecutive frames pass the clock policy (tail dropped)."""
    n = len(pts_sec)
    return np.array(
        [s for s in range(0, n - length + 1) if uniform_clock_ok(pts_sec[s : s + length], max_rel_dev)],
        dtype=np.int64,
    )


def window_starts(num_frames: int, length: int, stride: int) -> np.ndarray:
    if num_frames < length:
        return np.zeros((0,), np.int64)
    return np.arange(0, num_frames - length + 1, stride, dtype=np.int64)
