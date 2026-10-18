"""Timestamp-based windows and letterbox geometry (proposal11 §6.3).

- effective fps = min(source fps, target fps); frames are chosen by time (no upsampling, no interpolation);
- letterbox: keep aspect ratio, resize the full frame to S x S with symmetric padding; the pixel/token valid map
  excludes padding from every loss; landmarks are mapped with the same affine transform.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Letterbox:
    scale: float
    ox: float
    oy: float
    size: int
    src_w: int
    src_h: int

    def apply(self, xy):
        return xy * self.scale + np.array([self.ox, self.oy], np.float32)

    def invert(self, xy):
        return (xy - np.array([self.ox, self.oy], np.float32)) / self.scale

    def pixel_valid(self):
        v = np.zeros((self.size, self.size), bool)
        x0, y0 = int(round(self.ox)), int(round(self.oy))
        v[y0:y0 + int(round(self.src_h * self.scale)), x0:x0 + int(round(self.src_w * self.scale))] = True
        return v

    def token_valid(self, patch=16):
        """token valid iff its patch centre lies inside the real image area."""
        g = self.size // patch
        c = (np.arange(g) + 0.5) * patch
        xs = (c >= self.ox) & (c < self.ox + self.src_w * self.scale)
        ys = (c >= self.oy) & (c < self.oy + self.src_h * self.scale)
        return ys[:, None] & xs[None, :]  # [g, g]


def make_letterbox(w, h, size):
    s = size / max(w, h)
    return Letterbox(s, (size - w * s) / 2.0, (size - h * s) / 2.0, size, w, h)


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    meta = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(cap.get(cv2.CAP_PROP_FPS) or 30.0), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return meta


def window_indices(n_frames, fps, T, target_fps, start_s):
    """source frame indices for T samples at effective fps from start_s; -1 = beyond the clip (padding)."""
    eff = min(fps, target_fps)
    ts = start_s + np.arange(T) / eff
    idx = np.round(ts * fps).astype(np.int64)
    idx[idx >= n_frames] = -1
    return idx, ts, eff


def random_start(n_frames, fps, T, target_fps, rng):
    dur = n_frames / fps; need = T / min(fps, target_fps)
    return float(rng.uniform(0.0, max(0.0, dur - need)))


def decode(path, indices):
    """decode the requested (sorted, possibly repeated / -1) frame indices; returns uint8 [T,H,W,3] RGB and valid flags."""
    want = sorted({int(i) for i in indices if i >= 0})
    cap = cv2.VideoCapture(str(path)); got = {}; cur = 0
    for i in want:
        while cur < i:
            if not cap.grab():
                break
            cur += 1
        ok, fr = cap.read()
        if not ok:
            break
        got[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB); cur = i + 1
    cap.release()
    h, w = (next(iter(got.values())).shape[:2] if got else (0, 0))
    out = np.zeros((len(indices), h, w, 3), np.uint8); valid = np.zeros(len(indices), bool)
    for k, i in enumerate(indices):
        if i in got:
            out[k], valid[k] = got[i], True
    return out, valid


def letterbox_frames(frames, lb: Letterbox):
    T = len(frames)
    out = np.zeros((T, lb.size, lb.size, 3), np.uint8)
    nw, nh = int(round(lb.src_w * lb.scale)), int(round(lb.src_h * lb.scale))
    x0, y0 = int(round(lb.ox)), int(round(lb.oy))
    for t in range(T):
        if frames[t].size:
            out[t, y0:y0 + nh, x0:x0 + nw] = cv2.resize(frames[t], (nw, nh), interpolation=cv2.INTER_AREA)
    return out
