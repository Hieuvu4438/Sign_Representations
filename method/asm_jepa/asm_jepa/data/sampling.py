"""Full-clip temporal sampling (§7.2): 64 samples spanning the whole clip, endpoints kept, right padding."""
from __future__ import annotations

import numpy as np


def full_clip_indices(n_native: int, max_frames: int, rng: np.random.Generator | None = None,
                      jitter_fraction: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """-> (indices [max_frames] int64 with -1 for padding, time_valid [max_frames] bool).

    n_native > max_frames: one sample per evenly spaced bin from first to last frame (deterministic linspace
    at eval; train jitters interior samples by at most `jitter_fraction` of a bin, keeping order, endpoints
    and uniqueness). n_native <= max_frames: every frame once, right-padded.
    """
    if n_native <= 0:
        return np.full(max_frames, -1, np.int64), np.zeros(max_frames, bool)
    if n_native <= max_frames:
        idx = np.full(max_frames, -1, np.int64)
        idx[:n_native] = np.arange(n_native)
        valid = np.zeros(max_frames, bool)
        valid[:n_native] = True
        return idx, valid
    base = np.linspace(0.0, n_native - 1, max_frames)
    idx = np.round(base).astype(np.int64)
    if rng is not None and jitter_fraction > 0:
        step = (n_native - 1) / (max_frames - 1)
        offs = rng.uniform(-jitter_fraction, jitter_fraction, size=max_frames) * step
        offs[0] = offs[-1] = 0.0
        cand = np.clip(np.round(base + offs).astype(np.int64), 0, n_native - 1)
        # keep strictly increasing order; any collision falls back to the deterministic index of that bin
        ok = np.ones(max_frames, bool)
        ok[1:] = np.diff(cand) > 0
        ok[:-1] &= np.diff(cand) > 0
        cand[~ok] = idx[~ok]
        if np.all(np.diff(cand) > 0):
            idx = cand
    return idx, np.ones(max_frames, bool)


def patch_times(timestamps: np.ndarray, time_valid: np.ndarray, patch_frames: int):
    """Mean real-frame timestamp per patch + whether the patch has >=1 real frame."""
    T = len(timestamps)
    n_p = T // patch_frames
    ts = timestamps.reshape(n_p, patch_frames)
    tv = time_valid.reshape(n_p, patch_frames)
    cnt = tv.sum(1)
    mean = np.where(cnt > 0, (ts * tv).sum(1) / np.maximum(cnt, 1), 0.0)
    return mean, cnt > 0
