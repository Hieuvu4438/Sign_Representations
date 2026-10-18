"""Shared-view geometric augmentation and student-only observation jitter (§10.4).

The shared view (rotation +-5 deg, isotropic scale 0.9-1.1 about the shoulder root) is applied to the
body-frame coordinates seen by teacher, intact online branch and anchor targets alike. Student jitter
only perturbs observed joints and never turns a missing joint into an observation. The random draws do
not depend on coordinate values, so hidden values cannot influence what is sampled.
"""
from __future__ import annotations

import numpy as np


def shared_view_params(rng: np.random.Generator | None, rotation_degrees: float, scale_range) -> tuple[float, float]:
    if rng is None:
        return 0.0, 1.0
    theta = np.deg2rad(rng.uniform(-rotation_degrees, rotation_degrees)) if rotation_degrees > 0 else 0.0
    lo, hi = scale_range
    scale = rng.uniform(lo, hi) if hi > lo else float(lo)
    return float(theta), float(scale)


def apply_shared_view(xy_b: np.ndarray, observed: np.ndarray, theta: float, scale: float) -> np.ndarray:
    if theta == 0.0 and scale == 1.0:
        return xy_b
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], np.float64) * scale
    out = xy_b.astype(np.float64) @ rot.T
    return np.where(observed[..., None], out, 0.0).astype(np.float32)


def student_jitter(xy_b: np.ndarray, observed: np.ndarray, rng: np.random.Generator | None, sigma: float) -> np.ndarray:
    if rng is None or sigma <= 0:
        return xy_b
    noise = np.clip(rng.normal(0.0, sigma, size=xy_b.shape), -3 * sigma, 3 * sigma)
    return np.where(observed[..., None], xy_b + noise, 0.0).astype(np.float32)
