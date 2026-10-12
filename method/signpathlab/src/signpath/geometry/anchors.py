import numpy as np


def timing_motion(points, timestamps):
    points, times = np.asarray(points, dtype=np.float64), np.asarray(timestamps, dtype=np.float64)
    if points.ndim != 2 or len(points) < 2 or times.shape != (len(points),) or not np.isfinite(points).all() or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("Motion requires finite distinct time-ordered observations")
    increments = np.diff(points, axis=0)
    velocity = increments / np.diff(times)[:, None]
    return np.array([times[-1] - times[0], np.linalg.norm(increments, axis=-1).sum(),
                     np.sqrt(np.mean(np.sum(velocity ** 2, axis=-1)))])


def signature_descriptor(points, timestamps):
    from .logsig2 import logsig2_reference
    points = np.asarray(points, dtype=np.float64)
    timing = timing_motion(points, timestamps)
    return np.concatenate([logsig2_reference(points), points[0], points.mean(axis=0), points[-1], timing])
