import numpy as np
from .anchors import timing_motion


def ordinary_motion_descriptor(points, timestamps):
    x, times = np.asarray(points, dtype=np.float64), np.asarray(timestamps, dtype=np.float64)
    timing = timing_motion(x, times)
    increments = np.diff(x, axis=0)
    velocity = increments / np.diff(times)[:, None]
    return np.concatenate([x[0], x.mean(axis=0), x[-1], x.std(axis=0),
                           x[-1] - x[0], velocity.mean(axis=0), np.sqrt(np.mean(velocity ** 2, axis=0)),
                           np.abs(increments).sum(axis=0), timing])
