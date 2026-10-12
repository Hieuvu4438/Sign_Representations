from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BodyCoordinates:
    xy: np.ndarray
    coordinate_valid: np.ndarray
    quality: np.ndarray
    center_px: np.ndarray
    scale_px: np.ndarray


def body_coordinates(xy_normalized, observed, quality, frame_sizes, *, epsilon_px=1.0, min_shoulder_quality=0.5):
    xy = np.asarray(xy_normalized, dtype=np.float64)
    observed, quality = np.asarray(observed, dtype=bool), np.asarray(quality, dtype=np.float64)
    sizes = np.asarray(frame_sizes, dtype=np.float64)  # [T,2] = width,height
    if xy.ndim != 3 or xy.shape[1:] != (49, 2) or observed.shape != xy.shape[:2] or quality.shape != observed.shape or sizes.shape != (len(xy), 2):
        raise ValueError("Expected xy[T,49,2], observed/quality[T,49], sizes[T,2]")
    if epsilon_px <= 0 or not np.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError("Invalid coordinate scale/frame size")
    if not np.isfinite(quality).all() or ((quality < 0) | (quality > 1)).any():
        raise ValueError("Quality must lie in [0,1]")
    pixels = xy * sizes[:, None, :]
    finite = np.isfinite(pixels).all(axis=-1)
    shoulders = (observed[:, 1:3] & finite[:, 1:3] & (quality[:, 1:3] >= min_shoulder_quality)).all(axis=1)
    center = pixels[:, 1:3].mean(axis=1)
    scale = np.linalg.norm(pixels[:, 1] - pixels[:, 2], axis=-1)
    normalization_valid = shoulders & np.isfinite(scale) & (scale > epsilon_px)
    valid = observed & finite & normalization_valid[:, None]
    # Invalid prerequisites never participate in arithmetic division.
    safe_center = np.where(normalization_valid[:, None], center, 0)
    safe_scale = np.where(normalization_valid, scale, 1)
    output = np.zeros_like(pixels)
    np.subtract(pixels, safe_center[:, None], out=output, where=valid[..., None])
    output /= safe_scale[:, None, None]
    return BodyCoordinates(output.astype(np.float32), valid, np.where(valid, quality, 0).astype(np.float32),
                           np.where(normalization_valid[:, None], center, np.nan),
                           np.where(normalization_valid, scale, np.nan))


def static_pose_streams(xy, quality, coordinate_valid):
    """Keep x/y and per-joint quality/validity; static handshape is never differenced."""
    from .topology import POSE_GROUPS
    valid = np.asarray(coordinate_valid, dtype=bool)
    values = np.concatenate([np.where(valid[..., None], xy, 0),
                             np.where(valid, quality, 0)[..., None], valid[..., None]], axis=-1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite valid pose values")
    return tuple(values[:, joints].reshape(len(values), -1).astype(np.float32) for joints in POSE_GROUPS)
