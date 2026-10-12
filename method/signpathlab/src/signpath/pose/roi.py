from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Crop:
    image: np.ndarray
    affine: np.ndarray  # original pixel xy -> output xy
    box: tuple[int, int, int, int]
    padding_fraction: float


def landmark_box(points_px, *, margin=1.5, min_size=8, square=True):
    points = np.asarray(points_px)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        return None
    lo, hi = points.min(axis=0), points.max(axis=0)
    extent = hi - lo
    if min(extent) < min_size:
        return None
    if square:
        extent[:] = extent.max()
    center = (hi + lo) / 2
    extent = extent * margin
    return tuple(np.concatenate([center - extent / 2, center + extent / 2]))


def letterbox(frame_rgb, box=None, *, output_size=224, fill=0):
    import cv2
    image = np.asarray(frame_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or output_size < 1:
        raise ValueError("Expected uint8 RGB image and positive output size")
    height, width = image.shape[:2]
    if box is None:
        box = (0, 0, width, height)
    if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Invalid ROI box")
    x0, y0, x1, y1 = int(np.floor(box[0])), int(np.floor(box[1])), int(np.ceil(box[2])), int(np.ceil(box[3]))
    crop_width, crop_height = x1 - x0, y1 - y0
    if crop_width > 4 * width or crop_height > 4 * height:
        raise ValueError("Implausibly large ROI")
    canvas = np.full((crop_height, crop_width, 3), fill, dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if sx0 >= sx1 or sy0 >= sy1:
        raise ValueError("ROI does not intersect frame")
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    scale = output_size / max(crop_width, crop_height)
    resized_w, resized_h = max(1, round(crop_width * scale)), max(1, round(crop_height * scale))
    resized = cv2.resize(canvas, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    left, top = (output_size - resized_w) // 2, (output_size - resized_h) // 2
    result = np.full((output_size, output_size, 3), fill, dtype=np.uint8)
    result[top:top + resized_h, left:left + resized_w] = resized
    # Rounded raster dimensions mean the realized affine has separate subpixel scales.
    sx, sy = resized_w / crop_width, resized_h / crop_height
    affine = np.array([[sx, 0, left - sx * x0], [0, sy, top - sy * y0]], dtype=np.float64)
    observed_area = (sx1 - sx0) * (sy1 - sy0) * sx * sy
    return Crop(result, affine, (x0, y0, x1, y1), float(1 - observed_area / output_size ** 2))
