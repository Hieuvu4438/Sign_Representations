from __future__ import annotations

import numpy as np


def nearest_frame_plan(source_pts, start_s, end_s, frames=32, grid_fps=25.0, pts_tolerance_s=None,
                       allowed_start_s=None):
    pts = np.asarray(source_pts, dtype=np.float64)
    if pts.ndim != 1 or len(pts) == 0 or not np.isfinite(pts).all() or not np.all(np.diff(pts) > 0):
        raise ValueError("source PTS must be finite and strictly increasing")
    if not start_s < end_s or frames < 2 or grid_fps <= 0:
        raise ValueError("invalid sampling interval/grid")
    # Crop-grid origin differs from the source interval's allowed beginning.
    # A nearest frame slightly before a randomized crop origin is legal if it
    # remains inside the original manifest interval.
    allowed_start = start_s if allowed_start_s is None else allowed_start_s
    if allowed_start > start_s: raise ValueError("requested grid begins before allowed source interval")
    eligible = np.flatnonzero((pts >= allowed_start) & (pts < end_s))
    if not len(eligible):
        raise ValueError("no decoded frames in half-open interval")
    requested = start_s + np.arange(frames)/grid_fps
    candidates = pts[eligible]
    right = np.searchsorted(candidates, requested).clip(0, len(candidates)-1)
    left = (right-1).clip(0, len(candidates)-1)
    # Stable ties go to the earlier source frame.
    selected = np.where(abs(candidates[left]-requested) <= abs(candidates[right]-requested), left, right)
    indices = eligible[selected]
    valid = (requested < end_s) & (requested <= candidates[-1]+0.5/grid_fps)
    repeat = np.r_[False, indices[1:] == indices[:-1]] & valid
    # Nearest-source-frame quantization is bounded by the source frame spacing,
    # not half the requested grid spacing. A 24-fps camera on a 25-Hz grid can
    # legitimately be 20.83ms from the nearest frame. Lower-fps repeats remain
    # controlled separately by max_repeat_fraction_spectral.
    tolerance = float(pts_tolerance_s) if pts_tolerance_s is not None else .5*float(np.median(np.diff(pts))) if len(pts)>1 else .5/grid_fps
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive finite source PTS tolerance required")
    timing = abs(pts[indices]-requested) <= tolerance+1e-9
    repeat_fraction = float(repeat.sum()/max(valid.sum()-1, 1))
    return {"requested_timestamp_s": requested, "actual_timestamp_s": pts[indices], "frame_index": indices,
            "frame_valid": valid, "is_repeat": repeat, "timing_valid": timing & valid,
            "repeat_fraction": repeat_fraction}
