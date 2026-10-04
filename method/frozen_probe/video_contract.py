"""Decode/timeline validation and transparent perceptual-screen fingerprints."""
import math
from temporal import window_plan


def summarize_timeline(pts, decoded_count, length=16, stride=2):
    if len(pts) != decoded_count:
        raise ValueError('OpenCV/ffprobe decoded frame count mismatch')
    plan = window_plan(pts, length, stride)
    return {'frames': decoded_count, 'windows': len(plan),
            'uncovered_tail_frames': decoded_count-plan[-1]['stop_frame_exclusive'],
            'first_pts_s': pts[0], 'last_pts_s': pts[-1],
            'last_window_end_s': plan[-1]['end_s_exclusive'],
            'end_boundary_estimated': plan[-1]['end_boundary_estimated']}


def fingerprint_indices(frame_count, quantiles):
    if frame_count < 1 or not quantiles or any(not math.isfinite(q) or q < 0 or q > 1 for q in quantiles):
        raise ValueError('Invalid fingerprint sampling contract')
    return [math.floor((frame_count-1)*q) for q in quantiles]


def dhash64(gray):
    """9×8 uint8 thumbnail; row-major bit1 if right pixel is brighter."""
    if tuple(gray.shape) != (8,9):
        raise ValueError('Expected grayscale9×8 thumbnail')
    result = 0
    for bit in (gray[:,1:] > gray[:,:-1]).flatten():
        result = (result << 1) | int(bit)
    return f'{result:016x}'
