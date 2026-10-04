"""Native-frame window contract; no resampling or invented padding."""
import math
import statistics


def window_plan(pts, length=16, stride=2):
    if type(length) is not int or type(stride) is not int or length < 1 or stride < 1:
        raise ValueError('Window length/stride must be positive integers')
    if len(pts) < length or len(pts) < 2:
        raise ValueError('Clip too short for complete native-frame windows')
    if not all(math.isfinite(t) for t in pts) or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError('PTS must be finite and strictly increasing')
    last_duration = statistics.median(b-a for a, b in zip(pts, pts[1:]))
    windows = []
    for start in range(0, len(pts)-length+1, stride):
        stop = start + length
        windows.append({'start_frame': start, 'stop_frame_exclusive': stop,
                        'start_s': pts[start],
                        'end_s_exclusive': pts[stop] if stop < len(pts) else pts[-1]+last_duration,
                        'end_boundary_estimated': stop == len(pts),
                        'center_s': (pts[start]+pts[stop-1])/2,
                        'valid_mask': [True]*length})
    return windows
