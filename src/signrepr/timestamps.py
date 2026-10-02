"""Decoded-frame presentation clock; validate rather than infer a constant frame rate."""
import json
import subprocess

import numpy as np


def frame_intervals(frames, expected_frames, fps):
    if len(frames) != expected_frames or not frames:
        raise ValueError('Source presentation clock count differs from decoded native streams')
    try:
        starts = np.asarray([float(frame['best_effort_timestamp_time']) for frame in frames], dtype=np.float64)
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError('Source frame presentation timestamp is unavailable') from error
    if not np.isfinite(starts).all() or (np.diff(starts) <= 0).any():
        raise ValueError('Source presentation clock is not finite and strictly increasing')
    last_duration = frames[-1].get('pkt_duration_time')
    try:
        last_duration = float(last_duration)
    except (TypeError, ValueError):
        last_duration = None
    if last_duration is not None and np.isfinite(last_duration) and last_duration > 0:
        tail_policy = 'reported_last_packet_duration'
    elif len(starts) > 1:
        last_duration = starts[-1] - starts[-2]
        tail_policy = 'last_observed_interframe_gap_for_final_end_only'
    else:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Single-frame source has no valid duration')
        last_duration = 1 / fps
        tail_policy = 'average_fps_for_single_frame_final_end_only'
    ends = np.concatenate([starts[1:], [starts[-1] + last_duration]])
    return np.stack([starts, ends], axis=1), tail_policy


def source_frame_intervals(path, expected_frames, fps):
    raw = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time,pkt_duration_time',
        '-of', 'json', str(path)], timeout=120)
    return frame_intervals(json.loads(raw)['frames'], expected_frames, fps)
