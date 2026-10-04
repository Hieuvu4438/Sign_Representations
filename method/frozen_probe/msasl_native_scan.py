"""Streaming native-frame eligibility, with bounded independent interval PTS audit.

Existing msasl_native remains immutable. This version retains no video arrays;
it hashes every resized frame and native PTS sequence for later extraction joins.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess

import av
import cv2
import numpy as np

from msasl_native import crop_bounds


class EligibilityIssue(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def scan_interval(path, row, max_frames=1024, max_decoded=4096):
    left, right = Fraction(str(row['start_time'])), Fraction(str(row['end_time']))
    if not 0 <= left < right or right-left > 8:
        raise EligibilityIssue('interval_contract')
    ticks, bounds, previous, decoded = [], None, None, 0
    resized_sha = hashlib.sha256()
    first_window_sha = hashlib.sha256()
    with av.open(path if hasattr(path, 'read') else str(path)) as container:
        if len(container.streams.video) != 1:
            raise EligibilityIssue('video_stream_count')
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        if stream.start_time is None or stream.time_base is None:
            raise EligibilityIssue('unknown_stream_origin')
        tb = stream.time_base
        origin = stream.start_time*tb
        meta = dict(time_base=str(tb), origin_pts=int(stream.start_time),
                    origin_seconds=float(origin), stream_width=stream.width,
                    stream_height=stream.height, average_rate=str(stream.average_rate),
                    reported_stream_frames=stream.frames,
                    stream_duration_seconds=None if stream.duration is None else float(stream.duration*tb))
        container.seek(math.floor((origin+left)/tb), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(video=0):
            decoded += 1
            if decoded > max_decoded:
                raise EligibilityIssue('resource_seek_frame_bound')
            if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                raise EligibilityIssue('invalid_native_pts_or_corrupt_frame')
            if frame.time_base != tb:
                raise EligibilityIssue('dynamic_frame_time_base')
            t = frame.pts*tb-origin
            if previous is not None and t <= previous:
                raise EligibilityIssue('native_pts_not_increasing')
            previous = t
            if t < left:
                continue
            if t >= right:
                break
            if len(ticks) >= max_frames:
                raise EligibilityIssue('resource_selected_frame_bound')
            if frame.width > 3840 or frame.height > 2160:
                raise EligibilityIssue('resource_decoded_resolution_bound')
            current = crop_bounds(row['box'], frame.width, frame.height)
            if bounds is not None and current != bounds:
                raise EligibilityIssue('dynamic_decoded_crop_shape')
            bounds = current
            y0, x0, y1, x1 = bounds
            rgb = frame.to_ndarray(format='rgb24')
            resized = cv2.resize(rgb[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_LINEAR)
            data = resized.tobytes()
            resized_sha.update(data)
            if len(ticks) < 16:
                first_window_sha.update(data)
            ticks.append(int(frame.pts))
    n = len(ticks)
    return dict(meta, crop_yxyx_pixels=bounds, decoded_from_keyframe_frames=decoded,
                selected_native_frames=n, native16_eligible=n >= 16,
                complete_stride2_windows=0 if n < 16 else (n-16)//2+1,
                uncovered_tail_frames=n if n < 16 else (n-16) % 2,
                first_selected_relative_pts=None if not ticks else float(ticks[0]*tb-origin),
                last_selected_relative_pts=None if not ticks else float(ticks[-1]*tb-origin),
                native_pts_ticks_sha256=hashlib.sha256(np.asarray(ticks, dtype='<i8').tobytes()).hexdigest(),
                all_resized_RGB_sha256=resized_sha.hexdigest(),
                first_native16_RGB_sha256=None if n < 16 else first_window_sha.hexdigest()), ticks


def ffprobe_interval(path, row, scan, ticks, timeout_seconds=30, max_bytes=1 << 20):
    """Absolute interval end +1s flush margin, then exact rational selection.

ffprobe seeks to a keyframe; its output must be filtered, never assumed to start
at the request. PTS equality uses integer ticks, not rounded decimal timestamps.
"""
    tb = Fraction(scan['time_base'])
    origin = scan['origin_pts']*tb
    left, right = Fraction(str(row['start_time'])), Fraction(str(row['end_time']))
    interval = f'{float(origin+left):.9f}%{float(origin+right+1):.9f}'
    cmd = ['ffprobe', '-v', 'error', '-threads', '1', '-select_streams', 'v:0',
           '-read_intervals', interval, '-show_entries',
           'stream=time_base,start_pts,width,height:frame=pts', '-of', 'json', str(path)]
    # A per-child timeout is a hard process timeout. PIPE responses are checked
    # after capture; this is not an OS memory/output quota (see protocol).
    p = subprocess.run(cmd, check=True, capture_output=True, timeout=timeout_seconds)
    if len(p.stdout) > max_bytes or len(p.stderr) > max_bytes:
        raise EligibilityIssue('resource_reference_response_bound')
    data = json.loads(p.stdout)
    streams = data.get('streams', [])
    if len(streams) != 1:
        raise EligibilityIssue('reference_stream_count')
    s = streams[0]
    if Fraction(s['time_base']) != tb or int(s['start_pts']) != scan['origin_pts']:
        raise EligibilityIssue('reference_origin_or_timebase_mismatch')
    all_ticks = [int(x['pts']) for x in data.get('frames', [])]
    if any(b <= a for a, b in zip(all_ticks, all_ticks[1:])):
        raise EligibilityIssue('reference_pts_not_increasing')
    selected = [t for t in all_ticks if left <= t*tb-origin < right]
    if selected != ticks:
        raise EligibilityIssue('independent_interval_pts_mismatch')
    return dict(command=cmd, response_bytes=len(p.stdout), stderr_bytes=len(p.stderr),
                reported_frames=len(all_ticks), selected_native_frames=len(selected),
                integer_pts_exact=True, reference_dimensions=[s['width'], s['height']])


def file_digest(path, byte_limit=512 << 20):
    path = Path(path)
    before = path.stat()
    if before.st_size > byte_limit:
        raise EligibilityIssue('resource_source_bytes_bound')
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise EligibilityIssue('source_changed_during_hash')
    return dict(sha256=digest.hexdigest(), bytes=before.st_size, mtime_ns=before.st_mtime_ns)
