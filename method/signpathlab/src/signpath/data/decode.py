"""Decode observed PTS and resample by nearest observation, never interpolate RGB."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class DecodedVideo:
    frames: np.ndarray
    timestamps: np.ndarray
    source_frame_index: np.ndarray
    timestamp_fallback: bool
    nominal_fps: float | None


@dataclass(frozen=True)
class FrameSelection:
    indices: np.ndarray
    requested_timestamps: np.ndarray
    actual_timestamps: np.ndarray
    source_frame_index: np.ndarray
    duplicate_selection: np.ndarray


def decode_video(path: str | Path, start_s=0.0, end_s=None, *, allow_fps_fallback=False) -> DecodedVideo:
    import av
    frames, times, sources = [], [], []
    fallback = False
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream: {path}")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or not frame.time_base:
                if not allow_fps_fallback or fps is None or fps <= 0:
                    raise ValueError(f"Frame {index} lacks PTS/time_base; explicit FPS fallback required")
                timestamp = index / fps
                fallback = True
            else:
                timestamp = float(frame.pts * frame.time_base)
            if not np.isfinite(timestamp):
                raise ValueError("Nonfinite video timestamp")
            if timestamp >= start_s and (end_s is None or timestamp < end_s):
                frames.append(frame.to_ndarray(format="rgb24"))
                times.append(timestamp)
                sources.append(index)
    if not frames:
        raise ValueError(f"No frames in requested interval: {path}")
    order = np.argsort(times, kind="stable")
    return DecodedVideo(np.stack(frames)[order], np.asarray(times, dtype=np.float64)[order],
                        np.asarray(sources, dtype=np.int64)[order], fallback, fps)


def nearest_observations(timestamps, source_frame_index, target_fps) -> FrameSelection:
    times = np.asarray(timestamps, dtype=np.float64)
    sources = np.asarray(source_frame_index)
    if times.ndim != 1 or not len(times) or sources.shape != times.shape or sources.dtype != np.int64:
        raise ValueError("Expected nonempty timestamps and int64 source indices [T]")
    if not np.isfinite(times).all() or (np.diff(times) < 0).any() or not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("Invalid timestamps or target FPS")
    grid = times[0] + np.arange(int(np.floor((times[-1] - times[0]) * target_fps + 1e-9)) + 1) / target_fps
    right = np.searchsorted(times, grid, side="left").clip(0, len(times) - 1)
    left = (right - 1).clip(0, len(times) - 1)
    indices = np.where(np.abs(grid - times[left]) <= np.abs(times[right] - grid), left, right)
    selected_sources = sources[indices]
    _, inverse, counts = np.unique(selected_sources, return_inverse=True, return_counts=True)
    return FrameSelection(indices.astype(np.int64), grid, times[indices], selected_sources, counts[inverse] > 1)
