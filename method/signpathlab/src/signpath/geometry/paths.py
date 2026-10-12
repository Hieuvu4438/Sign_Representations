from dataclasses import dataclass
import numpy as np
from .anchors import signature_descriptor
from .motion_controls import ordinary_motion_descriptor
from .supports import Primitive, normalized_joint_support


@dataclass(frozen=True)
class PathGroup:
    name: str
    joints: tuple[int, ...]
    edge: bool = False

    @property
    def dimension(self):
        return 2 if self.edge else 2 * len(self.joints)

    def values(self, coordinates):
        if self.edge:
            return coordinates[:, self.joints[1]] - coordinates[:, self.joints[0]]
        return coordinates[:, self.joints].reshape(len(coordinates), -1)


DEFAULT_PATHS = (
    PathGroup("left_wrist", (7,)), PathGroup("right_wrist", (28,)),
    PathGroup("left_index_tip", (15,)), PathGroup("right_index_tip", (36,)),
    PathGroup("left_wrist_to_index", (7, 15), True), PathGroup("right_wrist_to_index", (28, 36), True),
    PathGroup("left_finger_chain", (11, 15, 27)), PathGroup("right_finger_chain", (32, 36, 48)),
    PathGroup("left_arm_chain", (1, 3, 5)), PathGroup("right_arm_chain", (2, 4, 6)),
    PathGroup("bimanual_wrist_chain", (7, 28)))


@dataclass(frozen=True)
class GeometryToken:
    id: str
    group: str
    scale: int
    center_index: int
    timestamp: float
    bounds: tuple[float, float]
    actual_span: float
    point_count: int
    distinct_raw_frames: int
    support: frozenset[Primitive]
    descriptor: np.ndarray | None
    reason: str | None

    @property
    def valid(self):
        return self.descriptor is not None


def build_geometry(sample_id, timestamps, source_frame_index, coordinates, coordinate_valid, quality,
                   *, window_seconds=(0.16, 0.32, 0.64), center_stride_samples=4,
                   descriptor="logsig2_anchor_timing", min_quality=0.5, min_distinct=2, extra_supports=None):
    times, sources = np.asarray(timestamps), np.asarray(source_frame_index)
    xy, valid, quality = np.asarray(coordinates), np.asarray(coordinate_valid), np.asarray(quality)
    n = len(times)
    if times.shape != (n,) or sources.shape != (n,) or xy.shape != (n, 49, 2) or valid.shape != (n, 49) or quality.shape != (n, 49) or n < 1:
        raise ValueError("Malformed geometry inputs")
    if not np.isfinite(times).all() or (np.diff(times) < 0).any() or center_stride_samples <= 0 or min_distinct < 2:
        raise ValueError("Invalid geometry schedule")
    if descriptor not in {"logsig2_anchor_timing", "ordinary_motion_anchor_timing"}:
        raise ValueError(f"Unsupported descriptor: {descriptor}")
    if extra_supports is not None and len(extra_supports) != n:
        raise ValueError("Expected extra preprocessing supports for each frame")
    describe = signature_descriptor if descriptor == "logsig2_anchor_timing" else ordinary_motion_descriptor
    tokens = []
    for center in range(0, n, center_stride_samples):
        for scale, duration in enumerate(window_seconds):
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError("Invalid geometry duration")
            lo, hi = float(times[center] - duration / 2), float(times[center] + duration / 2)
            # Allow rounding noise at exact observed boundaries, not partial windows.
            boundary = lo < times[0] - 1e-9 or hi > times[-1] + 1e-9
            selected = np.flatnonzero((times >= lo - 1e-12) & (times < hi - 1e-12))
            _, first = np.unique(sources[selected], return_index=True)
            unique = selected[np.sort(first)]
            span = float(times[unique[-1]] - times[unique[0]]) if len(unique) else 0.0
            for group in DEFAULT_PATHS:
                support = frozenset(p for idx in selected for p in normalized_joint_support(sources[idx], group.joints))
                if extra_supports is not None:
                    support |= frozenset(p for idx in selected for p in extra_supports[idx])
                reason = None
                if boundary:
                    reason = "incomplete_window"
                elif len(unique) < min_distinct:
                    reason = "too_few_distinct_observations"
                elif (np.diff(times[unique]) <= 0).any():
                    reason = "nonpositive_dt"
                elif not valid[np.ix_(unique, group.joints)].all() or not (quality[np.ix_(unique, group.joints)] >= min_quality).all():
                    reason = "missing_or_low_quality"
                points = group.values(xy[unique])
                if reason is None and not np.isfinite(points).all():
                    reason = "nonfinite_coordinate"
                values = describe(points, times[unique]).astype(np.float32) if reason is None else None
                tokens.append(GeometryToken(f"{sample_id}:{center}:{scale}:{group.name}", group.name, scale, center,
                    float(times[center]), (lo, hi), span, len(selected), len(unique), support, values, reason))
    return tuple(tokens)


def geometry_from_features(bundle, **kwargs):
    """Preserve declared extraction/matching prerequisites in derived geometry."""
    extra = [frozenset(p for support in bundle.raw_supports[t * 7 + 4:t * 7 + 7] for p in support)
             for t in range(len(bundle.timestamps))]
    return build_geometry(bundle.meta["sample_id"], bundle.timestamps, bundle.source_frame_index,
                          bundle.pose_xy_body, bundle.pose_coordinate_valid, bundle.pose_quality,
                          extra_supports=extra, **kwargs)
