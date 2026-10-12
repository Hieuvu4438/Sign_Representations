"""Typed feature boundary shared by preprocessing, tokenizer and cache reader."""
from dataclasses import dataclass
import numpy as np

from .hashing import is_sha256
from ..geometry.supports import Primitive, normalized_joint_support
from ..pose.topology import POSE_GROUPS, TOPOLOGY_HASH

VISUAL_STREAMS = ("global", "left_hand", "right_hand", "face")
ARRAY_DTYPES = {"timestamps": "float64", "source_frame_index": "int64", "visual": "float16",
                "visual_valid": "bool", "pose_xy_body": "float32", "pose_observed": "bool",
                "pose_quality": "float32", "pose_coordinate_valid": "bool"}
IDENTITY_FIELDS = ("extractor_hash", "weights_sha256", "feature_dim", "visual_stream_order", "topology_hash",
                   "normalization", "temporal_dependency_policy", "detector_mode")


@dataclass(frozen=True)
class FeatureBundle:
    meta: dict
    timestamps: np.ndarray
    source_frame_index: np.ndarray
    visual: np.ndarray
    visual_valid: np.ndarray
    pose_xy_body: np.ndarray
    pose_observed: np.ndarray
    pose_quality: np.ndarray
    pose_coordinate_valid: np.ndarray
    raw_supports: tuple[frozenset[Primitive], ...]  # flatten time-major [T,7]

    def validate(self, *, allow_fixture=False):
        required = {"schema_version", "sample_id", "extractor_id", "extractor_hash", "weights_sha256", "feature_dim",
                    "visual_stream_order", "topology_hash", "normalization", "timestamp_unit", "source_sha256",
                    "quality_kind", "detector_mode", "temporal_dependency_policy"}
        if not required <= self.meta.keys():
            raise ValueError(f"Missing feature metadata: {sorted(required - self.meta.keys())}")
        if self.meta["schema_version"] != 1 or self.meta["visual_stream_order"] != list(VISUAL_STREAMS) or self.meta["topology_hash"] != TOPOLOGY_HASH:
            raise ValueError("Incompatible feature schema, stream order or topology")
        if self.meta["timestamp_unit"] != "seconds" or self.meta["normalization"] != "per_frame_shoulder_center_isotropic_scale":
            raise ValueError("Incompatible timestamp/coordinate definition")
        fixture = self.meta.get("synthetic_correctness_only", False)
        if fixture and not allow_fixture:
            raise ValueError("Synthetic features cannot be used in scientific runs")
        for key in ("source_sha256", "extractor_hash", "weights_sha256"):
            if not (fixture and allow_fixture and self.meta[key] is None) and not is_sha256(self.meta[key]):
                raise ValueError(f"Unresolved feature provenance: {key}")
        if self.timestamps.ndim != 1 or not len(self.timestamps):
            raise ValueError("FeatureBundle requires nonempty timestamps [T]")
        n, d = len(self.timestamps), self.meta["feature_dim"]
        if type(d) is not int or d <= 0:
            raise ValueError("Invalid feature dimension")
        shapes = {"timestamps": (n,), "source_frame_index": (n,), "visual": (n, 4, d), "visual_valid": (n, 4),
                  "pose_xy_body": (n, 49, 2), "pose_observed": (n, 49), "pose_quality": (n, 49), "pose_coordinate_valid": (n, 49)}
        for name, dtype in ARRAY_DTYPES.items():
            value = getattr(self, name)
            if value.dtype != np.dtype(dtype) or value.shape != shapes[name]:
                raise ValueError(f"Feature {name}: expected {dtype}{shapes[name]}, got {value.dtype}{value.shape}")
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"Feature {name} contains nonfinite values")
        if (np.diff(self.timestamps) < 0).any() or (self.source_frame_index < 0).any():
            raise ValueError("Invalid timestamps or raw frame IDs")
        valid = self.pose_coordinate_valid
        if (valid & ~self.pose_observed).any() or (self.pose_quality < 0).any() or (self.pose_quality > 1).any():
            raise ValueError("Invalid pose validity/quality semantics")
        if (self.pose_xy_body[~valid] != 0).any() or (self.pose_quality[~valid] != 0).any():
            raise ValueError("Invalid model coordinates/quality must be zero with validity false")
        if (valid.any(axis=1) & ~valid[:, 1:3].all(axis=1)).any():
            raise ValueError("Body coordinates depend on valid shoulders")
        if len(self.raw_supports) != n * 7 or any(not support for support in self.raw_supports):
            raise ValueError("Expected nonempty supports for all [T,7] tokens")
        for t, frame in enumerate(self.source_frame_index):
            for stream in range(7):
                support = self.raw_supports[t * 7 + stream]
                if not isinstance(support, frozenset) or any(not isinstance(p, Primitive) for p in support):
                    raise ValueError("Raw supports must contain typed primitive sets")
                required = ({Primitive("rgb", int(frame))} if stream < 4
                            else normalized_joint_support(frame, POSE_GROUPS[stream - 4]))
                if not required <= support:
                    raise ValueError("Support omits RGB or pose normalization prerequisites")
        # Same-frame cache must never hide an undeclared tracking dependency.
        if self.meta["temporal_dependency_policy"] == "same_frame":
            if self.meta["detector_mode"] != "IMAGE":
                raise ValueError("Same-frame features require IMAGE detector mode")
            for index, support in enumerate(self.raw_supports):
                if any(p.frame != self.source_frame_index[index // 7] for p in support):
                    raise ValueError("Same-frame support references a different raw frame")
        # An alias must represent the same observation in every cached array.
        first = {}
        for index, source in enumerate(self.source_frame_index):
            if source in first:
                other = first[source]
                if any(not np.array_equal(getattr(self, name)[index], getattr(self, name)[other])
                       for name in ARRAY_DTYPES):
                    raise ValueError("Resampled aliases have inconsistent cached observations")
                if self.raw_supports[index * 7:(index + 1) * 7] != self.raw_supports[other * 7:(other + 1) * 7]:
                    raise ValueError("Resampled aliases have inconsistent supports")
            else:
                first[source] = index
        return self
