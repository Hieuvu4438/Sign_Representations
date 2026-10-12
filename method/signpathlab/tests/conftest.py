from dataclasses import replace
import numpy as np
import pytest

from signpath.cache.hashing import canonical_hash
from signpath.cache.schema import FeatureBundle
from signpath.geometry.supports import Primitive, normalized_joint_support
from signpath.pose.coordinates import body_coordinates
from signpath.pose.topology import POSE_GROUPS, TOPOLOGY_HASH


def synthetic_bundle(n=48, d=8):
    rng = np.random.default_rng(11)
    raw = rng.uniform(0.2, 0.8, (n, 49, 2))
    raw[:, 1] = [0.4, 0.4]
    raw[:, 2] = [0.6, 0.4]
    observed = np.ones((n, 49), dtype=bool)
    normalized = body_coordinates(raw, observed, np.ones((n, 49)), np.tile([640, 480], (n, 1)))
    supports = []
    for frame in range(n):
        for stream, joints in enumerate((tuple(range(49)), tuple(range(7, 28)), tuple(range(28, 49)), ())):
            supports.append(frozenset({Primitive("rgb", frame)}) | frozenset(Primitive("joint", frame, j) for j in joints))
        supports.extend(normalized_joint_support(frame, joints) for joints in POSE_GROUPS)
    meta = {"schema_version": 1, "sample_id": "fixture", "extractor_id": "fixture",
            "extractor_hash": canonical_hash({"fixture": 1}), "weights_sha256": None,
            "source_sha256": None, "feature_dim": d, "visual_stream_order": ["global", "left_hand", "right_hand", "face"],
            "topology_hash": TOPOLOGY_HASH, "detector_mode": "IMAGE", "temporal_dependency_policy": "same_frame",
            "normalization": "per_frame_shoulder_center_isotropic_scale", "timestamp_unit": "seconds",
            "quality_kind": "synthetic_binary", "synthetic_correctness_only": True}
    return FeatureBundle(meta, np.arange(n, dtype=np.float64) / 25, np.arange(n, dtype=np.int64),
        rng.normal(size=(n, 4, d)).astype(np.float16), np.ones((n, 4), dtype=bool), normalized.xy,
        observed, normalized.quality, normalized.coordinate_valid, tuple(supports))


@pytest.fixture
def bundle():
    return synthetic_bundle()
