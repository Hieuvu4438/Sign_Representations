"""Owned synthetic provider for correctness smoke, never a scientific data fallback."""
import numpy as np
from ..cache.hashing import canonical_hash
from ..cache.schema import FeatureBundle
from ..geometry.supports import Primitive, normalized_joint_support
from ..pose.coordinates import body_coordinates
from ..pose.topology import POSE_GROUPS, TOPOLOGY_HASH


def synthetic_features(sample_id, length=48, visual_dim=384, seed=17, phase=0.0):
    rng = np.random.default_rng(seed)
    times = np.arange(length, dtype=np.float64) / 25
    xy = np.broadcast_to(rng.uniform(.3, .7, (1, 49, 2)), (length, 49, 2)).copy()
    xy[:, 1], xy[:, 2] = [.4, .45], [.6, .45]
    motion = .03 * np.sin(2 * np.pi * times + phase)
    xy[:, 7:, 0] += motion[:, None]
    observed = np.ones((length, 49), dtype=bool)
    # A short missing segment exercises validity, not fake stationary targets.
    observed[6:9, 7:28] = False
    xy[~observed] = np.nan
    quality = observed.astype(np.float32)
    coords = body_coordinates(xy, observed, quality, np.tile([640, 480], (length, 1)))
    frequencies = np.arange(visual_dim, dtype=np.float64) % 9 + 1
    pattern = np.sin(times[:, None] * frequencies[None] + phase)
    visual = np.stack([pattern + stream * .05 for stream in range(4)], axis=1)
    visual += rng.normal(0, .01, visual.shape)
    visual_valid = np.ones((length, 4), dtype=bool)
    visual_valid[6:9, 1] = False
    visual[~visual_valid] = 0
    supports = []
    for frame in range(length):
        supports.extend(frozenset({Primitive("rgb", frame)}) for _ in range(4))
        supports.extend(normalized_joint_support(frame, joints) for joints in POSE_GROUPS)
    meta = {"schema_version": 1, "sample_id": sample_id, "extractor_id": "synthetic_fixture",
        "extractor_hash": canonical_hash({"fixture_version": 1, "visual_dim": visual_dim}), "weights_sha256": None,
        "source_sha256": None, "feature_dim": visual_dim, "visual_stream_order": ["global", "left_hand", "right_hand", "face"],
        "topology_hash": TOPOLOGY_HASH, "normalization": "per_frame_shoulder_center_isotropic_scale",
        "timestamp_unit": "seconds", "quality_kind": "synthetic_observed_binary", "detector_mode": "IMAGE",
        "temporal_dependency_policy": "same_frame", "synthetic_correctness_only": True}
    return FeatureBundle(meta, times, np.arange(length, dtype=np.int64), visual.astype(np.float16), visual_valid,
        coords.xy, observed, coords.quality, coords.coordinate_valid, tuple(supports)).validate(allow_fixture=True)
