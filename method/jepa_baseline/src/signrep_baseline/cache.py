"""Validated, frame-indexed transport between the isolated teacher and RGB trainer."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .priors import build_priors


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_sha256():
    """Resume must pin implementation, not merely unchanged user configuration."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_standardized(path, expected_bones_sha256=None, expected_video_sha256=None):
    with np.load(path, allow_pickle=False) as file:
        data = {key: file[key].copy() for key in file.files}
    metadata = json.loads(str(data["metadata_json"]))
    if metadata.get("schema") != "signrep_standardized_v1":
        raise ValueError("Supervision requires corpus-standardized 3-D teacher cache")
    if expected_bones_sha256 and metadata["bone_statistics_sha256"] != expected_bones_sha256:
        raise ValueError("Pose cache uses a different bone table")
    if expected_video_sha256 and metadata["video_sha256"] != expected_video_sha256:
        raise ValueError("Pose cache belongs to a different video")
    frames = metadata["decoded_frames"]
    shapes = {"pose_3d": (frames, 61, 3), "confidence": (frames, 61),
              "body_angles": (frames, 22), "right_angles": (frames, 41),
              "left_angles": (frames, 41)}
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f"Invalid {key} cache shape")
    for part in ["body", "right", "left"]:
        flags = data[f"{part}_detected"]
        if flags.shape != (frames,) or flags.dtype != np.bool_:
            raise ValueError("Invalid teacher observation flags")
    if not np.array_equal(data["frame_indices"], np.arange(frames)):
        raise ValueError("Cache dropped/reordered decoded frames")
    return data, metadata


def clip_targets(data, indices, choices=None):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= len(data["pose_3d"])):
        raise ValueError("Clip indices outside teacher cache")
    return build_priors(torch.from_numpy(data["pose_3d"][indices]),
                        torch.from_numpy(data["confidence"][indices]),
                        {part: data[f"{part}_angles"][indices] for part in ("body", "right", "left")},
                        choices, angle_observed={part: data[f"{part}_detected"][indices]
                                                 for part in ("body", "right", "left")})
