"""Internal cache contract; detector confidence is never a depth coordinate."""
from pathlib import Path
import hashlib
import json
import numpy as np

PART_ORDER = ("body", "left_hand", "right_hand", "face")
PART_INDICES = {
    "body": [0, 3, 4, 5, 6, 7, 8, 9, 10],
    "left_hand": list(range(91, 112)),
    "right_hand": list(range(112, 133)),
    "face": list(range(23, 40, 2)) + list(range(83, 91)) + [53],
}
ROOT_LOCAL_INDEX = {"left_hand": 0, "right_hand": 0, "face": 17}
LABEL_FIELDS = {"gloss_id", "gloss_namespace", "gloss_sequence", "translation"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_pose(pose):
    if not isinstance(pose, dict):
        raise ValueError('Pose must be a dictionary of cache arrays')
    required = {"xy", "confidence", "observed", "timestamps_sec", "timing_known", "frame_indices", "track_continuity"}
    if required - pose.keys():
        raise ValueError(f"Missing cache fields: {sorted(required - pose.keys())}")
    for key in required-{'timing_known'}:
        if not isinstance(pose[key], np.ndarray):
            raise ValueError(f'Cache field {key} must be a NumPy array')
    xy, q, obs = pose["xy"], pose["confidence"], pose["observed"]
    if xy.ndim != 3 or xy.shape[1:] != (133, 2) or len(xy) == 0:
        raise ValueError("Expected nonempty [T,133,2] COCO-WholeBody xy")
    t = len(xy)
    if xy.dtype != np.float32 or q.dtype != np.float32:
        raise ValueError('xy/confidence cache arrays must be float32')
    if q.shape != (t, 133) or obs.shape != q.shape or obs.dtype != np.bool_:
        raise ValueError("Invalid confidence/observation shape or dtype")
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("Confidence must be finite in [0,1]")
    if not np.isfinite(xy[obs]).all():
        raise ValueError("Observed coordinates must be finite")
    times, indices, continuity = (pose[k] for k in ("timestamps_sec", "frame_indices", "track_continuity"))
    if times.dtype != np.float64 or indices.dtype != np.int64:
        raise ValueError('Cache timestamps must be float64 and frame indices int64')
    if times.shape != (t,) or indices.shape != (t,) or continuity.shape != (t - 1,):
        raise ValueError("Invalid timing/index/continuity shape")
    if indices.dtype.kind not in "iu" or np.any(np.diff(indices) <= 0) or continuity.dtype != np.bool_:
        raise ValueError("Source indices must increase; continuity must be boolean")
    if np.asarray(pose["timing_known"]).shape != () or np.asarray(pose['timing_known']).dtype != np.bool_:
        raise ValueError("timing_known must be a scalar")
    if bool(pose["timing_known"]) and (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Known timestamps must be finite and strictly increasing")
    return pose


def load_pose(path):
    with np.load(path, allow_pickle=False) as cache:
        return validate_pose({k: cache[k] for k in cache.files})


def save_pose(path, pose):
    validate_pose(pose)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pose)


def read_manifest(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        row["pose_path"] = str((path.parent / row["pose_path"]).resolve())
    return rows


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
