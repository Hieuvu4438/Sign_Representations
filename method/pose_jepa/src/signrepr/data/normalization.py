"""Fixed full-sample shoulder reference, before windows or artificial masking."""
import numpy as np
import torch
from .schema import PART_ORDER, PART_INDICES, ROOT_LOCAL_INDEX, validate_pose


def canonicalize_sample(pose, metadata=None, confidence_threshold=0.3,
                        min_part_valid_fraction=0.5, coordinate_profile="viewport_normalized", reference=None):
    validate_pose(pose)
    metadata = metadata or {}
    if not isinstance(metadata, dict):
        raise ValueError('Metadata must be a dictionary')
    if metadata.get("joint_layout", "coco_wholebody_133") != "coco_wholebody_133":
        raise ValueError("Unknown joint convention")
    if metadata.get("coordinate_system", "viewport_normalized") != "viewport_normalized":
        raise ValueError("Adapter must convert pixel coordinates to viewport first")
    xy = np.array(pose["xy"], dtype=np.float32, copy=True)
    if coordinate_profile == "aspect_corrected":
        w, h = metadata.get("image_width"), metadata.get("image_height")
        if not w or not h or min(w, h) <= 0:
            raise ValueError("Aspect-corrected profile requires image dimensions")
        xy[..., 0] *= w / h
    elif coordinate_profile != "viewport_normalized":
        raise ValueError("Unknown coordinate profile")
    q = pose["confidence"].astype(np.float32)
    valid = pose["observed"] & np.isfinite(xy).all(-1) & (q >= confidence_threshold)
    if reference is None:
        shoulder_valid = valid[:, 5] & valid[:, 6]
        pairs = xy[shoulder_valid][:, [5, 6]]
        ok = len(pairs) >= min(3, len(xy))
        origin = np.median(pairs.mean(1), axis=0) if len(pairs) else np.zeros(2, np.float32)
        scale = float(np.median(np.linalg.norm(pairs[:, 0] - pairs[:, 1], axis=-1))) if len(pairs) else 0.0
        ok = ok and np.isfinite(scale) and scale > 1e-4
        reference = {"origin": origin.tolist(), "scale": scale, "valid": bool(ok), "coordinate_profile": coordinate_profile}
    if reference["coordinate_profile"] != coordinate_profile:
        raise ValueError("Reference coordinate profile mismatch")
    ok = reference["valid"]
    g = (np.where(valid[..., None], xy, 0) - np.asarray(reference["origin"], np.float32)) / (reference["scale"] if ok else 1.0)
    g = np.where((valid & ok)[..., None], g, 0).astype(np.float32)
    valid &= ok
    parts = {}
    for part in PART_ORDER:
        ids = PART_INDICES[part]
        v, score, glob = valid[:, ids].copy(), q[:, ids].copy(), g[:, ids].copy()
        global_valid = v.copy()
        anchor = np.zeros((len(xy), 3), np.float32)
        local = glob.copy()
        if part in ROOT_LOCAL_INDEX:
            root = ROOT_LOCAL_INDEX[part]
            v &= v[:, root:root + 1]
            local = glob - glob[:, root:root + 1]
            anchor = np.concatenate([glob[:, root], score[:, root:root + 1]], -1)
        fv = v.sum(-1) >= int(np.ceil(len(ids) * min_part_valid_fraction))
        quality = (score * v).mean(-1) * fv
        parts[part] = {k: torch.from_numpy(a.copy()) for k, a in {
            "xy": np.where(v[..., None], local, 0), "global_xy": glob,
            "confidence": score, "joint_valid": v, "global_joint_valid": global_valid, "frame_valid": fv,
            "quality": quality, "anchor": np.where(fv[:, None], anchor, 0),
        }.items()}
    known = bool(pose["timing_known"])
    times = pose["timestamps_sec"].copy() if known else np.zeros(len(xy), np.float64)
    position = 25 * (times - times[0]) if known else pose["frame_indices"] - pose["frame_indices"][0]
    return {"parts": parts, "timestamps_sec": torch.from_numpy(times), "position": torch.from_numpy(position.astype(np.float32)),
            "frame_indices": torch.from_numpy(pose["frame_indices"].astype(np.int64)),
            "track_continuity": torch.from_numpy(pose["track_continuity"]), "timing_known": known,
            "length": len(xy), "reference": reference,
            "status": "ok" if ok else "normalization_failed", "sample_id": metadata.get("sample_id", "unknown")}


def select_frames(sample, indices):
    """Subsample with continuity of every intermediate edge, preserving position origin."""
    indices = torch.as_tensor(indices, dtype=torch.long)
    out = {**sample, "length": len(indices)}
    out["parts"] = {p: {k: v[indices] for k, v in stream.items()} for p, stream in sample["parts"].items()}
    for key in ("timestamps_sec", "position", "frame_indices"):
        out[key] = sample[key][indices]
    out["track_continuity"] = torch.tensor([bool(sample["track_continuity"][a:b].all()) for a, b in zip(indices[:-1], indices[1:])], dtype=torch.bool)
    return out
