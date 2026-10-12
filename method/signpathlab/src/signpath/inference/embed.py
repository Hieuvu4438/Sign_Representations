"""Actual raw-video extraction to timestamped unmasked embeddings and optional logits."""
from dataclasses import replace
from pathlib import Path
import os
import tempfile
import time
import numpy as np
import torch

from ..backbones.dino import DINOAdapter
from ..cache.hashing import file_hash
from ..cache.schema import IDENTITY_FIELDS
from ..cli import write_json
from ..data.decode import decode_video, nearest_observations
from ..pose.context import ExtractionContext
from ..pose.extract import LandmarkExtractor
from ..preprocessing import asset_plan
from .deployment import graph_inputs, validate_inputs
from .extract import window_starts
from .run import load_run
from ..training.runner import full_clip_batch


def extract_video_features(run, video_path, *, assets_file=None, device="cpu"):
    plan = asset_plan(run.resolved, assets_file, include_dino=True)
    video = decode_video(video_path)
    with LandmarkExtractor(*(plan["detectors"][key]["path"] for key in ("pose", "hand", "face")),
                           min_body_visibility=run.resolved.values["pose"]["min_body_visibility"]) as extractor:
        raw = extractor.extract(video.frames)
    adapter = DINOAdapter().load(plan["dino"], device)
    context = ExtractionContext(Path(video_path).stem, file_hash(video_path), video.source_frame_index, raw,
        {"source_mirrored": None, "hand_assignment": raw.provenance["hand_assignment"]},
        min_body_visibility=run.resolved.values["pose"]["min_body_visibility"])
    bundle = adapter.extract(video.frames, video.timestamps, context)
    actual = {key: bundle.meta[key] for key in IDENTITY_FIELDS}
    if actual != run.input_identity:
        raise ValueError("Raw inference extractor differs from the encoder's registered feature identity")
    selection = nearest_observations(video.timestamps, video.source_frame_index, run.resolved.values["data"]["target_fps"])
    names = ("timestamps", "source_frame_index", "visual", "visual_valid", "pose_xy_body", "pose_observed", "pose_quality", "pose_coordinate_valid")
    return replace(bundle, **{name: getattr(bundle, name)[selection.indices] for name in names},
        raw_supports=tuple(s for t in selection.indices for s in bundle.raw_supports[t * 7:(t + 1) * 7]))


@torch.no_grad()
def embed_features(run, bundle, *, device="cpu"):
    from ..data.sampling import slice_features
    graph = run.graph().to(device).eval()
    ds = run.resolved.values["downstream"]
    n = len(bundle.timestamps)
    dimension = run.model_config["model"]["embed_dim"] if run.model is not None else bundle.meta["feature_dim"]
    sums, counts = torch.zeros(n, dimension, device=device), torch.zeros(n, 1, device=device)
    real = torch.zeros(n, dtype=torch.bool, device=device)
    for start in window_starts(n, ds["window_samples"], ds["overlap_samples"]):
        stop = min(n, start + ds["window_samples"])
        batch = full_clip_batch([slice_features(bundle, start, stop)], run.model_config).to(device)
        inputs = graph_inputs(batch)
        validate_inputs(inputs, run.model_config["model"]["raw_dimensions"])
        dense, _, valid, _, _ = graph(*inputs)
        sums[start:stop] += dense[0] * valid[0, :, None]
        counts[start:stop] += valid[0, :, None]
        real[start:stop] |= valid[0]
    dense = sums / counts.clamp_min(1)
    first = np.unique(bundle.source_frame_index, return_index=True)[1]
    unique = torch.zeros(n, dtype=torch.bool, device=device)
    unique[first] = True
    selected = unique & real
    if not selected.any():
        raise ValueError("Raw video has no usable model observations")
    clip = dense[selected].mean(dim=0)
    logits = None
    if run.head is not None:
        values = torch.nn.functional.normalize(dense, dim=-1) if ds["l2_normalize"] else dense
        if ds["head"] == "attentive":
            logits = run.head(values[None], real[None])[0]
        else:
            x = torch.nn.functional.normalize(clip, dim=-1) if ds["l2_normalize"] else clip
            logits = run.head(x[None])[0]
    return dense, real, clip, logits


def embed_video(run_dir, input_path, output_path, *, assets_file=None, device="cpu"):
    output = Path(output_path)
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise FileExistsError("Embedding artifact already exists; use a new output path")
    torch.set_num_threads(1)
    started = time.monotonic()
    run = load_run(run_dir, device)
    bundle = extract_video_features(run, input_path, assets_file=assets_file, device=device)
    dense, valid, clip, logits = embed_features(run, bundle, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"dense": dense.cpu().numpy(), "timestamps": bundle.timestamps, "valid": valid.cpu().numpy(),
        "clip": clip.cpu().numpy(), "source_frame_index": bundle.source_frame_index, "model_hash": np.array(run.model_hash),
        "preprocessing_hash": np.array(bundle.meta["preprocessing_hash"]), "schema_version": np.array(1),
        "class_ids": np.asarray(run.class_ids, dtype=np.int64)}
    if logits is not None:
        arrays["logits"] = logits.cpu().numpy()
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez(stream, **arrays)
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {"status": "completed", "schema_version": 1, "scope": "offline_multimodal_full_clip_inference",
        "source_sha256": file_hash(input_path), "model_hash": run.model_hash, "preprocessing_hash": bundle.meta["preprocessing_hash"],
        "feature_identity": run.input_identity, "dense_shape": list(dense.shape), "valid_frames": int(valid.sum()),
        "visual_valid_counts": bundle.visual_valid.sum(axis=0).tolist(), "class_ids": list(run.class_ids),
        "predicted_label_id": int(run.class_ids[int(logits.argmax())]) if logits is not None else None,
        "timestamps": "original selected PTS seconds", "asset_cost_included": "decode/IMAGE pose-hand-face/DINO/temporal/head",
        "runtime_seconds": time.monotonic() - started, "output": str(output), "artifact_sha256": file_hash(output)}
    write_json(output.with_suffix(output.suffix + ".json"), report)
    return report
