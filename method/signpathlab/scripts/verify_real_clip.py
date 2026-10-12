"""Opt-in full-clip preprocessing gate; does not fit a model or report utility scores."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml

from signpath.backbones.dino import DINOAdapter
from signpath.cache.hashing import file_hash
from signpath.cache.reader import read_bundle
from signpath.cache.schema import IDENTITY_FIELDS
from signpath.cache.writer import write_bundle
from signpath.cli import write_json
from signpath.data.decode import decode_video, nearest_observations
from signpath.geometry.paths import geometry_from_features
from signpath.masking.closure import ClosureResolver
from signpath.masking.samplers import MaskSampler
from signpath.models.tokenizer import SignTokenizer
from signpath.pose.context import ExtractionContext
from signpath.pose.extract import LandmarkExtractor


def main():
    parser = argparse.ArgumentParser()
    for name in ("video", "dino-repo", "dino-weights", "pose-model", "hand-model", "face-model", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite real-clip evidence: {output}")
    output.mkdir(parents=True)
    start = time.monotonic()
    decoded = decode_video(args.video)
    selected = nearest_observations(decoded.timestamps, decoded.source_frame_index, 25)
    # Infer original frames once and then select aliases; avoids stochastic duplicate observations.
    with LandmarkExtractor(args.pose_model, args.hand_model, args.face_model) as extractor:
        raw = extractor.extract(decoded.frames)
        # IMAGE mode should give identical output when an intervening frame changes.
        probe = decoded.frames[[0, min(1, len(decoded.frames) - 1), 0]].copy()
        probe[1] = 0
        repeats = extractor.extract(probe)
        np.testing.assert_allclose(repeats.xy[0], repeats.xy[2], equal_nan=True, atol=1e-6)
        np.testing.assert_allclose(repeats.face_raw[0], repeats.face_raw[2], equal_nan=True, atol=1e-6)
    root = Path(__file__).resolve().parents[1]
    registry = yaml.safe_load((root / "registries/checkpoints.yaml").read_text())["models"]["dinov2_vits14"]
    entry = {**registry, "repo_dir": args.dino_repo, "local_path": args.dino_weights, "sha256": file_hash(args.dino_weights)}
    adapter = DINOAdapter().load(entry, args.device)
    context = ExtractionContext(Path(args.video).stem, file_hash(args.video), decoded.source_frame_index,
                                raw, {"source_mirrored": None, "assignment": "same_frame_anatomical_body_wrists"})
    full = adapter.extract(decoded.frames, decoded.timestamps, context)
    names = ("timestamps", "source_frame_index", "visual", "visual_valid", "pose_xy_body", "pose_observed", "pose_quality", "pose_coordinate_valid")
    bundle = replace(full, **{name: getattr(full, name)[selected.indices] for name in names},
        raw_supports=tuple(s for t in selected.indices for s in full.raw_supports[t * 7:(t + 1) * 7]))
    # ROI transforms remain original-frame indexed in metadata with this explicit mapping.
    bundle.meta["selected_original_array_indices"] = selected.indices.tolist()
    bundle.meta["timestamp_fallback"] = decoded.timestamp_fallback
    write_bundle(output / "features", bundle)
    expected = {key: bundle.meta[key] for key in IDENTITY_FIELDS}
    cached = read_bundle(output / "features", expected)
    geometry = geometry_from_features(cached)
    tokens = SignTokenizer().build(cached, geometry)
    mask = MaskSampler().sample(tokens, np.random.default_rng(17))
    plan = ClosureResolver().resolve(tokens, mask)
    assert plan.visible and plan.semantic_targets
    assert all(not tokens.tokens[i].raw_support & mask.cells for i in plan.visible)
    assert all(tokens.geometry[i].support <= mask.cells for i in plan.geometry_targets)
    # Raw detector values retained separately; missing measurements remain NaN here.
    np.savez(output / "raw_landmarks.npz", xy=raw.xy, observed=raw.observed, quality=raw.quality,
             body_raw=raw.body_raw, hand_candidates=raw.hand_candidates, face_raw=raw.face_raw,
             handedness_scores=raw.handedness_scores, assignment=raw.assignment,
             timestamps=decoded.timestamps, source_frame_index=decoded.source_frame_index)
    write_json(output / "raw_landmarks.provenance.json", raw.provenance)
    report = {"status": "passed", "scope": "real_full_clip_preprocessing_cache_geometry_masking_only",
        "source_sha256": context.source_sha256, "decoded_frames": len(decoded.frames), "selected_samples": len(cached.timestamps),
        "selected_raw_distinct": len(np.unique(selected.source_frame_index)), "timestamp_fallback": decoded.timestamp_fallback,
        "stream_valid_counts": cached.visual_valid.sum(axis=0).tolist(), "coordinate_valid_counts": cached.pose_coordinate_valid.sum(axis=0).tolist(),
        "valid_geometry": sum(g.valid for g in geometry), "mask": plan.diagnostics, "device": args.device,
        "backbone": adapter.provenance(), "feature_extractor_hash": cached.meta["extractor_hash"],
        "mediapipe": raw.provenance, "image_mode_temporal_independence": "passed_repeat_frame_test",
        "runtime_seconds": time.monotonic() - start, "handedness_visual_review": "pending",
        "training": "not_run", "downstream": "not_run", "research_claim": None}
    write_json(output / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "scope", "decoded_frames", "selected_samples", "stream_valid_counts", "valid_geometry", "runtime_seconds")}))


if __name__ == "__main__":
    main()
