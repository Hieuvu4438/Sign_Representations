"""Rerun actual raw RGB -> detectors/DINO after editing only hidden source frames."""
import argparse
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import numpy as np
import torch
import yaml

from signpath.backbones.dino import DINOAdapter
from signpath.cache.hashing import file_hash
from signpath.cli import write_json
from signpath.data.decode import decode_video, nearest_observations
from signpath.geometry.paths import geometry_from_features
from signpath.masking.closure import ClosureResolver
from signpath.masking.samplers import MaskSampler
from signpath.models.tokenizer import SignTokenizer
from signpath.pose.context import ExtractionContext
from signpath.pose.extract import LandmarkExtractor


def selected_bundle(bundle, indices):
    fields = ("timestamps", "source_frame_index", "visual", "visual_valid", "pose_xy_body", "pose_observed", "pose_quality", "pose_coordinate_valid")
    return replace(bundle, **{name: getattr(bundle, name)[indices] for name in fields},
        raw_supports=tuple(s for t in indices for s in bundle.raw_supports[t * 7:(t + 1) * 7]))


def main():
    parser = argparse.ArgumentParser()
    for name in ("video", "dino-repo", "dino-weights", "pose-model", "hand-model", "face-model", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Counterfactual report exists; use another output path")
    torch.set_num_threads(1)
    decoded = decode_video(args.video)
    selection = nearest_observations(decoded.timestamps, decoded.source_frame_index, 25)
    root = Path(__file__).resolve().parents[1]
    entry = yaml.safe_load((root / "registries/checkpoints.yaml").read_text())["models"]["dinov2_vits14"]
    entry.update(repo_dir=args.dino_repo, local_path=args.dino_weights, sha256=file_hash(args.dino_weights))
    adapter = DINOAdapter().load(entry, args.device)
    # Fixed one-image batches remove batch-kernel rounding as a confound of this test.
    adapter.encode_crops = partial(adapter.encode_crops, batch_size=1)
    def extract(frames, extractor):
        raw = extractor.extract(frames)
        context = ExtractionContext(Path(args.video).stem, file_hash(args.video), decoded.source_frame_index,
                                    raw, {"source_mirrored": None, "counterfactual_gate": True})
        return selected_bundle(adapter.extract(frames, decoded.timestamps, context), selection.indices)
    with LandmarkExtractor(args.pose_model, args.hand_model, args.face_model) as extractor:
        original = extract(decoded.frames, extractor)
        original_geo = geometry_from_features(original)
        original_tokens = SignTokenizer().build(original, original_geo)
        mask = MaskSampler().sample(original_tokens, np.random.default_rng(17))
        plan = ClosureResolver().resolve(original_tokens, mask)
        hidden_frames = {p.frame for p in mask.cells}
        edited = decoded.frames.copy()
        for i, source in enumerate(decoded.source_frame_index):
            if int(source) in hidden_frames:
                edited[i] = 0
        changed = extract(edited, extractor)
    changed_geo = geometry_from_features(changed)
    changed_tokens = SignTokenizer().build(changed, changed_geo)
    max_error = 0.0
    for index in plan.visible:
        a, b = original_tokens.tokens[index], changed_tokens.tokens[index]
        assert a.valid == b.valid and a.target_valid == b.target_valid
        np.testing.assert_allclose(a.values, b.values, atol=1e-6, rtol=1e-6)
        max_error = max(max_error, float(np.max(np.abs(a.values - b.values))))
        assert a.id == b.id and a.stream == b.stream and a.timestamp == b.timestamp
    for index in plan.geometry_visible:
        a, b = original_geo[index], changed_geo[index]
        assert a.valid and b.valid and a.id == b.id
        np.testing.assert_allclose(a.descriptor, b.descriptor, atol=1e-6, rtol=1e-6)
    assert any(not np.array_equal(original_tokens.tokens[i].values, changed_tokens.tokens[i].values) for i in plan.hidden)
    report = {"status": "passed", "scope": "real_raw_video_pretrained_strict_temporal_counterfactual",
        "source_sha256": file_hash(args.video), "source_frames_edited": len(hidden_frames),
        "retained_raw_tokens_compared": len(plan.visible), "retained_geometry_tokens_compared": len(plan.geometry_visible),
        "max_visible_absolute_error": max_error, "atol": 1e-6, "rtol": 1e-6,
        "mask_and_public_query_schedule": "held_fixed", "hidden_detector_validity": "allowed_to_change; not used to select a different query schedule",
        "backbone": adapter.provenance(), "device": args.device, "research_claim": None}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
