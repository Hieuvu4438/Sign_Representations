from dataclasses import replace
import numpy as np
from signpath.data.collate import TrainingExample, collate_examples
from signpath.data.dataset import fitting_geometry
from signpath.data.sampling import slice_features
from signpath.geometry.paths import geometry_from_features
from signpath.masking.closure import ClosureResolver
from signpath.masking.samplers import MaskSampler
from signpath.models.tokenizer import SignTokenizer
from signpath.targets.projection import ProjectionObservation, TargetProjectors
from model_helpers import tiny_config


def test_train_fit_covers_group_scale_visible_only_at_random_crop_phase(bundle):
    config = tiny_config()
    valid, quality, xy = bundle.pose_coordinate_valid.copy(), bundle.pose_quality.copy(), bundle.pose_xy_body.copy()
    valid[:, 7:], quality[:, 7:] = False, 0
    valid[9:13, 7:], quality[9:13, 7:] = True, 1
    xy[~valid] = 0
    bundle = replace(bundle, pose_coordinate_valid=valid, pose_quality=quality, pose_xy_body=xy)
    key = "bimanual_wrist_chain/0"
    phase0 = {f"{g.group}/{g.scale}" for g in geometry_from_features(bundle) if g.valid}
    assert key not in phase0
    all_phases = list(fitting_geometry(bundle, config))
    assert key in {f"{g.group}/{g.scale}" for g in all_phases if g.valid}
    records = [ProjectionObservation("source", "train", f"{g.group}/{g.scale}", g.descriptor) for g in all_phases if g.valid]
    stats = TargetProjectors.fit(records, allowed_train_sources={"source"}, manifest_sha256="a" * 64,
        extractor_hash=bundle.meta["extractor_hash"], descriptor_config={"all_center_phases": True})
    crop = slice_features(bundle, 3, 40)
    geo = geometry_from_features(crop)
    tokens = SignTokenizer().build(crop, geo)
    raw = MaskSampler().sample(tokens, np.random.default_rng(1))
    plan = ClosureResolver().resolve(tokens, raw)
    collated = collate_examples([TrainingExample(crop, tokens, plan, stats)], config["model"]["raw_dimensions"])
    assert collated.geometry_valid.any()
