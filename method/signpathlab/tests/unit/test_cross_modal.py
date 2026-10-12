from dataclasses import replace
import numpy as np
import pytest
import torch
from signpath.data.collate import collate_examples
from signpath.data.dataset import BatchProvider
from signpath.data.fixtures import synthetic_features
from signpath.geometry.paths import geometry_from_features
from signpath.masking.cross_modal import CrossModalSampler, observation_view
from signpath.masking.closure import ClosureResolver
from signpath.models.tokenizer import SignTokenizer
from signpath.models.teacher_student import ResearchModel
from signpath.training.engine import Trainer
from model_helpers import tiny_config


@pytest.mark.parametrize("strategy", ["anatomical_pose", "stream_temporal", "modality_dropout"])
def test_cross_modal_masks_are_explicit_alias_aware_and_trainable(strategy):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = tiny_config(steps=2)
        config["geometry"]["enabled"] = False
        config["loss"]["geometry_weight"] = 0.
        config["masking"].update(regime="cross_modal", strategy=strategy)
        bundle = synthetic_features("sample", visual_dim=8)
        provider = BatchProvider(config, {"sample": bundle}, {"source": ["sample"]})
        batch = provider.next_batch()
        assert batch.semantic_targets.any() and batch.visible.any()
        assert all(d["mask_regime"] == "cross_modal" and d["mask_strategy"] == strategy for d in batch.diagnostics)
        if strategy == "anatomical_pose":
            assert torch.equal(batch.visible[:, :, :4], batch.valid[:, :, :4])
            assert not batch.semantic_targets[:, :, :4].any()
        trainer = Trainer(ResearchModel(config), config)
        assert trainer.update([batch]).succeeded
    finally:
        torch.set_num_threads(previous)


def test_pose_observation_counterfactual_preserves_declared_rgb_context(bundle):
    tokens = SignTokenizer().build(bundle, geometry_from_features(bundle))
    view = observation_view(tokens, bundle.source_frame_index, "anatomical_pose")
    raw = CrossModalSampler("anatomical_pose").sample(view, np.random.default_rng(17))
    plan = ClosureResolver().resolve(view, raw)
    assert all(not (view.tokens[i].raw_support & raw.cells) for i in plan.visible)
    changed_values = bundle.pose_xy_body.copy()
    for p in raw.cells:
        changed_values[p.frame, p.index] += 123
    changed = replace(bundle, pose_xy_body=changed_values)
    modified = observation_view(SignTokenizer().build(changed, geometry_from_features(changed)), changed.source_frame_index, "anatomical_pose")
    for i in plan.visible:
        np.testing.assert_array_equal(view.tokens[i].values, modified.tokens[i].values)
    assert all(i in plan.visible for i in range(len(view.tokens)) if view.tokens[i].valid and view.tokens[i].stream < 4)


def test_anatomical_aliases_share_same_pose_observation_mask(bundle):
    names = ("timestamps", "source_frame_index", "visual", "visual_valid", "pose_xy_body", "pose_observed", "pose_quality", "pose_coordinate_valid")
    alias = replace(bundle, **{name: np.repeat(getattr(bundle, name), 2, axis=0) for name in names},
        raw_supports=tuple(s for t in range(48) for _ in range(2) for s in bundle.raw_supports[t * 7:(t + 1) * 7]))
    tokens = observation_view(SignTokenizer().build(alias), alias.source_frame_index, "anatomical_pose")
    plan = ClosureResolver().resolve(tokens, CrossModalSampler("anatomical_pose").sample(tokens, np.random.default_rng(4)))
    for frame in range(48):
        for stream in range(7):
            assert (frame * 14 + stream in plan.hidden) == (frame * 14 + 7 + stream in plan.hidden)
