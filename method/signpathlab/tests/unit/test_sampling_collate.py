from dataclasses import replace
import numpy as np
import torch
import pytest
from signpath.data.collate import collate_examples
from signpath.data.dataset import BatchProvider
from signpath.data.fixtures import synthetic_features
from signpath.data.sampling import SourceSegmentSampler
from signpath.training.runner import smoke_inputs
from model_helpers import example, tiny_config


def test_source_sampler_resume_exact_and_short_clip_padding():
    bundles = {"short": synthetic_features("short", length=13, visual_dim=8)}
    config = tiny_config()
    config["geometry"]["enabled"] = False
    config["loss"]["geometry_weight"] = 0.
    provider = BatchProvider(config, bundles, {"source": ("short",)})
    state = provider.state_dict()
    mask_state = provider.mask_generator.bit_generator.state
    expected = provider.next_batch()
    provider.load_state_dict(state)
    provider.mask_generator.bit_generator.state = mask_state
    actual = provider.next_batch()
    assert actual.lengths.tolist() == [13, 13]
    assert not actual.valid[:, 13:].any() and not actual.semantic_targets[:, 13:].any()
    assert (actual.source_frame_index[:, 13:] == -1).all()
    torch.testing.assert_close(actual.visible, expected.visible, atol=0, rtol=0)


def test_duplicate_source_observations_count_once_in_targets(bundle):
    names = ("timestamps", "source_frame_index", "visual", "visual_valid", "pose_xy_body", "pose_observed", "pose_quality", "pose_coordinate_valid")
    aliases = replace(bundle, **{name: np.repeat(getattr(bundle, name), 2, axis=0) for name in names},
        raw_supports=tuple(s for t in range(48) for _ in range(2) for s in bundle.raw_supports[t * 7:(t + 1) * 7]))
    config = tiny_config()
    batch = collate_examples([example(aliases)], config["model"]["raw_dimensions"])
    assert not batch.target_valid[:, 1::2].any()
    assert not batch.semantic_targets[:, 1::2].any()
    assert batch.valid[:, 1::2].all()


def test_synthetic_backend_is_explicit():
    bundle = synthetic_features("fixture", length=20, visual_dim=8)
    assert bundle.meta["synthetic_correctness_only"]
    assert not bundle.pose_observed[6:9, 7:28].any()
    assert not bundle.pose_coordinate_valid[6:9, 7:28].any()


def test_cross_modal_config_is_not_silently_run_as_strict_temporal():
    config = tiny_config()
    config["masking"]["regime"] = "cross_modal"
    with pytest.raises(ValueError, match="strict_temporal"):
        BatchProvider(config, {}, {"source": ("clip",)})
