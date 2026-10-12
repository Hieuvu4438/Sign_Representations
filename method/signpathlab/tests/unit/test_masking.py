from dataclasses import replace
import numpy as np

from signpath.contracts import RawMask
from signpath.geometry.paths import build_geometry
from signpath.geometry.supports import Primitive
from signpath.masking.closure import ClosureResolver
from signpath.masking.samplers import MaskSampler
from signpath.models.tokenizer import SignTokenizer


def tokenize(bundle, **kwargs):
    geo = build_geometry(bundle.meta["sample_id"], bundle.timestamps, bundle.source_frame_index,
                         bundle.pose_xy_body, bundle.pose_coordinate_valid, bundle.pose_quality, **kwargs)
    return SignTokenizer().build(bundle, geo)


def test_normalization_and_whole_group_closure(bundle):
    tokens = tokenize(bundle)
    mask = RawMask(frozenset({Primitive("joint", 12, 1)}), 0.1, 1 / 48, "cross_modal")
    plan = ClosureResolver().resolve(tokens, mask)
    assert all(12 * 7 + stream in plan.hidden for stream in (4, 5, 6))
    assert 12 * 7 + 3 in plan.visible  # independent face stream
    # Geometry with even one touched dependency cannot be visible.
    assert all(not tokens.geometry[i].support & mask.cells for i in plan.geometry_visible)
    assert not plan.geometry_targets  # a shoulder alone cannot fully hide a path


def test_strict_targets_and_counterfactual(bundle):
    original = tokenize(bundle)
    frames = set(range(8, 32))
    cells = frozenset(p for token in original.tokens for p in token.raw_support if p.frame in frames)
    cells |= frozenset(p for g in original.geometry for p in g.support if p.frame in frames)
    mask = RawMask(cells, .5, .5, "strict_temporal")
    plan = ClosureResolver().resolve(original, mask)
    assert plan.geometry_targets and plan.semantic_targets and plan.visible
    assert all(original.geometry[i].support <= cells for i in plan.geometry_targets)
    # Change all hidden values, retaining validity and the fixed public schedule.
    xy, visual = bundle.pose_xy_body.copy(), bundle.visual.copy()
    xy[list(frames)] += 17
    visual[list(frames)] *= -3
    modified = tokenize(replace(bundle, pose_xy_body=xy, visual=visual))
    modified_plan = ClosureResolver().resolve(modified, mask)
    assert modified_plan == plan
    for index in plan.visible:
        np.testing.assert_array_equal(original.tokens[index].values, modified.tokens[index].values)
    for index in plan.geometry_visible:
        np.testing.assert_array_equal(original.geometry[index].descriptor, modified.geometry[index].descriptor)
    assert any(not np.allclose(original.tokens[i].values, modified.tokens[i].values) for i in plan.hidden)


def test_aliases_and_mask_sampling(bundle):
    # Duplicate every source observation, exactly as low-fps nearest resampling does.
    names = ("timestamps", "source_frame_index", "visual", "visual_valid", "pose_xy_body", "pose_observed", "pose_quality", "pose_coordinate_valid")
    aliases = replace(bundle, **{name: np.repeat(getattr(bundle, name), 2, axis=0) for name in names},
        raw_supports=tuple(s for t in range(48) for _ in range(2) for s in bundle.raw_supports[t * 7:(t + 1) * 7]))
    aliases.validate(allow_fixture=True)
    tokens = tokenize(aliases)
    sampler = MaskSampler()
    raw = sampler.sample(tokens, np.random.default_rng(12))
    plan = ClosureResolver().resolve(tokens, raw)
    assert raw.actual_source_ratio == .5
    assert plan.diagnostics["token_mask_ratio_after_closure"] == .5
    for frame in range(48):
        for stream in range(7):
            assert (frame * 14 + stream in plan.hidden) == (frame * 14 + 7 + stream in plan.hidden)
    assert sampler.sample(tokens, np.random.default_rng(12)) == raw


def test_missing_pose_is_context_with_explicit_bits_but_not_target(bundle):
    valid = bundle.pose_coordinate_valid.copy()
    valid[10, 7] = False
    xy, quality = bundle.pose_xy_body.copy(), bundle.pose_quality.copy()
    xy[10, 7], quality[10, 7] = 0, 0
    tokens = tokenize(replace(bundle, pose_coordinate_valid=valid, pose_xy_body=xy, pose_quality=quality))
    left = tokens.tokens[10 * 7 + 5]
    assert left.valid and not left.target_valid
    np.testing.assert_array_equal(left.values[:4], [0, 0, 0, 0])
    mask = RawMask(left.raw_support, .5, 1 / 48, "cross_modal")
    plan = ClosureResolver().resolve(tokens, mask)
    assert 10 * 7 + 5 in plan.hidden and 10 * 7 + 5 not in plan.semantic_targets


def test_flattened_stream_indexing():
    assert [(i // 4, i % 4) for i in range(3, 6)] == [(0, 3), (1, 0), (1, 1)]
    assert [t * 4 + 3 for t in range(3)] == [3, 7, 11]
