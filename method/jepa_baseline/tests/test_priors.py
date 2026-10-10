import pytest
import torch
from signrep_baseline.priors import build_priors, activity_prior
from signrep_baseline.schema import FidelityChoices, PRIOR_SHAPES, BODY_SOURCE, BODY_DEST

CONSERVATIVE = FidelityChoices(angle_confidence="all_observed_points")


def inputs():
    pose = torch.arange(4 * 61 * 3, dtype=torch.float32).reshape(4, 61, 3) / 1000
    conf = torch.ones(4, 61)
    angles = {"body": torch.linspace(-3, 3, 88).reshape(4, 22),
              "right": torch.ones(4, 41) * .3, "left": torch.ones(4, 41) * -.7}
    return pose, conf, angles


def test_schema_geometry_and_periodic_angles():
    pose, conf, angles = inputs()
    targets, masks = build_priors(pose, conf, angles, CONSERVATIVE)
    for name, shape in PRIOR_SHAPES.items():
        assert targets[name].shape == (4, torch.tensor(shape).prod())
        assert masks[name].all()
    assert (targets["right_pose"].reshape(4, 21, 3)[:, 0] == 0).all()
    values = targets["full_body_angles"].reshape(4, 22, 2)
    torch.testing.assert_close(values.square().sum(-1), torch.ones(4, 22))
    shifted = {part: value + 2 * torch.pi for part, value in angles.items()}
    periodic, _ = build_priors(pose, conf, shifted, CONSERVATIVE)
    torch.testing.assert_close(periodic["full_body_angles"], targets["full_body_angles"], atol=1e-6, rtol=1e-6)
    distance = targets["body_dist"].reshape(4, 12, 22, 3)
    # Independent endpoint arithmetic, including a direction where the sign changes.
    for i, source in enumerate(BODY_SOURCE):
        for j, destination in enumerate(BODY_DEST):
            torch.testing.assert_close(distance[:, i, j], pose[:, source] - pose[:, destination])
    translated, _ = build_priors(pose + torch.tensor([2., -3., 7.]), conf, angles, CONSERVATIVE)
    for name in ["right_pose", "left_pose", "body_dist", "right_dist", "left_dist"]:
        torch.testing.assert_close(translated[name], targets[name], atol=1e-6, rtol=1e-5)


def test_missing_wrist_strict_threshold_and_nan_are_masked():
    pose, conf, angles = inputs()
    conf[:, 19] = .5  # must be >.5, not >=
    pose[0, 44] = float("nan")
    targets, masks = build_priors(pose, conf, angles, CONSERVATIVE)
    assert not masks["right_pose"].any()
    assert not masks["full_right_angles"].any()
    assert not masks["activity"].any()
    for value in targets.values():
        assert torch.isfinite(value).all()
    assert not masks["body_dist"].reshape(4, 12, 22, 3)[:, 0].any()


def test_activity_hand_order_and_stationary_below_threshold():
    pose = torch.zeros(16, 61, 3)
    pose[:, 19:40, 1] = 1.0  # right resting below torso
    pose[:, 40:61, 1] = -.1  # left above torso
    target, valid = activity_prior(pose, torch.ones(16, 61, dtype=torch.bool))
    assert valid.tolist() == [True, True]
    assert target.tolist() == [1., 0.]  # left, right
    pose[:, 19:40, 0] = torch.linspace(0, 1, 16)[:, None]
    target, _ = activity_prior(pose, torch.ones(16, 61, dtype=torch.bool))
    assert target.tolist() == [1., 1.]


def test_no_scalar_distance_or_wrong_angle_schema():
    with pytest.raises(ValueError):
        FidelityChoices(distance="euclidean")
    pose, conf, angles = inputs()
    angles["right"] = torch.zeros(4, 15)
    with pytest.raises(ValueError):
        build_priors(pose, conf, angles, CONSERVATIVE)


def test_global_anchor_missing_preserves_local_hand_geometry():
    pose, conf, angles = inputs()
    conf[:, 2] = 0
    _, masks = build_priors(pose, conf, angles, CONSERVATIVE)
    assert not masks["body_pose"].reshape(4, 61, 3)[:, 19:40].any()
    assert not masks["body_dist"].reshape(4, 12, 22, 3)[:, :6].any()
    assert masks["right_pose"].all() and masks["right_dist"].all()
    assert not masks["activity"][1]


def test_angles_use_explicit_teacher_observations_not_unrelated_joint_visibility():
    pose, conf, angles = inputs()
    conf[:, 7] = 0  # Low hip visibility masks keypoints but does not erase inferred body angles.
    flags = {part: torch.ones(4, dtype=torch.bool) for part in ["body", "right", "left"]}
    flags["right"][0] = False
    _, masks = build_priors(pose, conf, angles, angle_observed=flags)
    assert masks["full_body_angles"].all()
    assert not masks["full_right_angles"][0].any()
    assert not masks["body_pose"].reshape(4, 61, 3)[:, 7].any()
    with pytest.raises(ValueError, match="explicit teacher-detection"):
        build_priors(pose, conf, angles)
