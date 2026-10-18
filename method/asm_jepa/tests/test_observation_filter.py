"""Audit-driven observation filters: out-of-image joints, hands far from their own body wrist, no cross-part reads."""
import numpy as np

from asm_jepa.config import arch_diff, load_config
from asm_jepa.data.observation import observation_filter
from asm_jepa.data.schema import LH, RH

CFG = {"in_image_only": True, "in_image_margin": 0.0, "hand_wrist_max_dist": 0.3}
WH = np.array([640, 480], np.int32)


def _clip(T=6, seed=0):
    rng = np.random.default_rng(seed)
    xy = np.zeros((T, 133, 2), np.float32)
    xy[:] = rng.uniform(250, 390, size=(T, 133, 2))
    xy[:, 5] = (380, 200)  # COCO left shoulder (signer's left = image right)
    xy[:, 6] = (260, 200)  # shoulder width 120 px
    xy[:, 9] = (360, 330)  # body left wrist
    xy[:, 10] = (280, 330)  # body right wrist
    xy[:, LH] = xy[:, [9]] + rng.normal(0, 5, (T, 21, 2))
    xy[:, RH] = xy[:, [10]] + rng.normal(0, 5, (T, 21, 2))
    xy[:, LH[0]] = xy[:, 9]
    xy[:, RH[0]] = xy[:, 10]
    return xy, np.ones((T, 133), bool)


def test_disabled_filters_are_identity():
    xy, ob = _clip()
    xy[0, 30] = (-50, 900)
    out = observation_filter(xy, ob, WH, {})
    assert out.all() and out is not ob


def test_out_of_image_joints_become_unobserved():
    xy, ob = _clip()
    xy[1, RH[3]] = (300, 481)  # below the bottom edge
    xy[2, 40] = (-1, 100)
    out = observation_filter(xy, ob, WH, CFG)
    assert not out[1, RH[3]] and not out[2, 40]
    assert out.sum() == ob.sum() - 2
    out_margin = observation_filter(xy, ob, WH, {**CFG, "in_image_margin": 0.01})
    assert out_margin[1, RH[3]] and out_margin[2, 40]


def test_aliased_hand_far_from_its_body_wrist_is_dropped():
    xy, ob = _clip()
    xy[3, RH] = xy[3, LH]  # right hand copied onto the left hand (alias)
    out = observation_filter(xy, ob, WH, CFG)
    assert not out[3, RH].any() and out[3, LH].all()
    assert out[np.arange(6) != 3][:, RH].all()


def test_hand_decision_does_not_read_the_other_hand_or_face():
    xy, ob = _clip()
    base = observation_filter(xy, ob, WH, CFG)
    rng = np.random.default_rng(5)
    for other, own in ((RH, LH), (LH, RH)):
        donor = xy.copy()
        donor[:, other] = rng.uniform(0, 640, size=(6, 21, 2))
        donor[:, 23:91] = rng.uniform(0, 480, size=(6, 68, 2))
        assert np.array_equal(observation_filter(donor, ob, WH, CFG)[:, own], base[:, own])


def test_filter_keys_are_part_of_the_architecture_signature():
    a = load_config("configs/pretrain/full.yaml", "pretrain")
    b = load_config("configs/pretrain/final.yaml", "pretrain")
    assert any("hand_wrist_max_dist" in d for d in arch_diff(a, b))
    old = {**a, "data": {k: v for k, v in a["data"].items() if k not in CFG}}  # checkpoint written before the keys
    assert arch_diff(old, a) == []


def test_classifier_mode_follows_inherits():
    from asm_jepa.cli.train_classifier import config_mode

    for c in ("finetune.yaml", "finetune_full.yaml", "finetune_final.yaml", "finetune_final_nofilter.yaml"):
        assert config_mode(f"configs/eval/{c}") == "finetune_islr", c
    assert config_mode("configs/eval/linear.yaml") == "linear_probe"


def test_group_weights_default_is_identity_and_reweights():
    import torch

    from asm_jepa.training.losses import jepa_loss

    torch.manual_seed(0)
    n, D = 64, 16
    pred, z = torch.randn(n, D), torch.randn(n, D)
    groups = torch.arange(n) % 4  # four state groups
    kw = dict(weights=torch.ones(n), groups=groups, subgroups=torch.zeros(n, dtype=torch.long),
              centers=torch.zeros(9, D), student_temp=0.1, teacher_temp=0.06,
              family_weights={"state": 1.0, "relation": 0.5, "geometry": 0.25, "timing": 0.25})
    base, _ = jepa_loss(pred, z, **kw)
    same, _ = jepa_loss(pred, z, group_weights=[1.0] * 9, **kw)
    assert torch.allclose(base, same)
    low_face, logs = jepa_loss(pred, z, group_weights=[1, 1, 1, 0.0, 1, 1, 1, 1, 1], **kw)
    expect = torch.stack([logs[f"loss/group_{g}"] for g in range(3)]).mean()
    assert torch.allclose(low_face, expect, atol=1e-5) and not torch.allclose(low_face, base)
