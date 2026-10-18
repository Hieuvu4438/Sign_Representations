from copy import deepcopy
import torch
from signrepr.models.part_tokenizer import token_metadata
from signrepr.masking.part_temporal import MaskGenerator
from signrepr.losses.hybrid import HybridCriterion
from signrepr.losses.geometry import physical_huber_terms
from signrepr.losses.latent_ce import latent_ce_terms
from signrepr.training.system import SSLSystem
from signrepr.data.fixture import fixture_pose
from signrepr.data.normalization import canonicalize_sample
from signrepr.data.collate import collate_pose
from signrepr.losses.motion import motion_targets
import numpy as np


def test_motion_valid_endpoints_and_geometry_roots(config, batch):
    criterion = HybridCriterion(config['loss'], 16)
    meta = token_metadata(batch)
    masks = {'target': meta['valid']}
    targets, den = criterion.targets(batch, masks, meta)
    assert den['geometry'] > 0 and den['motion'] > 0
    for part, root in [('left_hand', 0), ('right_hand', 0), ('face', 17)]:
        assert not targets[part]['geometry'][1][:, :, root].any()
    changed = deepcopy(batch)
    changed['timing_known'][0] = False
    changed['track_continuity'][1, 3] = False
    changed['timestamps_sec'][1, 9:] += 1
    changed['parts']['body']['joint_valid'][1, 6, 2] = False
    changed['parts']['body']['global_joint_valid'][1, 6, 2] = False
    t, _ = criterion.targets(changed, masks, meta)
    for part in t:
        v = t[part]['motion'][1]
        assert not v[0].any()
        assert not v[1, 3].any()
        assert not v[1, 8].any()
        assert not v[:, -1].any()
    assert not t['body']['motion'][1][1, 5:7, 2].any()


def test_loss_mathematical_reference_and_nan_padding():
    pred = torch.tensor([[[.1, .2], [float('nan'), float('nan')]]], requires_grad=True)
    target = torch.tensor([[[.5, -.5], [float('nan'), float('nan')]]])
    num, den, stats = latent_ce_terms(pred, target, torch.tensor([[True, False]]), torch.tensor([[.7, .9]]), torch.zeros(2))
    expected = -(torch.softmax(target[0, 0]/.06, -1)*torch.log_softmax(pred[0, 0]/.1, -1)).sum()*.7
    torch.testing.assert_close(num, expected)
    torch.testing.assert_close(den, torch.tensor(.7))
    torch.testing.assert_close(stats, torch.tensor([.35, -.35]))
    p = torch.tensor([[.1, .1], [float('nan'), 0.]], requires_grad=True)
    t = torch.tensor([[0., 0.], [float('nan'), 0.]])
    num, den = physical_huber_terms(p, t, torch.tensor([.5, 1.]), torch.tensor([True, False]), .05)
    torch.testing.assert_close(num, torch.tensor(.0375))
    num.backward(); assert torch.isfinite(p.grad).all()


def test_teacher_no_grad_and_trainable_components(config, batch):
    system = SSLSystem(config['model'])
    teacher = deepcopy(system.encoder).eval().requires_grad_(False)
    criterion = HybridCriterion(config['loss'], 16)
    meta = token_metadata(batch)
    masks = MaskGenerator(config['mask'])(meta['valid'], torch.Generator().manual_seed(4))
    targets, _ = criterion.targets(batch, masks, meta)
    with torch.no_grad():
        t = teacher.encode_full(batch)['features']
    terms, _ = criterion(system(batch, masks), t, targets, masks, meta)
    sum(num/den.clamp_min(1) for num, den in terms.values()).backward()
    assert all(p.grad is None for p in teacher.parameters())
    for module in [system.encoder, system.predictor, system.geometry, system.motion]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_motion_physical_units_and_missing_root_geometry_separation(config):
    pose = fixture_pose(3)
    pose['xy'][:, 0] = np.array([[.5, .5], [.52, .5], [.56, .5]])
    pose['timestamps_sec'] = np.array([0., .08, .24])
    pose['observed'][1, 91] = False
    batch = collate_pose([canonicalize_sample(pose)])
    target, valid, weights = motion_targets(batch, batch['parts']['body'])
    torch.testing.assert_close(target[0, :2, 0], torch.tensor([[.05, 0.], [.05, 0.]]), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(weights[0, :2, 0], torch.full((2,), .81))
    criterion = HybridCriterion(config['loss'], 16)
    meta = token_metadata(batch)
    mask = torch.zeros_like(meta['valid']); mask[0, 1] = True
    targets, _ = criterion.targets(batch, {'target': mask}, meta)
    # Local finger geometry at t=1 is undefined without its root. Global finger
    # trajectory t=0->1 remains observed in the fixed shoulder coordinate frame.
    assert not targets['left_hand']['geometry'][1][0, 1].any()
    assert targets['left_hand']['motion'][1][0, 0, 1]
