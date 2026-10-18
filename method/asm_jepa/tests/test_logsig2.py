"""Level-2 log-signature properties and basis convention."""
import numpy as np
import torch

from asm_jepa.features.logsig2 import logsig2_dim, logsig2_np, logsig2_torch


def test_constant_path_is_zero():
    assert np.allclose(logsig2_np(np.ones((10, 3)) * 4.2), 0.0)


def test_translation_invariance():
    p = np.random.default_rng(0).normal(size=(12, 2)).cumsum(0)
    assert np.allclose(logsig2_np(p), logsig2_np(p + np.array([5.0, -3.0])))


def test_line_out_and_back_cancels():
    out = np.linspace([0, 0], [2, 1], 6)
    path = np.concatenate([out, out[::-1][1:]])
    assert np.allclose(logsig2_np(path), 0.0, atol=1e-12)


def test_basis_order_on_l_shape():
    # (0,0)->(1,0)->(1,1): increment (1,1); Levy area = 1/2 (int x dy - y dx) = 0.5
    ls = logsig2_np(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]))
    assert np.allclose(ls, [1.0, 1.0, 0.5])
    # reversed orientation flips the area sign only
    ls2 = logsig2_np(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    assert np.allclose(ls2, [1.0, 1.0, -0.5])
    assert logsig2_dim(2) == 3 and logsig2_dim(3) == 6


def test_time_channel_distinguishes_speed_profiles():
    geom = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    fast_then_slow = np.c_[geom, [0.0, 0.2, 1.0]]
    slow_then_fast = np.c_[geom, [0.0, 0.8, 1.0]]
    assert np.allclose(logsig2_np(fast_then_slow[:, :2]), logsig2_np(slow_then_fast[:, :2]))
    assert not np.allclose(logsig2_np(fast_then_slow), logsig2_np(slow_then_fast))


def test_increment_mask_equals_cropped_path_and_torch_matches():
    p = np.random.default_rng(1).normal(size=(20, 3)).cumsum(0)
    support = np.zeros(20, bool)
    support[4:11] = True
    inc = support[1:] & support[:-1]
    assert np.allclose(logsig2_np(p, inc), logsig2_np(p[4:11]))
    t = logsig2_torch(torch.as_tensor(p), torch.as_tensor(inc))
    assert np.allclose(t.numpy(), logsig2_np(p, inc))
