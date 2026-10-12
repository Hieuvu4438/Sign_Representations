import numpy as np
import pytest
import torch
from signpath.geometry.anchors import signature_descriptor, timing_motion
from signpath.geometry.logsig2 import logsig2, logsig2_reference
from signpath.geometry.motion_controls import ordinary_motion_descriptor
from signpath.geometry.paths import DEFAULT_PATHS, build_geometry

L = np.array([[0., 0.], [1., 0.], [1., 1.]])


def test_analytical_signature():
    np.testing.assert_allclose(logsig2_reference(L), [1, 1, .5], atol=1e-10)
    np.testing.assert_allclose(logsig2_reference([[0, 0], [1, 1]]), [1, 1, 0], atol=1e-10)
    refined = np.array([[0, 0], [.5, 0], [1, 0], [1, .5], [1, 1]])
    np.testing.assert_allclose(logsig2_reference(refined), logsig2_reference(L), atol=1e-10)
    np.testing.assert_allclose(logsig2_reference(L + [8, -3]), logsig2_reference(L), atol=1e-10)
    np.testing.assert_allclose(logsig2_reference(L[::-1]), -logsig2_reference(L), atol=1e-10)
    np.testing.assert_allclose(logsig2_reference(3 * L), [3, 3, 4.5], atol=1e-10)
    transform = np.array([[2., 1.], [0., 3.]])
    assert logsig2_reference(L @ transform.T)[-1] == pytest.approx(np.linalg.det(transform) * .5)


@pytest.mark.parametrize("d,ls,native,motion", [(2, 3, 12, 19), (4, 10, 25, 35), (6, 21, 42, 51)])
def test_dimensions_and_static_anchors(d, ls, native, motion):
    a, b = np.tile(np.full(d, .2), (4, 1)), np.tile(np.full(d, .7), (4, 1))
    times = np.arange(4) / 25
    np.testing.assert_allclose(logsig2_reference(a), np.zeros(ls), atol=1e-10)
    np.testing.assert_allclose(logsig2_reference(b), np.zeros(ls), atol=1e-10)
    x, y = signature_descriptor(a, times), signature_descriptor(b, times)
    assert x.shape == (native,) and ordinary_motion_descriptor(a, times).shape == (motion,)
    assert not np.allclose(x, y)


@pytest.mark.parametrize("d", [2, 4, 6])
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-10), (torch.float32, 1e-5), (torch.float16, 1e-4)])
def test_torch_batched_oracle_and_gradients(d, dtype, tol):
    x = torch.tensor(np.random.default_rng(71).normal(size=(3, 17, d)), dtype=dtype, requires_grad=True)
    output = logsig2(x)
    oracle = np.stack([logsig2_reference(p) for p in x.detach().double().numpy()])
    np.testing.assert_allclose(output.detach().numpy(), oracle, atol=tol, rtol=1e-4)
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert output.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)


def test_path_groups_windows_and_missing(bundle):
    tokens = build_geometry("sample", bundle.timestamps, bundle.source_frame_index, bundle.pose_xy_body,
        bundle.pose_coordinate_valid, bundle.pose_quality)
    assert len(tokens) == 12 * 11 * 3
    assert len({t.id for t in tokens}) == len(tokens)
    assert all(t.reason == "incomplete_window" for t in tokens[:33])
    selected = next(t for t in tokens if t.center_index == 12 and t.group == "left_wrist" and t.scale == 1)
    assert selected.valid and selected.distinct_raw_frames == 8
    assert {p.index for p in selected.support} == {1, 2, 7}
    valid = bundle.pose_coordinate_valid.copy()
    valid[12, 7] = False
    invalid = build_geometry("sample", bundle.timestamps, bundle.source_frame_index, bundle.pose_xy_body, valid, bundle.pose_quality)
    changed = next(t for t in invalid if t.id == selected.id)
    assert not changed.valid and changed.reason == "missing_or_low_quality"
    assert DEFAULT_PATHS[4].dimension == 2
    np.testing.assert_allclose(DEFAULT_PATHS[4].values(bundle.pose_xy_body), bundle.pose_xy_body[:, 15] - bundle.pose_xy_body[:, 7])


def test_zero_dt_and_nonfinite_rejected():
    with pytest.raises(ValueError):
        timing_motion(L, [0, .1, .1])
    with pytest.raises(ValueError):
        logsig2_reference([[0, 0], [np.nan, 1]])
    with pytest.raises(ValueError):
        logsig2(torch.ones(1, 2))
