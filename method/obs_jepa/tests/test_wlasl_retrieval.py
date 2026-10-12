"""Protocol regressions: gloss ranking, raw pooling, activity fallback, pose roots."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oj.wlasl_retrieval import gpu_ready, pool_windows, score_retrieval


def test_shared_gpu_ignores_other_jobs_but_requires_free_memory():
    assert gpu_ready([123], 18000, 8000, True)
    assert not gpu_ready([123], 18000, 8000, False)
    assert not gpu_ready([123], 7000, 8000, True)
    assert gpu_ready([], 8000, 8000, False)


def test_pool_raw_before_clip_normalization():
    x = np.array([[4., 0.], [0., 2.]])
    result, fallback = pool_windows(x, np.array([[.2, .1], [.8, 0.]]))
    np.testing.assert_allclose(result["avg"], [2, 1])
    np.testing.assert_allclose(result["weighted"], [.8, 1.6])
    assert not fallback


def test_activity_clamp_and_zero_fallback():
    x = np.array([[4., 0.], [0., 2.]])
    result, fallback = pool_windows(x, np.array([[-2, -1], [2, 3]]))
    np.testing.assert_allclose(result["weighted"], [0, 2])
    assert not fallback
    result, fallback = pool_windows(x, np.full((2, 2), -1.))
    np.testing.assert_array_equal(result["avg"], result["weighted"])
    assert fallback


def test_distinct_gloss_rank_not_video_rank():
    gallery = np.array([[1., 0.], [.99, .01], [0., 1.], [-1., 0.]])
    labels = [0, 0, 1, 2]
    query = np.array([[1., 0.], [-1., 0.], [0., 0.]])
    metrics, ranks = score_retrieval(gallery, query, labels, [1, 2, 2], 3, chunk=1)
    np.testing.assert_array_equal(ranks, [2, 1, 3])
    assert metrics["Rec1"] == 1 / 3
    assert metrics["Rec5"] == 1
    assert abs(metrics["DCG"] - (1 / np.log2(3) + 1 + .5) / 3) < 1e-12
    assert metrics["true_score_ties"] == 1


def test_invalid_features_or_missing_gallery_classes_fail():
    with pytest.raises(ValueError, match="Nonfinite"):
        score_retrieval(np.array([[np.nan, 0]]), np.ones((1, 2)), [0], [0], 1)
    with pytest.raises(ValueError, match="coverage"):
        score_retrieval(np.ones((1, 2)), np.ones((1, 2)), [0], [0], 2)
    with pytest.raises(ValueError, match="Empty"):
        pool_windows(np.zeros((0, 2)))


def test_explicit_wlasl_landmark_root_uses_same_schema(tmp_path):
    from oj.data.landmarks import load_landmarks
    with (tmp_path / "sample.pkl").open("wb") as f:
        pickle.dump(dict(keypoints=[np.full((1, 133, 2), .5)], scores=[np.full((1, 133), 10.)]), f)
    lm = load_landmarks("sample", 20, 10, root=tmp_path)
    assert lm.xy.shape == (1, 67, 2) and lm.observed.all()
    np.testing.assert_allclose(lm.xy[0], np.broadcast_to([10., 5.], (67, 2)))


@pytest.mark.parametrize("local,qstride", [(False, 1), (False, 2), (True, 1), (True, 2)])
def test_fast_hiera_attention_matches_original(local, qstride):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method/jepa_baseline/src"))
    from signrep_baseline.upstream_signrep.models.hiera.hiera_clstoken import MaskUnitAttention
    from oj.extract_wlasl import fast_no_cls_attention
    torch.manual_seed(1)
    m = MaskUnitAttention(16, 32, heads=4, q_stride=qstride, window_size=4,
                          use_mask_unit_attn=local, num_cls_tokens=0).eval()
    x = torch.randn(2, 32, 16)
    with torch.inference_mode():
        expected = m(x)
        actual = fast_no_cls_attention(m, x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
