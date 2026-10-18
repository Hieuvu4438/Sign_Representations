"""Ranking metrics, gloss-max retrieval (chunked == dense), deterministic ties, failure accounting."""
import numpy as np

from asm_jepa.evaluation.classification import classification_metrics
from asm_jepa.evaluation.ranking import metrics_from_ranks, ranks_from_scores
from asm_jepa.evaluation.retrieval import dense_reference_ranks, evaluate_retrieval, gloss_max_scores


def test_metrics_from_known_ranks():
    m = metrics_from_ranks([1, 2, 5])
    assert np.isclose(m["r1"], 1 / 3) and np.isclose(m["r5"], 1.0) and np.isclose(m["r10"], 1.0)
    assert np.isclose(m["mrr"], (1 + 1 / 2 + 1 / 5) / 3)
    assert np.isclose(m["dcg"], (1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(6)) / 3)
    assert np.isclose(m["r1_pct"], 100 / 3)


def test_ties_are_broken_by_class_id_not_by_truth():
    scores = np.array([[0.5, 0.5, 0.5, 0.1]])
    assert ranks_from_scores(scores, [0]).tolist() == [1]
    assert ranks_from_scores(scores, [2]).tolist() == [3]
    assert ranks_from_scores(scores, [3]).tolist() == [4]


def test_max_instance_aggregation_on_toy_gallery():
    gallery = np.array([[1, 0], [0, 1], [0.6, 0.8], [-1, 0]], np.float32)
    labels = np.array([0, 0, 1, 2])
    q = np.array([[0.0, 1.0]], np.float32)
    s = gloss_max_scores(q, gallery, labels, 3)
    assert np.allclose(s[0], [1.0, 0.8, 0.0])  # class 0 uses its best instance, not the mean


def test_chunked_equals_dense_and_absent_classes():
    rng = np.random.default_rng(0)
    C = 23
    g = rng.normal(size=(300, 16)).astype(np.float32)
    gl = rng.integers(0, C - 2, 300)  # last 2 classes absent from the gallery
    q = rng.normal(size=(57, 16)).astype(np.float32)
    ql = rng.integers(0, C, 57)
    res, ranks, scores = evaluate_retrieval(q, ql, g, gl, C, query_chunk=7, gallery_chunk=29)
    assert np.array_equal(ranks, dense_reference_ranks(q, ql, g, gl, C))
    assert np.isinf(scores[:, -2:]).all() and res["gallery_class_counts"]["classes_absent"] == 2


def test_unusable_queries_stay_in_the_denominator():
    rng = np.random.default_rng(1)
    g = rng.normal(size=(40, 8)).astype(np.float32)
    gl = np.arange(40) % 10
    q = g[:6].copy()
    q[4:] = 0.0  # failed pose -> zero embedding
    usable = np.array([1, 1, 1, 1, 0, 0], bool)
    res, ranks, _ = evaluate_retrieval(q, gl[:6], g, gl, 10, query_usable=usable)
    assert res["full"]["n"] == 6 and res["failure_count"] == 2 and np.isclose(res["coverage"], 4 / 6)
    assert res["usable_only_diagnostic"]["n"] == 4 and res["usable_only_diagnostic"]["r1"] == 1.0
    assert ranks[4] == 5  # all-zero scores -> class-id order, label 4 ranks 5th
    m, _ = classification_metrics(np.random.default_rng(2).normal(size=(6, 10)), gl[:6], usable=usable)
    assert m["n"] == 6 and m["failure_count"] == 2


def test_unusable_gallery_clips_do_not_score_their_class():
    gallery = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], np.float32)  # clip 1 has no usable pose
    labels = np.array([0, 1, 2])
    q = np.array([[-1.0, -0.1]], np.float32)  # every real cosine is negative
    res, ranks, scores = evaluate_retrieval(q, [0], gallery, labels, 3, gallery_usable=[True, False, True])
    assert np.isinf(scores[0, 1]) and res["gallery_class_counts"]["n_gallery_excluded_unusable"] == 1
    _, ranks_naive, _ = evaluate_retrieval(q, [0], gallery, labels, 3)
    assert ranks_naive[0] == 3 and ranks[0] == 2  # the zero clip would otherwise outrank every true match
