import ast
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from signrep_baseline.retrieval import pool_windows, dictionary_retrieval, ranking_metrics
from signrep_baseline.recognition import class_distribution, recognition_loss

ROOT = Path(__file__).resolve().parents[1]


def test_equation12_raw_activity_pool_and_zero_denominator():
    features = torch.tensor([[1., 0.], [0., 1.]])
    activity = torch.tensor([[.2, .5], [1., .3]])
    expected = torch.tensor([1., 2.]) / math.sqrt(5)
    torch.testing.assert_close(pool_windows(features, activity), expected)
    # Raw negative predictions are preserved instead of being silently clamped.
    actual = pool_windows(features, torch.tensor([[-2., -1.], [2., 1.]]))
    torch.testing.assert_close(actual, torch.tensor([-1., 2.]) / math.sqrt(5))
    with pytest.raises(ValueError, match="denominator"):
        pool_windows(features, torch.zeros(2, 2))


def test_nearest_video_and_distinct_class_metrics_against_author():
    gallery = torch.tensor([[1., 0.], [.99, .01], [0., 1.], [-1., 0.]])
    query = torch.tensor([[1., 0.], [0., 1.]])
    metrics, detail = dictionary_retrieval(gallery, [0, 0, 1, 2], query, [1, 1],
                                          ['a', 'b', 'c', 'd'], ['x', 'y'])
    assert detail['nearest_video_indices'].tolist() == [0, 2]
    assert metrics['per_instance_top1'] == 50 and metrics['MRR'] == .75
    assert abs(metrics['DCG'] - ((1 / math.log2(3) + 1) / 2)) < 1e-7
    # Execute only the exact author metric function; no imports/model/data side effects.
    tree = ast.parse((ROOT / 'vendor/asl_citizen_metrics/I3D/aslcitizen_testing.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eval_metrics')
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {'np': np, 'math': math}
    exec(compile(module, '<pinned author metric>', 'exec'), namespace)
    expected = [namespace['eval_metrics'](row.numpy(), label)[1] for row, label in zip(detail['class_ranking'], [1, 1])]
    assert abs(metrics['DCG'] - np.mean([row[0] for row in expected])) < 1e-7
    assert metrics['MRR'] == np.mean([row[-1] for row in expected])
    with pytest.raises(ValueError, match="disjoint"):
        dictionary_retrieval(gallery, [0, 0, 1, 2], query, [1, 1], ['a', 'b', 'c', 'd'], ['a', 'y'])


def test_class_ranking_ties_match_first_best_gallery_video():
    # Class9 occurs first in gallery, but its best video occurs after class0's best.
    metrics, detail = dictionary_retrieval([[0., 1.], [1., 0.], [1., 0.]], [9, 0, 9],
                                          [[1., 0.]], [0], ['a', 'b', 'c'], ['q'])
    assert detail['class_ranking'].tolist() == [[0, 9]]
    assert metrics['per_instance_top1'] == 100 and metrics['MRR'] == metrics['DCG'] == 1


def test_class_cosines_equal_all_pairs_and_temperature_constraint():
    features = torch.tensor([[1., .01], [1., .02], [1., -.01], [1., -.02], [1., .0], [1., .03]])
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    table, temperatures, similarities = class_distribution(features, labels, 3)
    normalized = features.double() / features.double().norm(dim=-1, keepdim=True)
    all_pairs = normalized @ normalized.T
    for i in range(3):
        for j in range(3):
            torch.testing.assert_close(similarities[i, j], all_pairs[labels == i][:, labels == j].mean().float())
    assert (table.max(1).values < .5).all()
    grid = torch.arange(1, 101, dtype=torch.float64) / 1000
    for i in range(3):
        probabilities = (all_pairs[labels == i].reshape(2, 3, 2).mean((0, 2))[None] / grid[:, None]).softmax(1)
        maxima = probabilities.max(1).values
        assert abs(float(table[i].max()) - float(maxima[maxima < .5].max())) < 1e-6
    with pytest.raises(ValueError, match="No temperature"):
        class_distribution(torch.eye(3), [0, 1, 2], 3)


def test_equation13_kl_direction_diagonal_weight_and_smoothing():
    logits = torch.tensor([[math.log(.5), math.log(.3), math.log(.2)]], requires_grad=True)
    phi = torch.tensor([[.4, .35, .25], [.3, .4, .3], [.2, .35, .45]])
    loss, terms = recognition_loss(logits, torch.tensor([0]), phi)
    # Independent scalar arithmetic, including CE uniform smoothing over all classes.
    p = [.5, .3, .2]
    target = [.4, .35, .25]
    expected_ce = -.9 * math.log(.5) - (.1 / 3) * sum(math.log(v) for v in p)
    expected_kl = .2 * .4 * sum(a * math.log(a / b) for a, b in zip(target, p))
    assert abs(float(terms['cross_entropy'].detach()) - expected_ce) < 1e-6
    assert abs(float(terms['class_distribution'].detach()) - expected_kl) < 1e-6
    loss.backward()
    assert torch.isfinite(logits.grad).all()
