import unittest

import numpy as np
import torch

from method.signrep_spectral.evaluation.retrieval import class_scores, evaluate_retrieval, metrics_from_ranks, true_class_rank
from method.signrep_spectral.evaluation.statistics import paired_bootstrap


class TestRetrieval(unittest.TestCase):
    def test_manual_rank_and_max_aggregation(self):
        # classes A=0, B=1, C=2; query true class B; class scores A=.9, B=.8, C=.1
        sims = torch.tensor([[0.9, 0.2, 0.8, 0.5, 0.1]])
        labels = torch.tensor([0, 0, 1, 1, 2])
        g = torch.eye(5)
        cs = class_scores(sims, g, labels, 3)
        self.assertTrue(torch.allclose(cs, torch.tensor([[0.9, 0.8, 0.1]])))
        r = true_class_rank(cs, torch.tensor([1]))
        self.assertEqual(r.item(), 2)
        m = metrics_from_ranks(r.numpy(), np.array([1]))
        self.assertAlmostEqual(m["mrr"], 0.5)
        self.assertEqual(m["recall@1"], 0.0)
        self.assertEqual(m["recall@5"], 1.0)
        # extra support for class A does not add a candidate
        sims2 = torch.tensor([[0.9, 0.2, 0.85, 0.8, 0.5, 0.1]])
        labels2 = torch.tensor([0, 0, 0, 1, 1, 2])
        self.assertEqual(true_class_rank(class_scores(sims2, torch.eye(6), labels2, 3), torch.tensor([1])).item(), 2)

    def test_ties_by_class_id(self):
        cs = torch.tensor([[0.5, 0.5, 0.5]])
        self.assertEqual(true_class_rank(cs, torch.tensor([0])).item(), 1)
        self.assertEqual(true_class_rank(cs, torch.tensor([2])).item(), 3)

    def test_mrr_micro_macro(self):
        ranks = np.array([1, 2, 1])
        labels = np.array([0, 0, 1])
        m = metrics_from_ranks(ranks, labels)
        self.assertAlmostEqual(m["mrr"], (1 + 0.5 + 1) / 3)
        self.assertAlmostEqual(m["macro_mrr"], (0.75 + 1) / 2)
        m2 = metrics_from_ranks(np.array([1, 2]), np.array([0, 1]))
        self.assertAlmostEqual(m2["mrr"], 0.75)

    def test_policy_errors(self):
        x = np.eye(3, dtype=np.float32)
        gal = {"sample_id": np.array(["a", "b", "c"]), "label": np.array([0, 1, 2]), "x": x}
        qry = {"sample_id": np.array(["a"]), "label": np.array([0]), "x": x[:1]}
        with self.assertRaises(ValueError):
            evaluate_retrieval(gal, qry, 3, "cpu")
        qry = {"sample_id": np.array(["q"]), "label": np.array([3]), "x": x[:1]}
        with self.assertRaises(ValueError):
            evaluate_retrieval(gal, qry, 4, "cpu")
        qry = {"sample_id": np.array(["q"]), "label": np.array([0]), "x": np.full((1, 3), np.nan, np.float32)}
        with self.assertRaises(ValueError):
            evaluate_retrieval(gal, qry, 3, "cpu")


class TestStats(unittest.TestCase):
    def test_paired_bootstrap(self):
        a = {f"q{i}": (0.5, f"s{i % 3}") for i in range(30)}
        b = {f"q{i}": (0.6, f"s{i % 3}") for i in range(30)}
        r = paired_bootstrap(a, b, n_boot=200)
        self.assertAlmostEqual(r["delta"], 0.1)
        self.assertEqual(r["num_signers"], 3)
        with self.assertRaises(ValueError):
            paired_bootstrap(a, {"x": (1.0, "s")})


if __name__ == "__main__":
    unittest.main()
