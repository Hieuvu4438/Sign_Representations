import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from signrepr.interval_readouts import IntervalReadout, interval_iou, macro_values


class IntervalTests(unittest.TestCase):
    def test_controls_match_active_parameter_budget_and_ignore_padding(self):
        torch.manual_seed(42)
        x, mask = torch.randn(2, 5, 7), torch.ones(2, 5, dtype=torch.bool)
        pos = torch.linspace(0, 1, 5).expand(2, -1)
        counts = []
        for temporal in [False, True]:
            model = IntervalReadout(7, 3, temporal)
            counts.append(sum(p.numel() for p in model.parameters()))
            a = model(x, mask, pos)
            b = model(torch.cat([x, torch.randn(2, 4, 7) * 1000], dim=1),
                      torch.cat([mask, torch.zeros(2, 4, dtype=torch.bool)], dim=1),
                      torch.cat([pos, torch.full((2, 4), 999.)], dim=1))
            for i in [0, 1]:
                torch.testing.assert_close(a[i], b[i])
            if temporal:
                torch.testing.assert_close(a[2].sum(1), torch.ones(2, 6))
            (a[0].square().sum() + a[1].square().sum()).backward()
            for parameter in model.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertTrue((parameter.grad != 0).all())
        self.assertEqual(counts[0], counts[1])

    def test_iou_uses_original_clock_and_rejects_invalid_gold(self):
        np.testing.assert_allclose(interval_iou([[1., 3.], [2., 2.], [8., 9.]],
                                              [[2., 4.], [1., 3.], [1., 3.]]), [1/3, 0, 0])
        with self.assertRaises(ValueError):
            interval_iou([[0., 1.]], [[2., 2.]])

    def test_macro_handles_class_imbalance_and_seed_mean(self):
        np.testing.assert_allclose(macro_values([0, 0, 0, 1], [[1, 1, 1, 0], [0, 0, 0, 1]]), [.5, .5])


if __name__ == '__main__':
    unittest.main()
