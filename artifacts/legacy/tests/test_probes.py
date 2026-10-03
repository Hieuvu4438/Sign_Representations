import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from signrepr.probes import LinearProbe, TemporalProbe, classification_metrics, normalized_mean


class ProbeTests(unittest.TestCase):
    def test_padding_cannot_change_readout_for_valid_tokens(self):
        torch.manual_seed(42)
        tokens = torch.randn(1, 4, 7)
        padded = torch.cat([tokens, torch.randn(1, 5, 7) * 1000], dim=1)
        for head in [LinearProbe(7, 3), TemporalProbe(7, 3, 5)]:
            head.eval()
            with torch.no_grad():
                a = head(tokens, torch.ones(1, 4))
                b = head(padded, torch.tensor([[1., 1., 1., 1., 0., 0., 0., 0., 0.]]))
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_macro_recall_is_not_pooled_accuracy(self):
        metrics = classification_metrics([0, 0, 0, 1], [0, 0, 0, 0], 2)
        self.assertEqual(metrics['top1'], .75)
        self.assertEqual(metrics['macro_recall'], .5)
        self.assertEqual(metrics['class_support'], [3, 1])

    def test_pooling_excludes_invalid_windows(self):
        pooled = normalized_mean(np.asarray([[1., 0.], [1e6, 1e6]]), np.asarray([True, False]))
        np.testing.assert_allclose(pooled, [1., 0.])

    def test_registered_temporal_capacities_match_linear_parameter_budget(self):
        reference = sum(p.numel() for p in LinearProbe(768, 200).parameters())
        for hidden in [64, 128]:
            actual = sum(p.numel() for p in TemporalProbe(768, 200, hidden).parameters())
            self.assertLess(abs(actual - reference) / reference, .001)


if __name__ == '__main__':
    unittest.main()
