import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from signrepr.calibration import categorical_scores, select_temperature, paired_mean_bootstrap


class CalibrationTests(unittest.TestCase):
    def test_stable_extreme_logits_and_perfect_confidence_last_bin(self):
        scores = categorical_scores([[1000., 0., 0.], [0., 1000., 0.]], np.array([0, 0]))
        np.testing.assert_allclose(scores['nll'], [0., 1000.])
        np.testing.assert_allclose(scores['brier'], [0., 2.])
        self.assertEqual(scores['metrics']['reliability_bins'][-1]['count'], 2)
        self.assertEqual(scores['metrics']['ece'], .5)
        self.assertEqual(sum(b['count'] for b in scores['metrics']['reliability_bins']), 2)

    def test_temperature_uses_validation_only_and_preserves_argmax(self):
        logits, gold = np.array([[8., 0.], [8., 0.], [0., 8.]]), np.array([0, 1, 1])
        t, candidates = select_temperature(logits, gold, [1., 4., 8.])
        self.assertEqual(t, 8.)
        np.testing.assert_array_equal(categorical_scores(logits, gold, t)['prediction'], logits.argmax(1))
        self.assertEqual(len(candidates), 3)
        with self.assertRaises(ValueError):
            categorical_scores(logits, gold, 0.)

    def test_paired_group_bootstrap_constant_delta(self):
        x = np.array([[1., 2., 3.], [2., 3., 4.]])
        result = paired_mean_bootstrap(['a', 'a', 'b'], x, x+2.)
        np.testing.assert_allclose(result['delta_ci95'], [-2., -2.])
        self.assertEqual(result['delta'], -2.)
        self.assertEqual(result['groups'], 2)


if __name__ == '__main__':
    unittest.main()
