import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from signrepr.statistics import align_predictions, grouped_bootstrap, weighted_metrics


class StatisticsTests(unittest.TestCase):
    def test_paired_identical_models_have_exact_zero_delta_interval(self):
        correct = np.asarray([[1, 0, 1, 1], [0, 1, 1, 0]])
        result = grouped_bootstrap([0, 0, 1, 1], ['a', 'a', 'b', 'c'], correct, correct, 2000, 42)
        for metric in result['metrics'].values():
            self.assertEqual(metric['delta'], 0)
            self.assertEqual(metric['delta_ci95'], [0, 0])
        self.assertEqual(result['groups'], 3)
        self.assertGreater(result['resamples_with_missing_classes'], 0)

    def test_whole_recordings_are_resampled_as_units(self):
        # One recording is entirely right, the other entirely wrong. With two
        # group draws, every replicate must be 0, .5 or 1, never .25 or .75.
        result = grouped_bootstrap([0] * 100, ['a'] * 50 + ['b'] * 50, [[1] * 50 + [0] * 50], resamples=2000)
        self.assertEqual(result['metrics']['top1']['ci95'], [0, 1])
        self.assertEqual(result['metrics']['top1']['point'], .5)

    def test_weighted_macro_and_pooled_metrics_differ(self):
        result, classes = weighted_metrics(np.asarray([0, 0, 1]), np.asarray([[1, 1, 0]]), np.asarray([2, 1, 1]))
        self.assertEqual(classes, 2)
        self.assertEqual(result['macro_recall'][0], .5)
        self.assertEqual(result['top1'][0], .75)

    def test_pairing_rejects_coverage_or_gold_mismatch(self):
        row = {'gold': '0', 'split': 'test', 'recording_group': 'a', 'correct': '1'}
        with self.assertRaisesRegex(ValueError, 'coverage'):
            align_predictions([{'one': row}, {'two': row}])
        with self.assertRaisesRegex(ValueError, 'identity'):
            align_predictions([{'one': row}, {'one': dict(row, gold='1')}])

    def test_too_few_groups_or_resamples_are_unavailable(self):
        with self.assertRaises(ValueError):
            grouped_bootstrap([0, 0], ['a', 'a'], [[1, 0]])
        with self.assertRaises(ValueError):
            grouped_bootstrap([0, 0], ['a', 'b'], [[1, 0]], resamples=100)


if __name__ == '__main__':
    unittest.main()
