"""Manual and enumerated completion oracles, not paper measurements."""
import itertools
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from completion_bounds import directional_bounds, class_bounds, strict_winner, simulate_prefix


class CompletionOracles(unittest.TestCase):
    def test_tight_two_dimensional_direction(self):
        lo, hi = directional_bounds([[1., 0.]], np.eye(2), [1., 1.], [0., 0.])
        # z=(1,-1) or (-1,1), zero mean and squared norm=2.
        np.testing.assert_allclose(lo, [-1.], atol=1e-14)
        np.testing.assert_allclose(hi, [1.], atol=1e-14)

    def test_unknown_weight_and_all_adversarial_completions(self):
        observed = np.array([0.6, 0.4])
        lo, hi = class_bounds(observed, 1, np.array([0.1, -0.2]),
                              np.array([1., 1.]), [[0], [1]])
        np.testing.assert_array_equal(lo, [0.6, 0.2])
        self.assertIsNone(strict_winner(lo, hi)[0])
        for contribution in itertools.product([0., 0.5, 1.], repeat=2):
            final = observed + np.array(contribution)
            self.assertTrue(np.all(final >= lo) and np.all(final <= hi))
        # An unobserved competitor-only contribution reverses the current winner.
        self.assertEqual(int(np.argmax(observed + [0., 1.])), 1)

    def test_separated_intervals_permit_stop_and_max_over_instances(self):
        lo, hi = class_bounds(np.array([3., 2., 0.]), 1,
                              np.array([-0.1, -0.2, -0.2]),
                              np.array([0.1, 0.2, 0.2]), [[0, 1], [2]])
        self.assertEqual(strict_winner(lo, hi)[0], 0)
        for extra in itertools.product([-0.1, 0., 0.1], repeat=3):
            score = np.array([3., 2., 0.]) + extra
            self.assertGreater(max(score[:2]), score[2])

    def test_zero_activity_and_tie_force_fallback(self):
        for x, weights in [(np.array([[1., 0.], [0., 1.]]), [0., 0.]),
                           (np.array([[1., 1.], [1., 1.]]), [1., 1.])]:
            result = simulate_prefix(x, weights, np.eye(2), [[0], [1]],
                                     np.array([-1., -1.]), np.ones(2), [0, 1])
            self.assertFalse(result['early'])
            self.assertEqual(result['used_windows'], 2)


if __name__ == '__main__':
    unittest.main()
