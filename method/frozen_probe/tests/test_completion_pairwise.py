"""Independent shared-component and reversal oracles."""
import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from completion_pairwise import pair_lower_bounds, simulate_paired_prefix


class PairOracles(unittest.TestCase):
    def test_common_component_cancels(self):
        g = np.array([[1., .1], [1., -.1]])
        bound = pair_lower_bounds(g, np.eye(2), [100., 1.], [0., 2.])
        # Shared 100*z0 cancels; difference=.2*(z1+2), z1 in[-1,1].
        self.assertAlmostEqual(bound[0, 1], .2)
        self.assertAlmostEqual(bound[1, 0], -.6)
        for z1 in (-1., 0., 1.):
            difference = .2*(z1+2.)
            self.assertGreaterEqual(difference+1e-14, bound[0, 1])

    def test_stop_with_enumerated_completions(self):
        g = np.eye(2)
        bound = np.array([[0., -.1], [-.1, 0.]])
        result = simulate_paired_prefix([[2., 0.], [0., 0.]], [1., 1.], g,
                                        [[0], [1]], None, None, [0, 1], bound)
        self.assertTrue(result['early'])
        for remaining in itertools.product([-.05, 0., .05], repeat=2):
            self.assertGreater(2.+remaining[0], remaining[1])

    def test_adversarial_reversal_zero_activity_and_ties(self):
        g = np.eye(2)
        bound = np.array([[0., -2.], [-2., 0.]])
        for x, weights in [([[.1, 0.], [0., 1.]], [1., 1.]),
                           ([[1., 0.], [0., 1.]], [0., 0.]),
                           ([[1., 1.], [1., 1.]], [1., 1.])]:
            result = simulate_paired_prefix(x, weights, g, [[0], [1]],
                                            None, None, [0, 1], bound)
            self.assertFalse(result['early'])


if __name__ == '__main__':
    unittest.main()
