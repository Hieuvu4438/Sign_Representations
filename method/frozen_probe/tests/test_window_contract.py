import unittest
from temporal import window_plan


class WindowContractTests(unittest.TestCase):
    def test_counts_and_tail(self):
        for n, expected in ((16, 1), (17, 1), (20, 3), (83, 34)):
            plan = window_plan([i/30 for i in range(n)])
            self.assertEqual(len(plan), expected)
            self.assertEqual(plan[-1]['stop_frame_exclusive'], 16+2*(expected-1))
            self.assertEqual(plan[0]['valid_mask'], [True]*16)

    def test_irregular_pts_and_end_oracle(self):
        plan = window_plan([0., .04, .10, .15, .23], length=3, stride=2)
        self.assertAlmostEqual(plan[0]['end_s_exclusive'], .15)
        self.assertAlmostEqual(plan[0]['center_s'], .05)
        self.assertAlmostEqual(plan[1]['end_s_exclusive'], .285)
        self.assertTrue(plan[1]['end_boundary_estimated'])
        self.assertFalse(plan[0]['end_boundary_estimated'])

    def test_reject_invalid_timeline_and_short_clip(self):
        for pts in ([], [0.]*16, [float('nan')]+list(range(1,16)), list(range(15))):
            with self.assertRaises(ValueError):
                window_plan(pts)


if __name__ == '__main__':
    unittest.main()
