import unittest

from input_coverage import interval_coverage, native_coverage
from temporal import window_plan


class CoverageTests(unittest.TestCase):
    def test_overlap_union_and_boundary_gaps_oracle(self):
        result = interval_coverage([(2, 10), (4, 12), (16, 18)], 0, 20)
        self.assertEqual(result['covered'], 12)
        self.assertEqual(result['fraction'], .6)
        self.assertEqual(result['longest_uncovered_gap'], 4)
        self.assertEqual(result['leading_uncovered'], 2)
        self.assertEqual(result['trailing_uncovered'], 2)

    def test_uniform_native_coverage_and_incomplete_tail_oracle(self):
        plan = window_plan([i*.04 for i in range(31)], 16, 2)
        result = native_coverage(plan, [0, 4])
        self.assertEqual(result['frame']['covered'], 24)
        self.assertEqual(result['frame']['fraction'], .8)  # dense covers30, not31
        self.assertEqual(result['frame']['longest_uncovered_gap'], 6)
        self.assertAlmostEqual(result['time']['fraction'], .8)
        self.assertEqual(native_coverage(plan, list(range(8)))['frame']['fraction'], 1)

    def test_long_hole_nonuniform_pts_and_invalid_indices(self):
        pts = [5+i*.01 if i < 20 else 5.2+(i-20)*.03 for i in range(56)]
        plan = window_plan(pts, 16, 2)
        result = native_coverage(plan, [0, 20])
        self.assertEqual(result['frame']['covered'], 32)
        self.assertEqual(result['frame']['longest_uncovered_gap'], 24)
        self.assertNotAlmostEqual(result['frame']['fraction'], result['time']['fraction'])
        for selection in ([], [20, 0], [0, 0], [21], [-1]):
            with self.assertRaises(ValueError): native_coverage(plan, selection)
        for intervals in ([], [(-1, 2)], [(1, 1)], [(1, 11)]):
            with self.assertRaises(ValueError): interval_coverage(intervals, 0, 10)
