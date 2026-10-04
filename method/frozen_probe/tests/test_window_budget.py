import copy
import unittest
import numpy as np

from test_cache_baseline import fixture
from temporal import window_plan
from window_budget import METHODS, lookup, pooled_vector, selected_indices


class WindowBudgetTests(unittest.TestCase):
    def test_native_stride_alignment_and_odd_count(self):
        plan = window_plan([i * .04 for i in range(40)], 16, 2)
        dense = len(plan)
        chosen = selected_indices('uniform4', dense, 'a' * 64)
        self.assertEqual([plan[i]['start_frame'] for i in chosen], list(range(0, 25, 4)))
        self.assertEqual(dense, 13)
        self.assertEqual(len(chosen), 7)
        self.assertLess(dense / len(chosen), 2)
        self.assertEqual([plan[i]['start_frame'] for i in selected_indices('uniform8', dense, 'a' * 64)], [0, 8, 16, 24])

    def test_random_count_reproducibility_and_invalid_contract(self):
        for size in (1, 2, 13, 267):
            for density in (4, 8):
                chosen = selected_indices(f'random{density}_s2718', size, 'a' * 64)
                self.assertEqual(len(chosen), len(selected_indices(f'uniform{density}', size, 'a' * 64)))
                self.assertEqual(chosen, sorted(set(chosen)))
                self.assertTrue(all(0 <= i < size for i in chosen))
                self.assertEqual(chosen, selected_indices(f'random{density}_s2718', size, 'a' * 64))
        self.assertNotEqual(selected_indices('random8_s2718', 100, 'a' * 64), selected_indices('random8_s2718', 100, 'b' * 64))
        for args in [('unknown', 13, 'a' * 64), ('uniform4', 0, 'a' * 64), ('uniform4', 2.0, 'a' * 64), ('uniform4', 13, 'bad')]:
            with self.assertRaises(ValueError): selected_indices(*args)

    def test_selection_before_pooling_and_zero_activity_fallback(self):
        x = np.array([[1, 0], [0, 10], [1, 0]], dtype=np.float32)
        a = np.zeros((3, 2), dtype=np.float32)
        np.testing.assert_array_equal(pooled_vector(x, a, 'uniform4', [0, 2]), [1, 0])
        with self.assertRaises(ValueError): pooled_vector(x, a, 'uniform4', [2, 0])
        with self.assertRaises(ValueError): pooled_vector(x, a, 'uniform4', [0, 0])

    def test_gallery_only_complete_class_lookup(self):
        cohort, audit, identity = fixture()
        ids = [s['sample_id'] for s in cohort['samples']]
        x = np.array([np.eye(3)[s['class_id']] for s in cohort['samples']])
        matrices = {m: x.copy() for m in METHODS}
        queries, results, ranks = lookup(cohort, ids, matrices, [1701], 1)
        self.assertEqual(len(queries), 3)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(r['metrics']['dcg'] == 1 and r['metrics']['top1'] == 1 for r in results))
        for rank in ranks.values():
            self.assertEqual(rank.shape, (3, 3))
            self.assertEqual(rank[:, 0].tolist(), [0, 1, 2])
        altered = copy.deepcopy(cohort)
        altered['supports']['1701']['1']['A'][0] = 'val_0_0'
        with self.assertRaises(ValueError): lookup(altered, ids, matrices, [1701], 1)
