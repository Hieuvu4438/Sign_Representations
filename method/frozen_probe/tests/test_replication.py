import copy
import unittest

import numpy as np

from confirmatory import METHODS, SEEDS, build_galleries, search
from replication import quality_report, validate_cohort
from test_cache_baseline import fixture


class ReplicationTests(unittest.TestCase):
    def test_one_example_galleries_and_known_cosine_rankings(self):
        cohort, audit, identity = fixture()
        expected = dict(classes=3, train=6, val=3, windows=36)
        validate_cohort(cohort, audit, expected, identity['contract'])
        class Reader:
            def load(self, key):
                label = next(s['class_id'] for s in cohort['samples'] if s['sample_id'] == key)
                return (np.tile(np.eye(3, dtype=np.float32)[label], (4, 1)),
                        np.ones((4, 2), dtype=np.float32), np.arange(4, dtype=np.float64))
        galleries, metadata = build_galleries(Reader(), cohort, budget=1)
        self.assertEqual(metadata['unique_source_count'], 3)
        self.assertEqual(metadata['window_counts']['uniform8'], 3)
        self.assertEqual(metadata['window_counts']['full_activity'], 12)
        for method in METHODS:
            for seed, ranking in search(np.array([0., 1., 0.]), galleries[method], 3).items():
                self.assertEqual(ranking, [1, 0, 2])
        bad = copy.deepcopy(cohort)
        bad['supports']['1701']['1']['A'] = ['val_0_0']
        with self.assertRaises(ValueError):
            validate_cohort(bad, audit, expected, identity['contract'])
        bad = copy.deepcopy(cohort)
        bad['samples'][-1]['participant_id'] = 'P_train'
        with self.assertRaises(ValueError):
            validate_cohort(bad, audit, expected, identity['contract'])
        with self.assertRaises(ValueError):
            validate_cohort(cohort, audit, expected | {'windows': 35}, identity['contract'])

    def data(self, count=20):
        schema = dict(query_ids=[str(i) for i in range(count)], labels=[0]*count,
                      participant_ids=['A']*(count//2)+['B']*(count-count//2),
                      class_ids=[0, 1], split='val', gallery_budget=1, support_seeds=list(SEEDS))
        rankings = {f'{m}_s{s}': np.tile([0, 1], (count, 1)) for m in METHODS for s in SEEDS}
        return schema, rankings

    def test_dcg_oracle_and_per_gallery_failure_despite_mean_pass(self):
        schema, rankings = self.data()
        report = quality_report(schema, rankings)
        self.assertTrue(report['quality_screen'])
        self.assertEqual(report['mean_metrics']['uniform8']['dcg'], 1.)
        rankings['uniform8_s1701'][:2] = [1, 0]
        report = quality_report(schema, rankings)
        loss = (1 - 1/np.log2(3)) / 10
        self.assertAlmostEqual(report['per_gallery_dcg_loss']['1701'], loss)
        self.assertAlmostEqual(report['uniform_minus_dense_dcg'], -loss/3)
        self.assertGreater(report['uniform_minus_dense_dcg'], -.02)
        self.assertFalse(report['quality_screen'])

    def test_group_uncertainty_failure_despite_mean_and_gallery_pass(self):
        schema, rankings = self.data(40)
        for seed in SEEDS:
            rankings[f'uniform8_s{seed}'][:2] = [1, 0]
        report = quality_report(schema, rankings)
        self.assertGreater(report['uniform_minus_dense_dcg'], -.02)
        self.assertTrue(all(v <= .02 for v in report['per_gallery_dcg_loss'].values()))
        self.assertLess(report['conditional_participant_bootstrap']['interval95_dcg'][0], -.02)
        self.assertFalse(report['quality_screen'])

    def test_dev_schema_and_full_permutation_rejections(self):
        schema, ranks = self.data()
        with self.assertRaises(ValueError):
            quality_report(schema | {'split': 'test'}, ranks)
        with self.assertRaises(ValueError):
            quality_report(schema | {'gallery_budget': 8}, ranks)
        bad = copy.deepcopy(ranks)
        bad['uniform8_s1701'][0] = [0, 0]
        with self.assertRaises(ValueError):
            quality_report(schema, bad)
        bad = copy.deepcopy(ranks)
        del bad['random8_s31415_s1703']
        with self.assertRaises(ValueError):
            quality_report(schema, bad)


if __name__ == '__main__':
    unittest.main()
