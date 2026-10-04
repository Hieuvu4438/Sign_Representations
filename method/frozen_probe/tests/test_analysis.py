import copy
import unittest
import numpy as np

from analysis import analyze, participant_bootstrap, METHODS, SEEDS, BUDGETS
from metrics import evaluate


def fixture():
    ids = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5']
    schema = dict(query_ids=ids, labels=[0] * 6, participant_ids=['P1', 'P2', 'P2', 'P3', 'P3', 'P3'],
                  class_ids=[0, 1, 2])
    refs = [dict(sample_id=key, label=0) for key in ids]
    results, ranks = [], {}
    for method in METHODS:
        for seed in SEEDS:
            for budget in BUDGETS:
                name = f'{method}_s{seed}_n{budget}'
                correct = method == 'ordered_ridge' and budget in (2, 4)
                ranks[name] = np.tile([0, 1, 2] if correct else [1, 0, 2], (6, 1)).astype(np.int16)
                predictions = [dict(sample_id=key, ranking=row.tolist()) for key, row in zip(ids, ranks[name])]
                results.append(dict(name=name, metrics=evaluate(predictions, refs, [0, 1, 2])))
    return schema, results, ranks


class AnalysisTests(unittest.TestCase):
    def test_primary_all_seeds_budgets_and_constant_bootstrap_oracle(self):
        schema, results, ranks = fixture()
        report = analyze(schema, results, ranks)
        self.assertEqual(report['strongest_baseline'], 'activity_nearest')
        self.assertEqual(report['ordered_minus_baseline_pp'], 100)
        self.assertEqual(report['ordered_minus_shuffled_pp'], 100)
        self.assertEqual(report['conditional_participant_bootstrap']['interval95_pp'], [100, 100])
        self.assertEqual(report['label_curves']['ordered_ridge']['1']['mean_percent'], 0)
        self.assertTrue(report['practical_screen_signal'])
        for r in results:
            if r['name'].startswith('ordered_ridge_s1703_'):
                ranks[r['name']] = np.tile([1, 0, 2], (6, 1)).astype(np.int16)
                r['metrics'] = dict(dcg=1 / np.log2(3), top1=0., top5=1., top10=1., top20=1., mrr=.5)
        report = analyze(schema, results, ranks)
        self.assertAlmostEqual(report['primary_percent']['ordered_ridge'], 200 / 3)
        self.assertEqual(report['per_support_seed_primary_difference_pp']['1703'], 0)
        self.assertTrue(report['practical_screen_signal'])

    def test_participant_micro_weighting_independent_literal_oracle(self):
        result = participant_bootstrap([1, 0, 0], ['A', 'B', 'B'], reps=64, seed=17)
        draws = np.random.default_rng(17).integers(0, 2, size=(64, 2))
        values = []
        for draw in draws:
            repeated = []
            for group in draw.tolist(): repeated.extend([1] if group == 0 else [0, 0])
            values.append(sum(repeated) / len(repeated) * 100)
        self.assertAlmostEqual(result['estimate_pp'], 100 / 3)
        np.testing.assert_allclose(result['interval95_pp'], np.quantile(values, [.025, .975]))
        with self.assertRaises(ValueError): participant_bootstrap([1], ['A'])

    def test_corrupt_predictions_metrics_and_coverage_rejected(self):
        schema, results, ranks = fixture()
        for kind in ('metric', 'missing', 'ranking', 'ids'):
            s, r, p = copy.deepcopy(schema), copy.deepcopy(results), copy.deepcopy(ranks)
            if kind == 'metric': r[0]['metrics']['top1'] = .9
            elif kind == 'missing': p.pop(next(iter(p)))
            elif kind == 'ranking': p[next(iter(p))][0] = [0, 0, 2]
            else: s['query_ids'][1] = s['query_ids'][0]
            with self.assertRaises(ValueError): analyze(s, r, p)
