import math
import unittest
from metrics import evaluate, ranked_classes, score_ranking


class MetricsTests(unittest.TestCase):
    def test_independent_rank_oracle(self):
        result = score_ranking([0, 1, 2], 2, [0, 1, 2])
        self.assertEqual(result['dcg'], .5)
        self.assertEqual(result['mrr'], 1/3)
        self.assertEqual(result['top1'], 0)

    def test_ids_joined_not_zipped(self):
        refs = [{'sample_id': 'a', 'label': 0}, {'sample_id': 'b', 'label': 1}]
        preds = [{'sample_id': 'b', 'ranking': [1, 0]}, {'sample_id': 'a', 'ranking': [0, 1]}]
        self.assertEqual(evaluate(preds, refs, [0, 1])['top1'], 1)

    def test_incomplete_duplicate_extra_and_empty_predictions_rejected(self):
        refs = [{'sample_id': 'a', 'label': 0}]
        p = {'sample_id': 'a', 'ranking': [0, 1]}
        for preds in [[], [p, p], [{'sample_id': 'b', 'ranking': [0, 1]}]]:
            with self.subTest(predictions=preds), self.assertRaises(ValueError):
                evaluate(preds, refs, [0, 1])
        with self.assertRaises(ValueError):
            evaluate([], [], [0, 1])

    def test_invalid_ranking_and_unknown_labels_rejected(self):
        for ranking, label in [([], 0), ([0, 0], 0), ([0, 2], 0), ([0, 1], 3)]:
            with self.subTest(ranking=ranking, label=label), self.assertRaises(ValueError):
                score_ranking(ranking, label, [0, 1])

    def test_ties_and_nonfinite(self):
        self.assertEqual(ranked_classes([1, 1, 2], [9, 4, 5]), [5, 4, 9])
        for scores in [[math.nan, 0], [math.inf, 0]]:
            with self.assertRaises(ValueError):
                ranked_classes(scores, [0, 1])


if __name__ == '__main__':
    unittest.main()
