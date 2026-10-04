"""Independent diagnostic oracles including lexeme confounds and missing people."""
from collections import Counter
from pathlib import Path
import sys
import unittest
import numpy as np
from sklearn.metrics import balanced_accuracy_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from participant_probe import centroid_predictions, gloss_folds, permute_within_gloss, score_participants


class ParticipantProbeTests(unittest.TestCase):
    def test_denominators_and_public_reference(self):
        truth = ['A']*9 + ['B']
        cases = [(truth, 1., 1.), (['B']*9+['A'], 0., 0.), (['A']*10, .9, .5)]
        for predicted, micro, balanced in cases:
            result = score_participants(truth, predicted)
            self.assertEqual(result['micro'], micro)
            self.assertEqual(result['balanced'], balanced)
            self.assertEqual(result['balanced'], balanced_accuracy_score(truth, predicted))
        with self.assertRaises(ValueError): score_participants([], [])
        with self.assertRaises(ValueError): score_participants(truth, ['A'])

    def test_identity_transfer_and_class_only_confound(self):
        labels = ['A', 'B', 'A', 'B']
        result = centroid_predictions([[1,0], [0,1], [1,0], [0,1]], labels, [0,1], [2,3])
        self.assertEqual(result['scores']['balanced'], 1.)
        # Only lexeme changes across halves; no participant information.
        result = centroid_predictions([[1,0], [1,0], [0,1], [0,1]], labels, [0,1], [2,3])
        self.assertEqual(result['predictions'], ['A', 'A'])
        self.assertEqual(result['scores']['balanced'], .5)

    def test_absent_and_degenerate_not_dropped(self):
        result = centroid_predictions([[1,0], [1,0], [0,1]], ['A','A','B'], [0], [1,2])
        self.assertEqual(result['fit_coverage'], .5)
        self.assertEqual(result['scores']['balanced'], .5)
        self.assertEqual(result['scores']['by_participant']['B']['recall'], 0.)
        result = centroid_predictions([[1,0], [-1,0], [0,1], [1,0]], ['A','A','B','A'], [0,1,2], [3])
        self.assertEqual(result['degenerate_participants'], ['A'])
        self.assertEqual(result['candidate_coverage'], 0.)
        self.assertEqual(result['scores']['balanced'], 0.)

    def test_partition_and_permutation(self):
        glosses = ['c','c','d','d','e','e']
        labels = ['A','B','B','C','A','C']
        permuted = permute_within_gloss(labels, glosses, np.random.default_rng(1701))
        for g in set(glosses):
            self.assertEqual(Counter(l for l,k in zip(labels,glosses) if k==g),
                             Counter(l for l,k in zip(permuted,glosses) if k==g))
        a,b = gloss_folds(['a','b','c','d'],1701)
        self.assertFalse(set(a)&set(b))
        self.assertEqual(set(a+b), {'a','b','c','d'})
        self.assertEqual((a,b),gloss_folds(['d','c','b','a'],1701))
        with self.assertRaises(ValueError): gloss_folds(['a','a'],1701)
        with self.assertRaises(ValueError): centroid_predictions([[1],[1]], ['A','B'], [0], [0,1])


if __name__ == '__main__': unittest.main()
