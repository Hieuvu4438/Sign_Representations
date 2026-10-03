import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from signrepr.semantic_pairs import candidate_order, matched_credit, summarize, validate_pairs, wrong_video_mapping
from run_semantic_pairs import prepare


class SemanticPairsTests(unittest.TestCase):
    def rows(self):
        return [{'pair_id': p, 'utterance_id': u, 'recording_group': g, 'phenomenon': 'synthetic',
                 'matched_text': 'candidate A', 'mismatched_text': 'candidate B', 'gold_provenance': 'synthetic unit test only',
                 'review_status': 'EXPERT_VERIFIED_CANONICAL_VIDEO_AND_SEMANTIC_PAIR'}
                for p, u, g in [('p1', 'u1', 'g1'), ('p2', 'u1', 'g1'), ('p3', 'u2', 'g2')]]

    def test_candidate_swap_and_exact_ties_do_not_change_credit(self):
        for pair_id in ['p1', 'p2', 'other']:
            order = candidate_order(pair_id, 42)
            values = np.array([1., 3.])[order]
            self.assertEqual(matched_credit(values, order.index(0)), 1)
            self.assertEqual(order, candidate_order(pair_id, 42))
        self.assertEqual(matched_credit([2., 2.], 0), .5)
        self.assertEqual(matched_credit([2., 2.], 1), .5)
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            matched_credit([float('nan'), 0], 0)

    def test_duplicate_pairs_do_not_overweight_utterance(self):
        result = summarize(self.rows(), {'video': [1, 0, 1], 'wrong_video': [0, 0, 0]})
        self.assertEqual(result['conditions']['video']['macro_accuracy'], .75)
        self.assertEqual(result['video_minus_control']['wrong_video']['delta'], .75)
        self.assertEqual(result['utterances'], 2)
        self.assertEqual(result['pair_rows'], 3)

    def test_wrong_video_uses_different_recording_and_unreviewed_gold_is_rejected(self):
        rows = self.rows(); mapping = wrong_video_mapping(rows, 42)
        self.assertEqual(mapping, {'u1': 'u2', 'u2': 'u1'})
        rows[0]['review_status'] = 'METADATA_RESOLVED_REVIEW_PENDING'
        with self.assertRaisesRegex(ValueError, 'expert verification'):
            validate_pairs(rows)
        with self.assertRaisesRegex(ValueError, 'blocked or unlocked'):
            prepare({'status': 'BLOCKED_ACCESS'})


if __name__ == '__main__':
    unittest.main()
