import copy
import unittest

import numpy as np

from factorial import pair_key, summarize_pairs


class FactorialTests(unittest.TestCase):
    def fixture(self):
        schema = dict(query_ids=['q0', 'q1'], labels=[0, 1], participant_ids=['p0', 'p1'],
                      class_ids=[0, 1, 2], split='test')
        methods, seeds = ['full_activity', 'uniform8'], [7, 8]
        matrices = [[[0, 1, 2], [1, 0, 2]], [[1, 0, 2], [0, 1, 2]],
                    [[1, 0, 2], [1, 0, 2]], [[1, 2, 0], [0, 2, 1]]]
        rankings = {}
        for (q, g), rows in zip([(methods[0], methods[0]), (methods[0], methods[1]),
                                (methods[1], methods[0]), (methods[1], methods[1])], matrices):
            for seed in seeds:
                rankings[pair_key(q, g, seed)] = np.array(rows, dtype=np.int16)
        return schema, rankings, methods, seeds

    def test_hand_computed_query_gallery_axes_and_interaction(self):
        result = summarize_pairs(*self.fixture())
        d = 1/np.log2(3)
        contrasts = result['contrasts']['uniform8']
        expected = dict(query_only=(d-1)/2, gallery_only=d-1, joint=-.5,
                        interaction=.5-(d+1)/2-d+1)
        for name, value in expected.items():
            self.assertAlmostEqual(contrasts['mean']['dcg'][name], value)
        self.assertEqual(contrasts['mean']['top1']['query_only'], -.5)
        self.assertEqual(contrasts['mean']['top1']['gallery_only'], -1)
        self.assertEqual(contrasts['mean']['top1']['interaction'], .5)
        self.assertAlmostEqual(contrasts['by_participant']['p0']['dcg']['query_only'], d-1)
        self.assertEqual(contrasts['by_participant']['p1']['dcg']['query_only'], 0)
        self.assertEqual(result['ranking_matrices'], 8)
        self.assertEqual(result['participant_counts'], dict(p0=1, p1=1))

    def test_complete_cells_and_full_rank_permutations_required(self):
        for bad in ('missing', 'unknown', 'duplicate_class', 'float', 'short'):
            data = copy.deepcopy(self.fixture())
            key = next(iter(data[1]))
            if bad == 'missing': data[1].pop(key)
            elif bad == 'unknown': data[1]['unregistered'] = data[1][key]
            elif bad == 'duplicate_class': data[1][key][0] = [0, 0, 1]
            elif bad == 'float': data[1][key] = data[1][key].astype(float)
            else: data[1][key] = data[1][key][:1]
            with self.assertRaises(ValueError): summarize_pairs(*data)

    def test_reject_identity_signer_split_and_method_coverage(self):
        for kind in ('id', 'participant', 'split', 'methods', 'seeds'):
            data = list(copy.deepcopy(self.fixture()))
            if kind == 'id': data[0]['query_ids'] = ['q0', 'q0']
            elif kind == 'participant': data[0]['participant_ids'].pop()
            elif kind == 'split': data[0]['split'] = 'val'
            elif kind == 'methods': data[2] = ['uniform8']
            else: data[3] = [7, 7]
            with self.assertRaises(ValueError): summarize_pairs(*data)
