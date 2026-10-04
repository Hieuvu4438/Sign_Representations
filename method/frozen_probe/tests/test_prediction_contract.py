import copy
import hashlib
import unittest

import numpy as np

from confirmatory import METHODS, RECIPES
from prediction_contract import verify_pooled_arrays, verify_source_recipes
from test_cache_baseline import fixture
from window_budget import selected_indices


class PredictionContractTests(unittest.TestCase):
    def data(self):
        cohort, _, _ = fixture()
        samples = [dict(sample_id=f'test_{c}', gloss=g, class_id=c, participant_id=f'ptest{c}', split='test',
            source_sha256=hashlib.sha256(f'test_{c}'.encode()).hexdigest()) for c,g in enumerate(cohort['classes'])]
        test = dict(classes=cohort['classes'], samples=samples)
        audit = dict(passed=True, samples=[dict(s, pts_s=[i*.04 for i in range(22)], frames=22, windows=4) for s in samples])
        schema = dict(split='test', query_ids=[s['sample_id'] for s in samples], labels=[0,1,2],
                      participant_ids=[s['participant_id'] for s in samples], class_ids=[0,1,2])
        observations = []
        for s in samples:
            for method in RECIPES:
                indices = selected_indices(method, 4, s['source_sha256'])
                observations.append(dict(sample_id=s['sample_id'], method=method, native_window_indices=indices,
                                         encoded_windows=len(indices), source_windows=4))
        return cohort, test, audit, schema, observations

    def test_complete_source_recipe_and_reject_wrong_indices(self):
        data = self.data()
        self.assertEqual(verify_source_recipes(*data)['uniform8'], 3)
        changed = copy.deepcopy(data)
        changed[-1][1]['native_window_indices'] = [1]
        with self.assertRaises(ValueError): verify_source_recipes(*changed)
        changed = copy.deepcopy(data)
        changed[-1].pop()
        with self.assertRaises(ValueError): verify_source_recipes(*changed)
        changed = copy.deepcopy(data)
        changed[-1].append(changed[-1][0])
        with self.assertRaises(ValueError): verify_source_recipes(*changed)

    def test_label_signer_timeline_schema_binding(self):
        for kind in ('label','signer','timeline'):
            changed = copy.deepcopy(self.data())
            if kind == 'label': changed[3]['labels'][0] = 1
            elif kind == 'signer': changed[2]['samples'][0]['participant_id'] = 'other'
            else: changed[2]['samples'][0]['windows'] = 5
            with self.assertRaises(ValueError): verify_source_recipes(*changed)

    def test_pooled_real_dtype_capacity_and_normalization(self):
        arrays = {m:np.eye(3,dtype=np.float64) for m in METHODS}
        verify_pooled_arrays(arrays,3,3)
        for value in (np.eye(3,dtype=np.float32), np.ones((3,3)), np.ones((2,3)), np.full((3,3),np.nan)):
            bad = dict(arrays);bad['uniform8'] = value
            with self.assertRaises(ValueError): verify_pooled_arrays(bad,3,3)
        bad = dict(arrays);bad.pop('uniform8')
        with self.assertRaises(ValueError): verify_pooled_arrays(bad,3,3)
