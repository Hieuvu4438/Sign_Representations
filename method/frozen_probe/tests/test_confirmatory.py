import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from confirmatory import METHODS, RECIPES, SEEDS, activity_proxy, analyze, check_freeze, cost_summary, query_map, search, verify_rankings
from test_cache_baseline import fixture


class ConfirmatoryTests(unittest.TestCase):
    def test_erratum_cannot_reopen_scientific_config(self):
        import confirmatory
        from cache import file_sha
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            config = repo / 'config.json'
            config.write_text('{}')
            original = repo / 'original.json'
            original.write_text(json.dumps(dict(verdict='FROZEN_BEFORE_SELECTED_TEST_ACCESS',
                config_hashes={'config.json': file_sha(config)}, file_hashes={})))
            gate = repo / 'gate.json'
            record = dict(verdict='VALIDATED_TECHNICAL_ERRATUM_BEFORE_SCORING', original_freeze='original.json',
                original_freeze_sha256=file_sha(original), scientific_config_changes=[],
                config_hashes={'config.json': file_sha(config)}, file_hashes={})
            gate.write_text(json.dumps(record))
            with patch.object(confirmatory, 'REPO', repo):
                check_freeze(gate, config)
                record['scientific_config_changes'] = ['dcg_loss_margin']
                gate.write_text(json.dumps(record))
                with self.assertRaises(ValueError): check_freeze(gate, config)

    def test_test_split_is_explicit_and_never_a_gallery(self):
        cohort, _, _ = fixture()
        test = dict(classes=cohort['classes'], samples=[dict(sample_id=f'test_{c}', split='test', gloss=g,
            class_id=c, participant_id='P_test', source_sha256=str(c + 5) * 64) for c, g in enumerate(cohort['classes'])])
        self.assertEqual(len(query_map(cohort, test)), 3)
        for field, value in (('split', 'val'), ('participant_id', 'P_train'), ('gloss', 'B'),
                             ('sample_id', cohort['samples'][0]['sample_id']),
                             ('source_sha256', cohort['samples'][0]['source_sha256'])):
            changed = copy.deepcopy(test)
            changed['samples'][0][field] = value
            with self.assertRaises(ValueError): query_map(cohort, changed)
        duplicate = copy.deepcopy(test)
        duplicate['samples'][1]['source_sha256'] = duplicate['samples'][0]['source_sha256']
        with self.assertRaises(ValueError): query_map(cohort, duplicate)
        absent = copy.deepcopy(test)
        absent['samples'].pop()
        with self.assertRaises(ValueError): query_map(cohort, absent)

    def data(self):
        schema = dict(query_ids=['x', 'y'], labels=[0, 1], participant_ids=['p1', 'p2'], class_ids=[0, 1], split='test')
        ranks = {f'{m}_s{s}': np.array([[0, 1], [1, 0]], dtype=np.int16) for m in METHODS for s in SEEDS}
        observations = [dict(sample_id=k, method=m, encoded_windows=8 if m == 'full_activity' else 2,
            source_windows=8, decode_seconds=.1, preprocess_seconds=.1,
            encoder_wrapper_seconds=.6 if m == 'full_activity' else .1, pool_seconds=.1,
            three_gallery_search_seconds=.1, end_to_end_seconds=1 if m == 'full_activity' else .5)
            for k in schema['query_ids'] for m in RECIPES]
        diagnostics = {k: dict(source_windows=8, effective_activity_fraction=.8) for k in schema['query_ids']}
        thresholds = dict(native_window_count_quartiles=[6, 8, 10], effective_activity_fraction_quartiles=[.5, .8, .9])
        return schema, ranks, observations, diagnostics, thresholds

    def test_exact_search_class_max_and_tie_oracle(self):
        gallery = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float64)
        ranks = search([1, 0], {'1701': (gallery, [np.array([0]), np.array([1, 2])])}, 2)
        self.assertEqual(ranks['1701'], [0, 1])
        ranks = search([0, 1], {'1701': (gallery, [np.array([0]), np.array([1, 2])])}, 2)
        self.assertEqual(ranks['1701'], [1, 0])

    def test_complete_rankings_cost_and_noninferiority_oracle(self):
        data = self.data()
        result = analyze(*data)
        self.assertTrue(result['joint_confirmatory_screen'])
        self.assertEqual(result['mean_metrics']['uniform8']['dcg'], 1)
        self.assertEqual(result['conditional_participant_bootstrap']['interval95_dcg'], [0, 0])
        self.assertEqual(result['costs']['uniform8']['window_reduction_factor'], 4)
        self.assertEqual(result['qualitative_cases']['both_success']['eligible_N'], 2)
        self.assertEqual(result['descriptive_slices']['length']['Q2']['N'], 2)
        schema, ranks, observations, diagnostics, cuts = self.data()
        for seed in SEEDS:
            ranks[f'uniform8_s{seed}'] = np.array([[1, 0], [0, 1]])
        self.assertFalse(analyze(schema, ranks, observations, diagnostics, cuts)['quality_noninferiority_screen'])

    def test_reject_missing_permutations_coverage_and_time_accounting(self):
        schema, ranks, observations, _, _ = self.data()
        broken = dict(ranks)
        broken.pop('full_mean_s1701')
        with self.assertRaises(ValueError): verify_rankings(schema, broken)
        broken = dict(ranks)
        broken['full_mean_s1701'] = np.array([[0, 0], [0, 1]])
        with self.assertRaises(ValueError): verify_rankings(schema, broken)
        with self.assertRaises(ValueError): cost_summary(schema, observations[:-1])
        with self.assertRaises(ValueError): cost_summary(schema, observations + [observations[0]])
        changed = copy.deepcopy(observations)
        changed[0]['end_to_end_seconds'] = .9
        with self.assertRaises(ValueError): cost_summary(schema, changed)
        changed = copy.deepcopy(observations)
        changed[1]['source_windows'] = 9
        with self.assertRaises(ValueError): cost_summary(schema, changed)

    def test_proxy_clamping_and_zero_fallback(self):
        self.assertEqual(activity_proxy(np.zeros((4, 2)))['effective_activity_fraction'], 0)
        self.assertTrue(activity_proxy(np.full((4, 2), -1))['zero_activity'])
        self.assertEqual(activity_proxy(np.ones((4, 2)))['effective_activity_fraction'], 1)
        self.assertEqual(activity_proxy([[1, 0], [0, -1]])['effective_activity_fraction'], .5)
        with self.assertRaises(ValueError): activity_proxy([[float('nan'), 0]])
