"""Synthetic numerical/provenance tests; these fixtures are not linguistic gold."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from run_grounding import prepare
from signrepr import grounding
from signrepr.io import sha256


class GroundingTests(unittest.TestCase):
    policy = dict(maximum_cost=1e-12, min_duration=.1, max_duration=20, nms_iou=.5)

    def test_subsequence_finds_middle_and_covers_entire_query(self):
        query = np.eye(2)
        target = np.array([[-1, 0], [1, 0], [0, 1], [-1, 0]])
        found = grounding.subsequence_dtw(query, target, np.arange(4), np.arange(1, 5), **self.policy)
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0]['start_sec'], found[0]['end_sec']), (1., 3.))
        self.assertEqual(found[0]['query_tokens_covered'], 2)
        self.assertEqual(found[0]['score'], 0.)
        # A matching first query token alone must not yield a zero-cost hit.
        self.assertEqual(grounding.subsequence_dtw(query, query[:1], [0], [1], **self.policy), [])

    def test_missing_token_gap_keeps_source_clock_and_duration_policy(self):
        query = np.eye(2)
        found = grounding.subsequence_dtw(query, query, [0, 10], [1, 11], **self.policy)
        self.assertEqual((found[0]['start_sec'], found[0]['end_sec']), (0., 11.))
        self.assertEqual(grounding.subsequence_dtw(query, query, [0, 10], [1, 11],
                         **{**self.policy, 'max_duration': 3}), [])
        with self.assertRaisesRegex(ValueError, 'chronological'):
            grounding.subsequence_dtw(query, query, [10, 0], [11, 1], **self.policy)

    def test_empty_targets_retained_and_invalid_query_policy_rejected(self):
        self.assertEqual(grounding.subsequence_dtw(np.eye(2), np.empty((0, 2)), [], [], **self.policy), [])
        with self.assertRaisesRegex(ValueError, 'Zero feature'):
            grounding.subsequence_dtw(np.zeros((1, 2)), np.empty((0, 2)), [], [], **self.policy)
        with self.assertRaisesRegex(ValueError, 'finite search policy'):
            grounding.subsequence_dtw(np.eye(2), np.eye(2), [0, 1], [1, 2],
                                     **{**self.policy, 'max_duration': float('inf')})

    def test_nms_duplicate_hits_and_disjoint_occurrences(self):
        candidates = [dict(start_sec=0., end_sec=2., score=1.),
                      dict(start_sec=.1, end_sec=2.1, score=.9),
                      dict(start_sec=5., end_sec=7., score=.8)]
        self.assertEqual(grounding.nms(candidates, .5), [candidates[0], candidates[2]])

    def test_one_to_one_ap_duplicates_negatives_and_boundaries(self):
        def event(target, start=0, end=1, **more):
            return dict(query_id='q', target_id=target, start_sec=start, end_sec=end, **more)
        gold = [event('t0'), event('t1')]
        predictions = [event('t0', score=1), event('t0', score=.9),
                       event('t1', score=.8), event('t2', score=.7)]
        result = grounding.evaluate(predictions, gold, dict(t0=60, t1=60, t2=60), ['q', 'no_positive'], .5)
        q = result['per_query']['q']
        self.assertAlmostEqual(q['AP'], 5/6)
        self.assertAlmostEqual(result['pooled_AP'], 5/6)
        self.assertEqual(q['true_positives'], 2)
        self.assertEqual(q['false_positives'], 2)
        self.assertAlmostEqual(q['false_positives_per_minute'], 2/3)
        self.assertEqual(q['negative_target_false_positives_per_minute'], 1)
        self.assertEqual(q['mean_matched_onset_offset_error_sec'], [0, 0])
        self.assertIsNone(result['per_query']['no_positive']['AP'])
        missed = grounding.evaluate([], gold, dict(t0=60, t1=60, t2=60), ['q'], .5)
        self.assertEqual(missed['macro_AP'], 0)
        self.assertIsNone(missed['mean_matched_onset_offset_error_sec'])
        shifted = grounding.evaluate([event('t0', .1, 1.2, score=1)], gold, dict(t0=60, t1=60), ['q'], .5)
        np.testing.assert_allclose(shifted['mean_matched_onset_offset_error_sec'], [.1, .2])
        with self.assertRaisesRegex(ValueError, 'Duplicate gold'):
            grounding.evaluate([], gold+gold[:1], dict(t0=60, t1=60), ['q'], .5)

    def test_recording_bootstrap_keeps_grouped_positives_and_absences(self):
        targets = [dict(target_id='positive', recording_group='a', duration_sec=60),
                   dict(target_id='paired_negative', recording_group='a', duration_sec=60),
                   dict(target_id='other_negative', recording_group='b', duration_sec=60)]
        gold = [dict(query_id='q', target_id='positive', start_sec=0, end_sec=1)]
        prediction = [dict(**gold[0], score=1)]
        original = grounding.evaluate
        def observe(pred, truth, durations, queries, threshold):
            if all(':' in tid for tid in durations):
                self.assertEqual(len(durations), 2+len(truth))
                self.assertEqual(len(pred), len(truth))
                for g in truth:
                    self.assertIn(g['target_id'].replace('positive', 'paired_negative'), durations)
            return original(pred, truth, durations, queries, threshold)
        with patch.object(grounding, 'evaluate', side_effect=observe):
            result = grounding.recording_bootstrap(prediction, gold, targets, ['q'], .5)
        self.assertEqual(result['macro_AP_ci95_on_supported_draws'], [1, 1])
        self.assertGreater(result['draws_without_any_positive_query'], 0)
        self.assertLess(result['draws_without_any_positive_query'], 2000)

    def fixture(self, base):
        """Pin small fake sources/caches solely to exercise the runner contract."""
        identity = dict(config_sha256='a'*64, implementation_sha256='b'*64)
        config = dict(status='LOCKED_VERIFIED_ASL_GROUNDING', **self.policy,
                      common_feature_identity=identity, feature_dimension=2,
                      algorithm='subsequence_dtw_sum_cosine_then_path_length_normalization',
                      device='cpu', primary_iou=.5, secondary_iou_thresholds=[.3, .5, .7],
                      maximum_total_dtw_cells=1000, seed=42, resamples=2000,
                      media_root=str(base), cache_root=str(base.relative_to(ROOT)))
        def pin(name, value):
            path = base / (name+'.json')
            path.write_text(json.dumps(value))
            config[name] = str(path.relative_to(ROOT)); config[name+'_sha256'] = sha256(path)
        pin('gate_path', {g: dict(status='PASS_SYNTHETIC_UNIT_TEST_ONLY') for g in ['B-G0', 'B-G1', 'B-G2']})
        # gate_path uses the historic gate_sha256 key.
        config['gate_sha256'] = config.pop('gate_path_sha256')
        pin('validation_selection', dict(source_split='val', test_used=False, parameters=self.policy))
        def cache(identifier, role, tokens, valid):
            video = base / (identifier+'.fake_media'); video.write_bytes(identifier.encode())
            shard = base / (identifier+'.npz')
            np.savez(shard, embeddings=tokens, valid_mask=np.array(valid, dtype=bool),
                     valid_frame_mask=np.ones((len(tokens), 2), dtype=bool),
                     window_start_sec=np.arange(len(tokens), dtype=float),
                     window_end_sec=np.arange(1, len(tokens)+1, dtype=float))
            meta = shard.with_suffix('.json')
            meta.write_text(json.dumps(dict(**identity, source_sha256=sha256(video), shard_sha256=sha256(shard))))
            return {role+'_id': identifier, 'video_path': video.name, 'video_sha256': sha256(video),
                    'feature_path': shard.name, 'feature_sha256': sha256(shard), 'metadata_sha256': sha256(meta),
                    'language': 'ASL', 'split': 'test', 'canonical_signer_id': identifier,
                    'recording_group': identifier, 'fixture_scope': 'synthetic unit test only'}
        q = cache('q', 'query', np.eye(2), [True, True]); q['mapping_status'] = 'EXPERT_VERIFIED_SAME_ASL_LEXEME'
        t = cache('t', 'target', np.array([[-1, 0], [1, 0], [0, 1]]), [True]*3)
        n = cache('n', 'target', np.eye(2), [False, False])
        for target in [t, n]:
            target.update(duration_sec=4, annotation_status='EXPERT_VERIFIED_OCCURRENCES_AND_ABSENCE',
                          exhaustively_reviewed_query_ids=['q'])
        manifests = dict(queries=[q], targets=[t, n], occurrences=[dict(query_id='q', target_id='t', start_sec=1, end_sec=3)])
        for name, rows in manifests.items():
            path = base / (name+'.jsonl'); path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            config[name] = str(path.relative_to(ROOT)); config[name+'_sha256'] = sha256(path)
        return config, manifests

    def test_runner_rejects_unverified_mapping_absence_and_test_selection(self):
        with self.assertRaisesRegex(ValueError, 'blocked or unlocked'):
            prepare(dict(status='BLOCKED_DATA'))
        with tempfile.TemporaryDirectory(prefix='grounding_unit_', dir=ROOT / 'features') as folder:
            base = Path(folder); config, manifests = self.fixture(base)
            prepare(config)
            with self.assertRaisesRegex(ValueError, 'Pinned common feature'):
                prepare({**config, 'common_feature_identity': {}})
            for name, key, value, message in [('queries', 'mapping_status', 'ENGLISH_WORD_MATCH_ONLY', 'same-sign'),
                                              ('targets', 'exhaustively_reviewed_query_ids', [], 'cannot be a negative')]:
                row = manifests[name][0]; old = row[key]; row[key] = value
                path = ROOT / config[name]; path.write_text(''.join(json.dumps(r)+'\n' for r in manifests[name]))
                config[name+'_sha256'] = sha256(path)
                with self.assertRaisesRegex(ValueError, message):
                    prepare(config)
                row[key] = old; path.write_text(''.join(json.dumps(r)+'\n' for r in manifests[name]))
                config[name+'_sha256'] = sha256(path)
            selection = ROOT / config['validation_selection']; value = json.loads(selection.read_text()); value['test_used'] = True
            selection.write_text(json.dumps(value)); config['validation_selection_sha256'] = sha256(selection)
            with self.assertRaisesRegex(ValueError, 'validation-only'):
                prepare(config)

    def test_synthetic_cli_end_to_end_keeps_empty_negative_and_pins_results(self):
        with tempfile.TemporaryDirectory(prefix='grounding_unit_', dir=ROOT / 'features') as folder:
            base = Path(folder); config, _ = self.fixture(base)
            path = base / 'protocol.yaml'; path.write_text(yaml.safe_dump(config))
            output = base / 'output'
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_grounding.py'),
                                     '--config', str(path), '--output', str(output)], cwd=ROOT,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['metrics']['macro_AP'], 1)
            self.assertEqual(summary['metrics']['per_query']['q']['verified_negative_targets'], 1)
            self.assertEqual(summary['evaluated_query_target_pairs'], 2)
            self.assertEqual(summary['secondary_mean_macro_AP_across_registered_iou'], 1)
            self.assertEqual(summary['predictions_sha256'], sha256(output / 'predictions.jsonl'))
            metrics = json.loads((output / 'metrics.json').read_text())
            self.assertEqual(metrics['primary']['macro_AP'], 1)
            self.assertEqual(metrics['coverage_sha256'], sha256(output / 'coverage.json'))
            self.assertTrue((output / 'predictions.csv').is_file())


if __name__ == '__main__':
    unittest.main()
