import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import run_ncslgr_author_pool as suite
from signrepr.io import sha256


class AuthorPoolingTests(unittest.TestCase):
    def fixture(self, root, empty_split=False, invalid_observation=False):
        native, signrep, output = [root / name for name in ['native', 'signrep', 'output']]
        for folder in [native, signrep, output]:
            folder.mkdir()
        manifest, indexes = [], {'native': [], 'signrep': []}
        for split in ['train', 'val', 'test']:
            for label in ['NEG', 'WH', 'YN']:
                sid = split + ':' + label
                manifest.append({'sample_id': sid + ':body', 'utterance_sample_id': sid,
                                 'view_role': 'body', 'split': split, 'label': label,
                                 'utterance_start_ms': 0, 'utterance_end_ms': 966})
                for name, folder in [('native', native), ('signrep', signrep)]:
                    path = folder / (sid.replace(':', '_') + '.npz')
                    # Distinct directions make retaining the imputed middle token observable.
                    raw = np.zeros((3, 768), np.float32)
                    raw[0, 0], raw[1, 1], raw[2, 0] = 1, 4, 1
                    valid = np.array([True, False, True])
                    if name == 'native':
                        valid[:] = False
                    elif empty_split and split == 'val':
                        valid[:] = False
                    data = {'embeddings': raw, 'center_sec': np.array([.05, .5, .85]),
                            'valid_mask': valid, 'valid_frame_mask': valid[:, None]}
                    if name == 'native':
                        mask = np.zeros((3, 4), bool)
                        if invalid_observation and sid == 'train:NEG':
                            mask[0] = True
                        data.update(stream_observed_mask=mask, embeddings_layer_average=raw.copy())
                    np.savez(path, **data)
                    path.with_suffix('.json').write_text(json.dumps({'shard_sha256': sha256(path), 'cache_material': {'locked': 'identity'}}))
                    indexes[name].append({'sample_id': sid + ':body', 'status': 'SUCCESS', 'shard': str(path)})
        source = root / 'manifest.jsonl'
        for path, rows in [(source, manifest), (native / 'index.jsonl', indexes['native']),
                           (signrep / 'index.jsonl', indexes['signrep'])]:
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        (native / 'extraction_report.json').write_text(json.dumps({'status': 'PASS', 'manifest_sha256': 'native-source-hash'}))
        protocol_file = root / 'native_protocol.json'
        protocol_file.write_text(json.dumps({'manifest_sha256': 'native-source-hash'}))
        protocol = {'identity_sha256': {}, 'native_cache_identity': {'locked': 'identity'},
                    'backbones': ['signrep', 'shubert_last', 'shubert_average'], 'classes': ['NEG', 'WH', 'YN']}
        patches = patch.multiple(suite, NATIVE=native, SIGNREP=signrep, OUTPUT=output,
                                 MANIFEST=source, NATIVE_PROTOCOL=protocol_file)
        return patches, protocol, output, native

    def test_zero_observed_native_tokens_retained_without_forging_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            patches, protocol, output, native = self.fixture(Path(tmp))
            index_before = sha256(native / 'index.jsonl')
            with patches, patch.object(suite, 'identities', return_value={}):
                rows, features, gold, split = suite.prepare(protocol)
            self.assertEqual(len(rows), 9)
            self.assertEqual(features['shubert_last'].shape, (9, 768))
            np.testing.assert_allclose(features['shubert_last'][0, :2], np.array([2, 4]) / np.sqrt(20), atol=1e-6)
            np.testing.assert_allclose(features['signrep'][0, :2], [1, 0])
            coverage = json.loads((output / 'coverage.json').read_text())
            self.assertEqual(coverage['exclusions'], [])
            self.assertTrue(all(r['all_four_observed_supported_frames'] == 0 for r in coverage['observation_counts']))
            self.assertTrue(all(r['per_stream_observed_supported_frames'] == [0, 0, 0, 0] for r in coverage['observation_counts']))
            self.assertEqual(sha256(native / 'index.jsonl'), index_before)

    def test_empty_validation_population_records_counts_then_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            patches, protocol, output, _ = self.fixture(Path(tmp), empty_split=True)
            with patches, patch.object(suite, 'identities', return_value={}):
                with self.assertRaisesRegex(ValueError, 'Locked class lacks'):
                    suite.prepare(protocol)
            coverage = json.loads((output / 'coverage.json').read_text())
            self.assertEqual(coverage['class_counts']['val'], [0, 0, 0])
            self.assertEqual(len(coverage['exclusions']), 3)
            self.assertFalse((output / 'summary.json').exists())

    def test_inconsistent_observation_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            patches, protocol, _, _ = self.fixture(Path(tmp), invalid_observation=True)
            with patches, patch.object(suite, 'identities', return_value={}):
                with self.assertRaisesRegex(ValueError, 'observed mask differs'):
                    suite.prepare(protocol)

    def test_partial_and_failed_native_reports_do_not_trigger_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'extraction_report.json').write_text(json.dumps({'status': 'PARTIAL_BOUNDED_VALIDATION'}))
            (folder / 'status.json').write_text(json.dumps({'status': 'RUNNING', 'success': 1}))
            with patch.object(suite, 'NATIVE', folder):
                self.assertFalse(suite.native_ready()[0])
                (folder / 'status.json').write_text(json.dumps({'status': 'FAIL'}))
                with self.assertRaisesRegex(ValueError, 'did not PASS'):
                    suite.native_ready()


if __name__ == '__main__':
    unittest.main()
