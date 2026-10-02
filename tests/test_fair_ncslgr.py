import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import run_fair_ncslgr as suite
from signrepr.io import sha256


class FairNativeGrammarTests(unittest.TestCase):
    def fixture(self, root, zero=False):
        native, signrep, output = [root / name for name in ['native', 'signrep', 'output']]
        for p in [native, signrep, output]:
            p.mkdir()
        rows, manifest, indexes = [], [], {'native': [], 'signrep': []}
        for split in ['train', 'val', 'test']:
            for label in ['NEG', 'WH', 'YN']:
                sid = split + ':' + label
                rows.append({'sample_id': sid, 'split': split, 'label': label, 'scope_supervised': True,
                             'utterance_support_sec': [0., 1.]})
                manifest.append({'sample_id': sid + ':body'})
                for name, folder in [('native', native), ('signrep', signrep)]:
                    path = folder / (sid.replace(':', '_') + '.npz')
                    valid = np.array([True, False, True])
                    if zero and name == 'native' and sid == 'val:YN':
                        valid[:] = False
                    data = {'embeddings': np.ones((3, 768), np.float32), 'center_sec': np.array([.05, .5, .85]),
                            'valid_mask': valid, 'valid_frame_mask': valid[:, None]}
                    if name == 'native':
                        data.update(stream_observed_mask=np.repeat(valid[:, None], 4, axis=1),
                                    embeddings_layer_average=data['embeddings'])
                    np.savez(path, **data)
                    path.with_suffix('.json').write_text(json.dumps({'shard_sha256': sha256(path), 'cache_material': {'locked': 'identity'}}))
                    indexes[name].append({'sample_id': sid + ':body', 'status': 'SUCCESS', 'shard': str(path)})
        targets, source = root / 'targets.jsonl', root / 'manifest.jsonl'
        for path, values in [(targets, rows), (source, manifest), (native / 'index.jsonl', indexes['native']),
                             (signrep / 'index.jsonl', indexes['signrep'])]:
            path.write_text(''.join(json.dumps(r)+'\n' for r in values))
        (native / 'extraction_report.json').write_text(json.dumps({'status': 'PASS', 'manifest_sha256': sha256(source)}))
        p = {'identity_sha256': {}, 'native_cache_identity': {'locked': 'identity'},
             'backbones': ['signrep', 'shubert_last', 'shubert_average'], 'classes': ['NEG', 'WH', 'YN']}
        return native, signrep, output, targets, source, p

    def test_sparse_clock_gaps_retained_and_identical_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            native, signrep, output, targets, source, protocol = self.fixture(Path(tmp))
            with patch.multiple(suite, NATIVE=native, SIGNREP=signrep, OUTPUT=output, TARGETS=targets, MANIFEST=source), patch.object(suite, 'identities', return_value={}):
                rows, tokens, clocks = suite.load_data(protocol)
            self.assertEqual(len(rows), 9)
            for backbone in protocol['backbones']:
                self.assertEqual(len(tokens[backbone]), 9)
                np.testing.assert_allclose(clocks[backbone][0], [0., .05, .85, 1.])

    def test_loss_of_class_support_stops_before_fit_and_records_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            native, signrep, output, targets, source, protocol = self.fixture(Path(tmp), zero=True)
            with patch.multiple(suite, NATIVE=native, SIGNREP=signrep, OUTPUT=output, TARGETS=targets, MANIFEST=source), patch.object(suite, 'identities', return_value={}):
                with self.assertRaisesRegex(ValueError, 'Locked class lacks'):
                    suite.load_data(protocol)
            coverage = json.loads((output / 'coverage.json').read_text())
            self.assertEqual(coverage['excluded'][0]['sample_id'], 'val:YN')
            self.assertFalse((output / 'summary.json').exists())

    def test_live_status_overrides_stale_partial_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            native = Path(tmp)
            (native / 'status.json').write_text(json.dumps({'status': 'RUNNING', 'success': 1}))
            (native / 'extraction_report.json').write_text(json.dumps({'status': 'PARTIAL_BOUNDED_VALIDATION'}))
            with patch.object(suite, 'NATIVE', native):
                self.assertFalse(suite.native_ready()[0])
                (native / 'status.json').write_text(json.dumps({'status': 'FAIL'}))
                with self.assertRaisesRegex(ValueError, 'without full PASS'):
                    suite.native_ready()


if __name__ == '__main__':
    unittest.main()
