"""Risk checks for byte ceilings, provenance completeness and compact round trips."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import unittest

import numpy as np

from bounded_artifacts import BoundedArtifactStore, array_spec, inspect_file_plan, json_bytes, verify_store
from compact_geometry_artifacts import (FAMILIES, METRIC_COLUMNS, compact_specs, pack_geometry,
                                        pack_predictions, ranking_metadata, unpack_covariance,
                                        unpack_predictions, validate_rank_arrays)

ROOT = Path(os.environ['SIGNREP_ARTIFACT_FIXTURES'])


class BoundedTests(unittest.TestCase):
    def path(self, suffix=''):
        return ROOT / (self._testMethodName + suffix)

    def store(self, caps=None, ceiling=8192, **kwargs):
        return BoundedArtifactStore(self.path(), caps or {'data.bin': 128}, ceiling,
                                    manifest_cap=4096, **kwargs)

    def test_overbudget_rejected_before_directory_or_bytes(self):
        with self.assertRaises(ValueError):
            self.store({'six_dense_matrices.bin': 6 * 768 * 768 * 8}, 20 * 1024**2)
        self.assertFalse(self.path().exists())

    def test_exact_boundary_includes_atomic_replacement_scratch(self):
        plan = inspect_file_plan({'status.json': 100, 'data.bin': 200}, 400, ('status.json',))
        self.assertTrue(plan['passed'])
        self.assertFalse(inspect_file_plan({'status.json': 100, 'data.bin': 200}, 399, ('status.json',))['passed'])
        self.assertEqual(plan['replacement_scratch_bytes'], 100)

    def test_invalid_plan_paths_types_and_collisions(self):
        for name in ('../x', '/x', 'a//b', 'a/./b', 'a/../b', 'a\\b', '.partial', ''):
            with self.assertRaises(ValueError):
                inspect_file_plan({name: 1}, 10)
        for cap in (True, -1, 0, 1.2):
            with self.assertRaises(ValueError):
                inspect_file_plan({'x': cap}, 10)
        with self.assertRaises(ValueError):
            inspect_file_plan({'a': 1, 'a/b': 1}, 10)

    def test_cap_violation_undeclared_and_immutable_write_leave_bytes_unchanged(self):
        store = self.store()
        for name, payload in (('data.bin', b'x' * 129), ('unknown.bin', b'x')):
            with self.assertRaises(ValueError):
                store.write_bytes(name, payload)
        self.assertEqual(list(self.path().rglob('*')), [])
        store.write_bytes('data.bin', b'original')
        with self.assertRaises(ValueError):
            store.write_bytes('data.bin', b'replaced')
        self.assertEqual((self.path() / 'data.bin').read_bytes(), b'original')

    def test_status_replacement_snapshot_log_and_manifest_complete(self):
        caps = {'status.json': 1024, 'config.json': 1024, 'source_snapshot/fixture.py': 1024,
                'environment.json': 1024, 'stdout.txt': 1024, 'report.json': 1024}
        store = self.store(caps, 16384, replaceable=('status.json',))
        store.write_json('status.json', {'status': 'running'})
        for name in ('config.json', 'environment.json', 'report.json'):
            store.write_json(name, {'finite': 1.0})
        store.write_bytes('source_snapshot/fixture.py', b'# frozen fixture\n')
        store.write_bytes('stdout.txt', b'bounded log\n')
        store.write_json('status.json', {'status': 'completed', 'exit_code': 0})
        result = store.finalize()
        verified = verify_store(self.path())
        self.assertEqual(result['final_logical_bytes'], verified['logical_bytes'])
        self.assertTrue(verified['passed'])
        with self.assertRaises(ValueError):
            store.write_bytes('stdout.txt', b'late')
        with self.assertRaises(FileExistsError):
            self.store(caps, 16384)

    def test_missing_file_or_unexpected_file_cannot_finalize(self):
        store = self.store({'data.bin': 128, 'missing.bin': 128})
        store.write_bytes('data.bin', b'x')
        with self.assertRaises(ValueError):
            store.finalize()
        (self.path() / 'unexpected.bin').write_bytes(b'external')
        with self.assertRaises(ValueError):
            store.write_bytes('missing.bin', b'x')

    def test_same_size_corruption_is_detected_by_digest(self):
        store = self.store()
        store.write_bytes('data.bin', b'original')
        store.finalize()
        (self.path() / 'data.bin').write_bytes(b'changed!')
        with self.assertRaises(ValueError):
            verify_store(self.path())

    def test_symlink_rejected_without_following(self):
        store = self.store()
        (self.path() / 'link').symlink_to('/dev/null')
        with self.assertRaises(ValueError):
            store.write_bytes('data.bin', b'x')

    def test_nonfinite_object_wrong_dtype_shape_rejected_before_write(self):
        spec = array_spec((2, 3), '<f8')
        store = self.store({'array.npy': spec['file_bytes']})
        for a in (np.ones((3, 2)), np.ones((2, 3), dtype=np.float32),
                  np.full((2, 3), np.nan), np.full((2, 3), np.inf)):
            with self.assertRaises(ValueError):
                store.write_array('array.npy', a, spec)
        for dtype in ('O', 'U4', [('x', 'f8')]):
            with self.assertRaises(ValueError):
                array_spec((2,), dtype)
        with self.assertRaises(ValueError):
            store.write_json('array.npy', {'bad': float('nan')})
        self.assertEqual(list(self.path().rglob('*')), [])

    def test_npy_exact_size_and_fortran_input_roundtrip(self):
        a = np.asfortranarray(np.arange(24, dtype='<f8').reshape(4, 6))
        spec = array_spec(a.shape, a.dtype)
        store = self.store({'array.npy': spec['file_bytes']})
        store.write_array('array.npy', a, spec)
        store.finalize()
        self.assertEqual((self.path() / 'array.npy').stat().st_size, spec['file_bytes'])
        self.assertTrue(np.array_equal(np.load(self.path() / 'array.npy', allow_pickle=False), a))

    def test_rank_permutations_and_metric_bounds_not_silently_cast(self):
        metrics = np.zeros((2, 6), dtype='<f8')
        for ranks in (np.array([[0, 1, 1], [0, 1, 2]], dtype='<i2'),
                      np.array([[-1, 1, 2], [0, 1, 2]], dtype='<i2'),
                      np.array([[0, 1, 3], [0, 1, 2]], dtype='<i2'),
                      np.array([[0, 1, 2], [0, 1, 2]], dtype='<i4')):
            with self.assertRaises(ValueError):
                validate_rank_arrays(ranks, metrics, 3)
        valid = np.array([[0, 1, 2], [2, 1, 0]], dtype='<i2')
        for value in (-0.01, 1.01, np.nan):
            m = metrics.copy(); m[0, 0] = value
            with self.assertRaises(ValueError):
                validate_rank_arrays(valid, m, 3)
        with self.assertRaises(ValueError):
            compact_specs(768, 9192, 32769)

    def test_prediction_identity_cartesian_coverage_and_noncontiguous_class_roundtrip(self):
        meta = ranking_metadata([10, 90, 300], ['raw', 'lw'], [1701], ['a', 'b'], [90, 300], ['P1', 'P2'])
        rows = []
        for mode in meta['modes']:
            for qi, key in enumerate(meta['query_ids']):
                rows.append(dict(mode=mode, support_seed=1701, sample_id=key,
                                 label=meta['labels'][qi], participant_id=meta['participants'][qi],
                                 ranking=[300, 90, 10], dcg=0.5, top1=0., top5=1., top10=1., top20=1., mrr=0.5))
        arrays = pack_predictions(list(reversed(rows)), meta)
        self.assertEqual(arrays['ranks.npy'].tolist(), [[2, 1, 0]] * 4)
        self.assertEqual(unpack_predictions(arrays['ranks.npy'], arrays['row_metrics.npy'], meta), rows)
        for kind in ('duplicate', 'missing', 'label', 'participant', 'ranking', 'extra_field', 'metric_type'):
            bad = copy.deepcopy(rows)
            if kind == 'duplicate': bad[-1] = bad[0]
            elif kind == 'missing': bad.pop()
            elif kind == 'label': bad[0]['label'] = 300
            elif kind == 'participant': bad[0]['participant_id'] = 'wrong'
            elif kind == 'ranking': bad[0]['ranking'] = [10, 90, 90]
            elif kind == 'metric_type': bad[0]['dcg'] = '0.5'
            else: bad[0]['silently_dropped'] = 1
            with self.assertRaises(ValueError):
                pack_predictions(bad, meta)

    def test_full_size_shape_plan_and_lossless_covariance_rank_metric_records(self):
        d, n, classes = 768, 9192, 100
        specs = compact_specs(d, n, classes)
        fitted = {'mean': np.arange(d, dtype='<f8') / 1024}
        for i, family in enumerate(FAMILIES):
            fitted[family + '_covariance'] = np.eye(d, dtype='<f8') * (i + 1)
            fitted[family + '_eigenvalues'] = np.full(d, i + 1., dtype='<f8')
        arrays = pack_geometry(fitted)
        arrays['ranks.npy'] = (np.arange(classes)[None, :] + np.arange(n)[:, None]) % classes
        arrays['ranks.npy'] = arrays['ranks.npy'].astype('<i2')
        true_rank = (-np.arange(n)) % classes + 1
        arrays['row_metrics.npy'] = np.column_stack((1 / np.log2(true_rank + 1),
            true_rank <= 1, true_rank <= 5, true_rank <= 10, true_rank <= 20, 1 / true_rank)).astype('<f8')
        validate_rank_arrays(arrays['ranks.npy'], arrays['row_metrics.npy'], classes)
        caps = {name: spec['file_bytes'] for name, spec in specs.items()}
        caps.update({'metadata.json': 1024**2, 'source_snapshot/fixture.py': 1024**2,
                     'config.json': 1024**2, 'stdout.txt': 1024**2, 'report.json': 1024**2})
        store = self.store(caps, 20 * 1024**2)
        for name, array in arrays.items():
            store.write_array(name, array, specs[name])
        for name in ('metadata.json', 'config.json', 'report.json'):
            store.write_json(name, {'fixture_only': True, 'dimension': d, 'prediction_rows': n, 'class_count': classes})
        store.write_bytes('source_snapshot/fixture.py', Path(__file__).read_bytes())
        store.write_bytes('stdout.txt', b'synthetic serialization only; no task score\n')
        result = store.finalize()
        self.assertLess(result['plan']['maximum_logical_bytes'], 20 * 1024**2)
        verified = verify_store(self.path())
        self.assertTrue(verified['passed'])
        for family in FAMILIES:
            upper = np.load(self.path() / (family + '_covariance_upper.npy'), allow_pickle=False)
            self.assertTrue(np.array_equal(unpack_covariance(upper, d), fitted[family + '_covariance']))
        for name in ('mean.npy', 'ranks.npy', 'row_metrics.npy'):
            saved = np.load(self.path() / name, allow_pickle=False)
            self.assertTrue(np.array_equal(saved, arrays[name]))
            self.assertEqual(saved.tobytes(), arrays[name].tobytes())
        self.assertEqual(sum(s['payload_bytes'] for s in specs.values()), 9391296)

    def test_asymmetric_covariance_cannot_be_lossily_packed(self):
        fitted = {'mean': np.zeros(2, dtype='<f8')}
        for family in FAMILIES:
            fitted[family + '_covariance'] = np.eye(2, dtype='<f8')
            fitted[family + '_eigenvalues'] = np.ones(2, dtype='<f8')
        fitted['residual_covariance'][0, 1] = 1e-16
        with self.assertRaises(ValueError):
            pack_geometry(fitted)
        fitted['residual_covariance'][0, 1] = 0.
        fitted['residual_covariance'][1, 0] = -0.
        with self.assertRaises(ValueError):
            pack_geometry(fitted)


if __name__ == '__main__':
    unittest.main()
