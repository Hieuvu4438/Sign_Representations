import ast
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from signrepr.manifests import sample
from signrepr.validation import audit_rows, validate_rows
from signrepr.io import sha256, write_json
from signrepr.signrep import constructor_literal, windows
from extract_features import cache_key, valid_shard
from fetch_asset import validate_asset_bytes
import numpy as np


class IntegrityTests(unittest.TestCase):
    def row(self, root, split='train', name='clip.mp4'):
        (root / name).write_bytes(b'video')
        metadata = root / 'labels.json'
        metadata.write_text('{}')
        return sample('test', 'ase', root, name, split, metadata)

    def test_grouped_camera_views_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = self.row(root), self.row(root, 'test', 'view2.mp4')
            a['recording_group'] = b['recording_group'] = 'recording_one'
            report = audit_rows([a, b])
            self.assertEqual(report['status'], 'FAIL_RECORDING_OVERLAP')
            self.assertEqual(len(report['recording_overlaps']), 1)

    def test_unknown_groups_do_not_pass_clean_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.row(Path(directory))
            row['grouping_status'] = 'SOURCE_RECORDING_UNKNOWN'
            self.assertEqual(audit_rows([row])['status'], 'UNVERIFIED_GROUPS')

    def test_symlink_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            row = self.row(root)
            (root / 'escape.mp4').symlink_to(Path(outside))
            row['relative_path'] = 'escape.mp4'
            self.assertTrue(any(x['reason'] == 'PATH_ESCAPES_SOURCE_ROOT' for x in validate_rows([row])['errors']))

    def test_bad_timestamp_nan_and_outside_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.row(Path(directory))
            row.update(start_sec=float('nan'), end_sec=2.)
            self.assertEqual(validate_rows([row])['status'], 'FAIL')
            row.update(start_sec=1., end_sec=2., duration_sec=1.5)
            self.assertTrue(any(x['reason'] == 'INTERVAL_OUTSIDE_DURATION' for x in validate_rows([row])['errors']))

    def test_window_padding_and_tail_have_correct_time_support(self):
        self.assertEqual(windows(3, 16, 2), [(0, 3, 13)])
        self.assertEqual(windows(19, 16, 2), [(0, 16, 0), (2, 18, 0)])
        with self.assertRaises(ValueError):
            windows(0, 16, 2)

    def test_constructor_cannot_execute_calls(self):
        self.assertEqual(constructor_literal(ast.parse("{'x': 22 * 2}", mode='eval').body), {'x': 44})
        with self.assertRaises(ValueError):
            constructor_literal(ast.parse("__import__('os').getcwd()", mode='eval').body)

    def test_cache_changes_when_source_config_or_code_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.row(Path(directory))
            key = cache_key(row, 'bytes1', {'stride': 2}, 'code1')
            self.assertNotEqual(key, cache_key(row, 'bytes2', {'stride': 2}, 'code1'))
            self.assertNotEqual(key, cache_key(row, 'bytes1', {'stride': 4}, 'code1'))
            self.assertNotEqual(key, cache_key(row, 'bytes1', {'stride': 2}, 'code2'))
            other = dict(row, sample_id='other:clip')
            self.assertNotEqual(key, cache_key(other, 'bytes1', {'stride': 2}, 'code1'))

    def test_partial_or_corrupt_shard_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'one.npz'
            np.savez(path, embeddings=np.ones((2, 3)), valid_mask=np.ones(2, dtype=bool))
            self.assertFalse(valid_shard(path, 'key'))
            write_json(path.with_suffix('.json'), {'cache_key': 'key', 'shard_sha256': sha256(path)})
            self.assertTrue(valid_shard(path, 'key'))
            self.assertFalse(valid_shard(path, 'other'))
            with path.open('ab') as stream:
                stream.write(b'corrupt')
            self.assertFalse(valid_shard(path, 'key'))

    def test_pdf_header_alone_is_not_complete_download_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'paper.pdf'
            path.write_bytes(b'%PDF-1.7\ntruncated content')
            with self.assertRaises(ValueError):
                validate_asset_bytes(path, {'kind': 'pdf'})
            path.write_bytes(b'%PDF-1.7\ncontent\n%%EOF')
            with self.assertRaises(ValueError):
                validate_asset_bytes(path, {'kind': 'pdf', 'expected_bytes': 100})


if __name__ == '__main__':
    unittest.main()
