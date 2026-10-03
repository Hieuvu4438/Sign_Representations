import tempfile
import unittest
from pathlib import Path

from signrepr.io import sha256, write_json
from signrepr.native_cache import verified_preprocessing


class NativeCacheIntegrity(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        source = root / 'input.mp4'
        source.write_bytes(b'source')
        clip = root / 'cache'
        clip.mkdir()
        stream = clip / 'face.mp4'
        stream.write_bytes(b'face')
        landmarks = clip / 'pose.json'
        landmarks.write_text('{}')
        row = {'sample_id': 'train:a', 'source_root': str(root), 'relative_path': source.name}
        identity = {'manifest': 'locked', 'implementation': 'v1'}
        audit = {'status': 'PASS_NATIVE_PREPROCESSING', 'sample_id': row['sample_id'],
                 'source_path': str(source), 'source_sha256': sha256(source),
                 'cache_identity': identity, 'streams': {'face': str(stream)},
                 'landmarks_path': str(landmarks),
                 'stream_sha256': {'face': sha256(stream), 'landmarks': sha256(landmarks)}}
        path = clip / 'audit.json'
        write_json(path, audit)
        return path, row, identity, audit, source, stream

    def test_resume_rejects_stale_input_and_corrupt_intermediate(self):
        with tempfile.TemporaryDirectory() as folder:
            path, row, identity, audit, source, stream = self.fixture(folder)
            self.assertEqual(verified_preprocessing(path, row, identity), audit)
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'input content'):
                verified_preprocessing(path, row, identity)
            source.write_bytes(b'source')
            stream.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verified_preprocessing(path, row, identity)

    def test_resume_rejects_changed_identity_failed_clip_and_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            path, row, identity, audit, source, stream = self.fixture(folder)
            with self.assertRaisesRegex(ValueError, 'identity'):
                verified_preprocessing(path, row, {'implementation': 'v2'})
            audit['status'] = 'FAIL'
            write_json(path, audit)
            with self.assertRaisesRegex(ValueError, 'failed clip'):
                verified_preprocessing(path, row, identity)
            audit['status'] = 'PASS_NATIVE_PREPROCESSING'
            outside = Path(folder) / 'outside.mp4'
            outside.write_bytes(b'face')
            audit['streams']['face'] = str(outside)
            write_json(path, audit)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                verified_preprocessing(path, row, identity)
