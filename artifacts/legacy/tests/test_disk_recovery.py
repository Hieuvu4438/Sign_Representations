import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import recover_native_common_disk_stop as recovery
from signrepr.io import sha256


class DiskRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        def write(path, data):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        protocol = root / 'configs/protocol_native_common_v1.json'
        index = root / 'features/shubert_native_common_v1/index.jsonl'
        write(protocol, {}); write(index, {})
        write(root / 'reports/native_common_disk_recovery_audit.json', {
            'status': 'PASS_EXISTING_CACHE_IDENTITIES_SOURCE_STREAM_AND_SHARD_HASHES',
            'native_protocol_sha256': sha256(protocol), 'feature_index_sha256': sha256(index)})
        for name in ['shubert_native_common_v1', 'shubert_native_common_v1_preprocessing']:
            write(root / 'features' / name / 'status.json', {'status': 'FAIL', 'failure_reason': 'reserved disk limit'})
        write(root / 'runs/fair_common_v1/status.json', {'status': 'FAIL'})
        return patch.multiple(recovery, ROOT=root, OUTPUT=root / 'state/recovery', FAIR_OUTPUT=root / 'runs/new_attempt')

    def test_capacity_wait_starts_no_cpu_or_gpu_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.fixture(root), patch.object(recovery, 'free_gib', return_value=13), \
                    patch.object(recovery, 'stage') as stage, patch.object(recovery.time, 'sleep', side_effect=RuntimeError('bounded test end')):
                with self.assertRaisesRegex(RuntimeError, 'bounded test end'):
                    recovery.main()
                stage.assert_not_called()
            self.assertEqual(json.loads((root / 'features/shubert_native_common_v1/status.json').read_text())['status'], 'FAIL')

    def test_incomplete_cpu_stage_stops_before_gpu_and_never_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def incomplete(command, log, environment):
                self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '')
                self.assertIn('scripts/prepare_shubert_cohort.py', command)
                (root / 'features/shubert_native_common_v1_preprocessing/status.json').write_text(json.dumps({
                    'status': 'PARTIAL_BOUNDED_VALIDATION', 'passed': 1, 'samples': 2}))
            with self.fixture(root), patch.object(recovery, 'free_gib', return_value=61), patch.object(recovery, 'stage', side_effect=incomplete) as stage:
                with self.assertRaisesRegex(ValueError, 'Complete CPU preprocessing required'):
                    recovery.main()
                self.assertEqual(stage.call_count, 1)
            self.assertEqual(json.loads((root / 'state/recovery/status.json').read_text())['status'], 'FAIL')
            self.assertFalse((root / 'runs/new_attempt').exists())


if __name__ == '__main__':
    unittest.main()
