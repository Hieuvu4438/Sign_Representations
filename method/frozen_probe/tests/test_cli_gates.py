import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from resources import gpu_preflight

ROOT = Path(__file__).resolve().parents[1]


class CLIGateTests(unittest.TestCase):
    def test_cli_help_without_cuda_or_output(self):
        for name in ('gpu_smoke.py', 'extract_cache.py', 'run_baseline.py', 'analyze_baseline.py'):
            result = subprocess.run([sys.executable, str(ROOT / 'scripts' / name), '--help'],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            for option in ('--config', '--seed', '--run-id', '--output-dir'):
                self.assertIn(option, result.stdout)

    def test_baseline_refuses_missing_gate_and_extraction_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            common = ['--seed', '1337', '--run-id', 'fixture', '--output-dir', str(output)]
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_baseline.py'),
                '--config', str(ROOT / 'configs/BASE001.json'), *common,
                '--cache-manifest', str(Path(directory) / 'missing-cache.json'),
                '--launch-gate', str(Path(directory) / 'missing-gate.json')],
                capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('FileNotFoundError', result.stderr)
            self.assertFalse(output.exists())
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/extract_cache.py'),
                '--config', str(ROOT / 'configs/CACHE001.json'), *common,
                '--smoke-run', directory, '--resume'], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('resume is unsupported', result.stderr)
            self.assertFalse(output.exists())

    def test_gpu_busy_and_idle_observation_without_device_initialization(self):
        for free, utilization, passed in ((9000, 0, True), (1622, 99, False), (9000, 11, False)):
            result = subprocess.CompletedProcess([], 0, stdout=f'0, GPU-fixture, 49140, {free}, {utilization}\n')
            with patch('resources.subprocess.run', return_value=result):
                if passed:
                    self.assertEqual(gpu_preflight()['uuid'], 'GPU-fixture')
                else:
                    with self.assertRaises(RuntimeError): gpu_preflight()
