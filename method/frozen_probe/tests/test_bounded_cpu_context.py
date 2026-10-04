"""Integration checks for successful and failed CPU runs under complete caps."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import unittest

import numpy as np
from bounded_artifacts import array_spec, verify_store
from bounded_cpu_context import BoundedCPUContext

ROOT = Path(os.environ['SIGNREP_CONTEXT_FIXTURES'])
CODE = Path('method/frozen_probe')


class ContextTests(unittest.TestCase):
    def config(self):
        directory = ROOT / self._testMethodName
        directory.mkdir(exist_ok=False)
        sources = {f'source_snapshot/{name}': str(CODE / name) for name in (
            'bounded_cpu_context.py', 'bounded_artifacts.py', 'cache.py')}
        caps = {name: Path(path).stat().st_size for name, path in sources.items()}
        caps.update({'status.json': 16384, 'error.txt': 16384, 'config.json': 16384,
                     'environment.json': 16384, 'combined_log.txt': 1024, 'report.json': 1024,
                     'array.npy': array_spec((2, 3), '<f8')['file_bytes']})
        config = dict(protocol='ARTIFACTCTX001_synthetic', seed=1701, timeout_seconds=30,
                      GPU_allocation_seconds=0, artifact_ceiling_bytes=262144,
                      artifact_manifest_cap=16384, artifact_file_caps=caps, source_snapshots=sources,
                      input_hashes={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sources.values()})
        path = directory / 'fixture_config.json'
        path.write_text(json.dumps(config, indent=2) + '\n')
        return directory / 'run', path, config

    def test_complete_context_records_exact_argv_sources_logs_and_arrays(self):
        output, config, _ = self.config()
        with contextlib.redirect_stdout(io.StringIO()):
            with BoundedCPUContext(output, 'synthetic-success', config, 1701) as run:
                print('bounded body log')
                run.save('report.json', {'fixture': True})
                run.save_array('array.npy', np.arange(6, dtype='<f8').reshape(2, 3), array_spec((2, 3), '<f8'))
        status = json.loads((output / 'status.json').read_text())
        self.assertEqual(status['status'], 'completed')
        self.assertEqual(status['exit_code'], 0)
        self.assertTrue(verify_store(output)['passed'])
        self.assertEqual((output / 'combined_log.txt').read_text(), 'bounded body log\n')
        self.assertEqual((output / 'config.json').read_bytes(), config.read_bytes())
        self.assertEqual((output / 'source_snapshot/bounded_cpu_context.py').read_bytes(), (CODE / 'bounded_cpu_context.py').read_bytes())
        for name, digest in status['output_hashes'].items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
        self.assertEqual((output / 'error.txt').read_bytes(), b'')

    def test_overbudget_context_refuses_before_run_directory(self):
        output, path, config = self.config()
        config['artifact_ceiling_bytes'] = 1
        path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            BoundedCPUContext(output, 'overbudget', path, 1701)
        self.assertFalse(output.exists())

    def test_missing_declared_artifact_becomes_terminal_failure(self):
        output, config, _ = self.config()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            with BoundedCPUContext(output, 'missing-array', config, 1701) as run:
                run.save('report.json', {'fixture': True})
        status = json.loads((output / 'status.json').read_text())
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['exit_code'], 1)
        self.assertIn('array.npy', status['missing_artifacts'])
        self.assertIn('omitted declared artifacts', (output / 'error.txt').read_text())
        self.assertFalse((output / 'artifact_manifest.json').exists())

    def test_oversized_report_preserves_exception_and_resets_alarm_handlers(self):
        output, config, _ = self.config()
        previous = signal.getsignal(signal.SIGALRM)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            with BoundedCPUContext(output, 'oversized-report', config, 1701) as run:
                run.save('report.json', {'too_large': 'x' * 4096})
        status = json.loads((output / 'status.json').read_text())
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertIn('registered bound', (output / 'error.txt').read_text())
        self.assertFalse((output / 'report.json').exists())
        self.assertFalse(status['error_trace_truncated'])

    def test_bounded_log_overflow_is_terminal_and_original_bytes_retained(self):
        output, config, _ = self.config()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            with BoundedCPUContext(output, 'oversized-log', config, 1701):
                print('small log')
                print('x' * 4096)
        self.assertEqual(json.loads((output / 'status.json').read_text())['status'], 'failed')
        self.assertEqual((output / 'combined_log.txt').read_text(), 'small log\n')
        self.assertIn('log bound exhausted', (output / 'error.txt').read_text())


if __name__ == '__main__':
    unittest.main()
