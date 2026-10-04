import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from budget import budget_from_statuses, inspect_budget
from run_context import RunContext


def attempt(name, seconds, hypothesis='H001', state='completed'):
    return dict(run_id=name, GPU_hours=seconds / 3600, H_id=hypothesis, status=state)


class BudgetStatusTests(unittest.TestCase):
    def test_all_attempts_count_hand_oracle(self):
        statuses = [attempt('failed', 120, state='failed'), attempt('done', 30),
                    attempt('stopped', 60, state='interrupted'), attempt('other', 30, 'H002'),
                    attempt('invalid', 10, 'H002', 'invalidated'), attempt('cpu', 0)]
        result = budget_from_statuses(statuses)
        self.assertEqual(result['hypothesis_used_seconds'], 210)
        self.assertEqual(result['project_used_seconds'], 250)
        self.assertEqual(result['remaining_seconds'], 3390)
        self.assertEqual(len(result['counted_attempts']), 5)

    def test_reallocation_remains_under_project_ceiling(self):
        statuses = [attempt('previous', 600, 'H001'), attempt('other', 12000, 'H002')]
        result = budget_from_statuses(statuses, hypothesis_limit_seconds=7200)
        self.assertEqual(result['hypothesis_used_seconds'], 600)
        self.assertEqual(result['remaining_seconds'], 1800)

    def test_bad_cost_exhaustion_and_unreconciled_attempt(self):
        for hours in (None, -1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                budget_from_statuses([dict(run_id='bad', GPU_hours=hours, status='failed')])
        with self.assertRaises(ValueError):
            budget_from_statuses([dict(run_id='bad', GPU_hours=.1, status='completed')])
        for status in (attempt('active', 1, state='running'),
                       dict(run_id='active', GPU_hours=0, GPU_allocated=True, H_id='H001', status='running')):
            with self.assertRaises(RuntimeError): budget_from_statuses([status])
        with self.assertRaises(RuntimeError): budget_from_statuses([attempt('exhausted', 3600)])
        with self.assertRaises(RuntimeError): budget_from_statuses([attempt('exhausted', 14400, 'H002')])
        with self.assertRaises(ValueError): budget_from_statuses([attempt('same', 1), attempt('same', 1)])

    def test_disk_ledger_and_terminal_status_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            (source / 'fixture.py').write_text('x = 1\n')
            config = root / 'config.json'
            config.write_text(json.dumps(dict(seed=1337, protocol='P001_v3', H_id='H001', timeout_seconds=60)))
            output = root / 'runtime' / 'good'
            with patch('run_context.ROOT', source), patch('run_context.signal.alarm'):
                with RunContext(output, 'good', config, 1337) as run:
                    run.start_gpu(budget_from_statuses([]))
                    current = json.loads((output / 'status.json').read_text())
                    self.assertTrue(current['GPU_allocated'])
                    self.assertEqual(current['status'], 'running')
                    self.assertLessEqual(current['gpu_deadline_seconds'], 60)
                    self.assertGreater(current['gpu_deadline_seconds'], 0)
                status = json.loads((output / 'status.json').read_text())
                self.assertEqual(status['status'], 'completed')
                self.assertGreaterEqual(status['GPU_hours'], 0)
                self.assertEqual(inspect_budget([root / 'runtime'])['hypothesis_used_seconds'], status['GPU_hours'] * 3600)
                with self.assertRaises(FileExistsError): RunContext(output, 'good', config, 1337)
                with self.assertRaises(ValueError):
                    with RunContext(root / 'runtime' / 'failed', 'failed', config, 1337):
                        raise ValueError('fixture runtime failure')
                self.assertEqual(json.loads((root / 'runtime/failed/status.json').read_text())['status'], 'failed')

    def test_setup_failure_has_durable_terminal_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps(dict(seed=1337, protocol='P001_v3', timeout_seconds=60)))
            output = root / 'output'
            with patch('run_context.RunContext._snapshot', side_effect=OSError('fixture setup failure')):
                with self.assertRaises(OSError): RunContext(output, 'setup_fail', config, 1337)
            status = json.loads((output / 'status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(status['GPU_hours'], 0)
            self.assertTrue((output / 'error.txt').exists())

    def test_pre_cuda_setup_is_subtracted_from_job_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps(dict(seed=1337, protocol='P001_v3', H_id='H001', timeout_seconds=60)))
            with patch('run_context.ROOT', root / 'empty_source'), patch('run_context.signal.alarm') as alarm:
                with patch('run_context.time.monotonic', return_value=100):
                    run = RunContext(root / 'output', 'deadline_fixture', config, 1337)
                with patch('run_context.time.monotonic', return_value=112.5):
                    run.start_gpu(budget_from_statuses([]))
                self.assertEqual(run.status['gpu_deadline_seconds'], 47)
                alarm.assert_called_with(47)
                with patch('run_context.time.monotonic', return_value=120):
                    run.__exit__(None, None, None)
                self.assertAlmostEqual(run.status['GPU_hours'] * 3600, 7.5)
