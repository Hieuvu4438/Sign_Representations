"""Accounting regressions: immutable original, full charge, tamper/double rejection."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_accounting


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='shubert_budget_fixture_', dir='/dev/shm')
        self.root = Path(self.temp.name)
        self.previous = gpu_accounting.REPO
        gpu_accounting.REPO = self.root
        self.roots = ['method/' + d + '/runtime' for d in
                      ('frozen_probe', 'research_lab', 'stream_reliability')]
        for root in self.roots:
            (self.root / root).mkdir(parents=True)
        self.status = self.root / self.roots[2] / 'detector/status.json'
        self.status.parent.mkdir()
        self.status.write_text(json.dumps(dict(run_id='detector', H_id='H003', status='completed',
                                               GPU_hours=0, wall_seconds=10)))
        other = self.root / self.roots[0] / 'other/status.json'
        other.parent.mkdir()
        other.write_text(json.dumps(dict(run_id='other', H_id='H002', status='completed', GPU_hours=3/3600)))
        self.original = self.status.read_bytes()
        self.erratum = self.root / 'correction.json'
        self.correct = dict(charged_run_id='detector', H_id='H003',
                            original_status_path=str(self.status.relative_to(self.root)),
                            original_status_sha256=hashlib.sha256(self.original).hexdigest(),
                            original_GPU_hours=0, conservative_GPU_seconds=10)
        self.cfg = dict(runtime_roots=self.roots, cost_errata=[], H_id='H003',
                        program_ceiling_seconds=14400, hypothesis_ceiling_seconds=3600)
        self.bind(self.correct)

    def bind(self, value):
        self.erratum.write_text(json.dumps(value))
        self.cfg['cost_errata'] = [dict(path='correction.json',
                                      sha256=hashlib.sha256(self.erratum.read_bytes()).hexdigest())]

    def tearDown(self):
        gpu_accounting.REPO = self.previous
        self.temp.cleanup()

    def test_whole_wall_charge_once_and_original_preserved(self):
        result = gpu_accounting.inspect_effective_budget(self.cfg)
        self.assertEqual(result['project_used_seconds'], 13)
        self.assertEqual(result['hypothesis_used_seconds'], 10)
        self.assertEqual(self.status.read_bytes(), self.original)

    def test_original_status_tamper_rejected(self):
        self.status.write_bytes(self.original + b'\n')
        with self.assertRaises(ValueError):
            gpu_accounting.inspect_effective_budget(self.cfg)

    def test_duplicate_charge_rejected(self):
        self.cfg['cost_errata'] *= 2
        with self.assertRaises(ValueError):
            gpu_accounting.inspect_effective_budget(self.cfg)

    def test_incomplete_wall_charge_rejected(self):
        self.bind(dict(self.correct, conservative_GPU_seconds=9))
        with self.assertRaises(ValueError):
            gpu_accounting.inspect_effective_budget(self.cfg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
