import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('audit_video', Path(__file__).resolve().parents[1]/'scripts/audit_video.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class TimelineTests(unittest.TestCase):
    def test_threshold_roundoff_and_large_jitter(self):
        self.assertFalse(audit.has_material_jitter([.0329999999999995, .0340000000000007]))
        self.assertTrue(audit.has_material_jitter([.033, .040]))
        self.assertFalse(audit.has_material_jitter([]))
