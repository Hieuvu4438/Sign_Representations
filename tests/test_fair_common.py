import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_fair_common


class FairCommonTests(unittest.TestCase):
    def test_missing_internal_frames_keep_positions_and_are_zeroed(self):
        samples = [{'tokens': np.full((4, 768), 999., dtype=np.float32),
                    'valid': np.asarray([True, False, False, True]), 'label': 1}]
        x, mask, y = run_fair_common.collate(samples, [0])
        self.assertEqual(tuple(x.shape), (1, 4, 768))
        np.testing.assert_array_equal(mask.numpy(), [[1, 0, 0, 1]])
        self.assertEqual(float(x[0, 1:3].sum()), 0.)
        self.assertEqual(int(y[0]), 1)

    def test_partial_or_failed_native_cache_cannot_be_declared_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / 'extraction_report.json').write_text(json.dumps({'status': 'PARTIAL_BOUNDED_VALIDATION'}))
            with patch.object(run_fair_common, 'NATIVE', p):
                (p / 'status.json').write_text(json.dumps({'status': 'RUNNING'}))
                self.assertFalse(run_fair_common.native_ready()[0])
                (p / 'status.json').write_text(json.dumps({'status': 'FAIL'}))
                with self.assertRaises(ValueError):
                    run_fair_common.native_ready()


if __name__ == '__main__':
    unittest.main()
