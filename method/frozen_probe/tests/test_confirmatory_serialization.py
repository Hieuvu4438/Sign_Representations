import json
import unittest

from confirmatory import analyze
import test_confirmatory


class ConfirmatorySerializationTests(unittest.TestCase):
    def test_full_analysis_roundtrips_strict_json_without_changing_oracle(self):
        data = test_confirmatory.ConfirmatoryTests().data()
        result = analyze(*data)
        decoded = json.loads(json.dumps(result, allow_nan=False))
        self.assertTrue(decoded['joint_confirmatory_screen'])
        self.assertEqual(decoded['mean_metrics']['uniform8']['dcg'], 1)
        self.assertEqual(decoded['costs']['uniform8']['window_reduction_factor'], 4)
        self.assertEqual(decoded['costs']['uniform8']['median_paired_query_ratio'], .5)
        self.assertIs(type(result['costs']['uniform8']['actual_cost_screen']), bool)
        self.assertFalse(decoded['costs']['full_activity']['actual_cost_screen'])
