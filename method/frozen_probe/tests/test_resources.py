import unittest
from resources import reserve_check, normalized_uuid


class ResourceTests(unittest.TestCase):
    def test_pytorch_nvml_uuid_format_oracle(self):
        observed = '3775c606-49da-34fe-322c-756749074f5b'
        self.assertEqual(normalized_uuid('GPU-' + observed), observed)
        self.assertEqual(normalized_uuid(observed.upper()), observed)
        self.assertNotEqual(normalized_uuid('4775c606-49da-34fe-322c-756749074f5b'), observed)
        for invalid in ('GPU-fixture', '', '3775', 'GPU-GPU-' + observed):
            with self.assertRaises(ValueError): normalized_uuid(invalid)

    def test_reserve_boundary_and_corrupt_observation(self):
        self.assertTrue(reserve_check(1000, 300, 100))
        self.assertFalse(reserve_check(1000, 300, 101))
        self.assertFalse(reserve_check(1000, 19, 0))
        for args in ((0, 0, 0), (100, 101, 0), (100, -1, 0), (100, 50, -1)):
            with self.assertRaises(ValueError): reserve_check(*args)
