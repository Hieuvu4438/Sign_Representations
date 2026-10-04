import unittest
from video_contract import fingerprint_indices, summarize_timeline, dhash64


class VideoContractTests(unittest.TestCase):
    def test_boundary_and_quantile_oracles(self):
        self.assertEqual(fingerprint_indices(83,[0,.25,.5,.75,1]),[0,20,41,61,82])
        result=summarize_timeline([i/30 for i in range(83)],83)
        self.assertEqual(result['windows'],34)
        self.assertEqual(result['uncovered_tail_frames'],1)
        self.assertFalse(result['end_boundary_estimated'])
        self.assertTrue(summarize_timeline([i/30 for i in range(20)],20)['end_boundary_estimated'])

    def test_count_mismatch_rejected(self):
        with self.assertRaises(ValueError): summarize_timeline(list(range(20)),19)
        with self.assertRaises(ValueError): fingerprint_indices(20,[float('nan')])

    def test_dhash_direction_oracle(self):
        import numpy as np
        rising=np.tile(np.arange(9,dtype=np.uint8),(8,1))
        self.assertEqual(dhash64(rising),'ffffffffffffffff')
        self.assertEqual(dhash64(rising[:,::-1]),'0000000000000000')
        with self.assertRaises(ValueError):dhash64(rising[:,:8])


if __name__=='__main__': unittest.main()
