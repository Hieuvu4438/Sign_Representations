import io
import unittest
import numpy as np
from bounded_artifacts import array_spec
from native_pts_artifact import native_pts_bytes


class NativePTSArtifactTests(unittest.TestCase):
    def test_variable_vector_roundtrip_cap_and_rejections(self):
        for count in (16,84,240):
            values=np.arange(count,dtype='<f8')/30.
            raw=native_pts_bytes(values)
            np.testing.assert_array_equal(np.load(io.BytesIO(raw),allow_pickle=False),values)
            self.assertEqual(len(raw),array_spec((count,),'<f8')['file_bytes'])
            self.assertLessEqual(len(raw),array_spec((240,),'<f8')['file_bytes'])
        for values in (np.arange(15),np.arange(241),np.zeros(16),np.full(16,float('nan')),np.ones((16,1))):
            with self.assertRaises(ValueError):native_pts_bytes(values)


if __name__=='__main__':unittest.main()
