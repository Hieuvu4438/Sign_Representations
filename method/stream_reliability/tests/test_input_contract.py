"""Independent state/alignment/missingness fixtures for the adapted input contract."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from input_contract import AlignedNativeCrops


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.face = load('fixture_face', 'third_party/SHuBERT/dataset/crop_face.py')
        cls.hands = load('fixture_hands', 'third_party/SHuBERT/dataset/crop_hands.py')
        cls.body = load('fixture_body', 'third_party/SHuBERT/features/body_features.py')

    def setUp(self):
        self.adapter = AlignedNativeCrops(self.face, self.hands, self.body)
        self.rgb = np.full((80, 100, 3), [240, 20, 5], dtype=np.uint8)
        self.pose = [[.5, .5, 0.] for _ in range(33)]
        self.pose[11] = [.3, .5, 0.]; self.pose[12] = [.7, .5, 0.]
        self.pose[15] = [.5, .5, 0.]; self.pose[16] = [.54, .5, 0.]
        self.raw = dict(pose_landmarks=[self.pose], hand_landmarks=[[[.52, .5, 0.] for _ in range(21)]], face_landmarks=None)

    def test_initial_whole_row_missing_retains_one_frame_and14D(self):
        arrays, meta = self.adapter.step(self.rgb, None, 0)
        self.assertEqual(set(arrays), {'face', 'left_hand', 'right_hand', 'body_posture'})
        self.assertEqual(arrays['body_posture'].shape, (14,))
        np.testing.assert_array_equal(arrays['body_posture'], np.full(14, -9999.))
        self.assertEqual(self.adapter.next_frame, 1)
        self.assertTrue(all(not p['observed'] and p['source_frame_index'] is None for p in meta['streams'].values()))

    def test_observed_then_missing_retains_pixels_but_not_observed_mask(self):
        first, first_meta = self.adapter.step(self.rgb, self.raw, 0)
        second, meta = self.adapter.step(self.rgb, None, 1)
        np.testing.assert_array_equal(first['left_hand'], np.tile([240,20,5],(224,224,1)))
        np.testing.assert_array_equal(second['left_hand'], first['left_hand'])
        self.assertTrue(first_meta['streams']['left_hand']['observed'])
        self.assertFalse(meta['streams']['left_hand']['observed'])
        self.assertEqual(meta['streams']['left_hand']['source_frame_index'], 0)
        self.assertEqual(meta['streams']['left_hand']['fill_kind'], 'repeat_previous')

    def test_native_duplicate_assignment_preserved_and_flagged(self):
        arrays, meta = self.adapter.step(self.rgb, self.raw, 0)
        self.assertTrue(meta['same_detection'])
        self.assertEqual((meta['left_detection_index'], meta['right_detection_index']), (0, 0))
        np.testing.assert_array_equal(arrays['left_hand'], arrays['right_hand'])

    def test_frame_gap_rejected_without_state_advance(self):
        with self.assertRaises(ValueError): self.adapter.step(self.rgb, None, 1)
        self.assertEqual(self.adapter.next_frame, 0)
        self.assertEqual(self.adapter.previous, {})

    def test_collapsed_body_rejected_without_nan_or_state_commit(self):
        self.pose[12] = self.pose[11][:]
        with self.assertRaises(ValueError): self.adapter.step(self.rgb, self.raw, 0)
        self.assertEqual(self.adapter.next_frame, 0)
        self.assertEqual(self.adapter.previous, {})

    def test_malformed_face_error_not_silently_filled(self):
        self.raw['face_landmarks'] = [[[.5,.5,0.]]]
        with self.assertRaises(IndexError): self.adapter.step(self.rgb, self.raw, 0)
        self.assertEqual(self.adapter.next_frame, 0)
        self.assertEqual(self.adapter.previous, {})


if __name__ == '__main__': unittest.main(verbosity=2)
