from fractions import Fraction
import hashlib
import numpy as np
import pytest
from signrepr.data.extract_pose import extract_video
from signrepr.data.schema import load_pose
from signrepr.data.fixture import fixture_pose


def test_streaming_variable_pts_missing_detection_and_raw_score_audit(tmp_path):
    av = pytest.importorskip('av')
    video = tmp_path/'source.mp4'
    with av.open(str(video), 'w') as container:
        stream = container.add_stream('mpeg4', rate=25)
        stream.width = stream.height = 64; stream.pix_fmt = 'yuv420p'
        stream.time_base = Fraction(1, 25)
        for i, pts in enumerate([0, 1, 3, 5]):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i*40, np.uint8), format='bgr24')
            frame.pts = pts; frame.time_base = Fraction(1, 25)
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    class MockDetector:
        def __init__(self): self.index = 0
        def __call__(self, frame):
            i = self.index; self.index += 1
            if i == 2: return np.empty((0, 133, 2), np.float32), np.empty((0, 133), np.float32)
            return fixture_pose(1)['xy']*64, np.full((1, 133), 9., np.float32)
    metadata = {'score_profile': 'simcc_positive_rational_v1', 'extractor_profile': 'synthetic_mock_for_pipeline_test',
                'extractor_weights_sha256': hashlib.sha256(b'mock specification, no real detector weights').hexdigest()}
    result = extract_video(video, tmp_path/'pose.npz', metadata, detector=MockDetector())
    pose = load_pose(tmp_path/'pose.npz')
    assert result['num_frames'] == 4 and result['timing_source'] == 'decoded_source_frame_pts'
    np.testing.assert_allclose(pose['timestamps_sec'], [0., .04, .12, .20], atol=1e-9)
    assert pose['timing_known'] and not pose['observed'][2].any()
    assert pose['track_continuity'].tolist() == [True, False, False]
    np.testing.assert_allclose(pose['confidence'][[0, 1, 3]], .9)
    with np.load(result['raw_detector_scores_path']) as raw:
        assert raw['frame_person_offsets'].tolist() == [0, 1, 2, 2, 3]
        assert raw['detector_scores'].max() == 9
