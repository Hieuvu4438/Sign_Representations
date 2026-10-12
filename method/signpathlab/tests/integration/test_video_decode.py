from fractions import Fraction
import numpy as np
import pytest
from signpath.data.decode import decode_video, nearest_observations


def test_real_container_vfr_pts_rgb_and_aliases(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "vfr.mkv"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "bgr0"
        stream.time_base = Fraction(1, 1000)
        for pts in (0, 40, 120, 200):
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            rgb[..., 0] = 255  # catches BGR/RGB inversion
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    decoded = decode_video(path)
    np.testing.assert_allclose(decoded.timestamps, [0, .04, .12, .2], atol=1e-9)
    np.testing.assert_array_equal(decoded.source_frame_index, np.arange(4))
    assert not decoded.timestamp_fallback
    assert decoded.frames[0, 0, 0, 0] == 255 and decoded.frames[0, 0, 0, 2] == 0
    selected = nearest_observations(decoded.timestamps, decoded.source_frame_index, 25)
    assert selected.duplicate_selection.any()
    cropped = decode_video(path, start_s=.04, end_s=.2)
    np.testing.assert_allclose(cropped.timestamps, [.04, .12])
    np.testing.assert_array_equal(cropped.source_frame_index, [1, 2])
