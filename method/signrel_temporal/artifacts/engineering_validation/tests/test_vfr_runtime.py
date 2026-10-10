from fractions import Fraction
import numpy as np
import pytest
from signrel.data.decode import iter_grid_frames


def test_actual_irregular_pts_video(tmp_path):
    av=pytest.importorskip('av')
    path=tmp_path/'irregular.mkv'
    with av.open(str(path),'w') as container:
        stream=container.add_stream('ffv1')
        stream.width=stream.height=32
        stream.pix_fmt='bgr0'
        stream.time_base=Fraction(1,1000)
        stream.codec_context.time_base=Fraction(1,1000)
        for j,pts in enumerate([0,33,90,250]):
            frame=av.VideoFrame.from_ndarray(np.full((32,32,3),j*50,np.uint8),format='rgb24')
            frame.pts=pts;frame.time_base=Fraction(1,1000)
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    rows=list(iter_grid_frames(path))
    assert [r.grid_time_ms for r in rows]==[0,40,80,120,160,200,240]
    assert [r.valid for r in rows]==[True,True,True,True,False,False,True]
    assert [r.source_frame_index for r in rows]==[0,1,2,2,2,3,3]
