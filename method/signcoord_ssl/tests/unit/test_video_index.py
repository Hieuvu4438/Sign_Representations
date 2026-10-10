import json

import numpy as np
import pytest

from signcoord.data.video_index import create_video_index,record_metadata,verify_media
from signcoord.provenance import sha256_file


def test_video_index_reuse_and_media_index_mutation_detection(tmp_path):
    import av
    video=tmp_path/'source.mp4'
    with av.open(str(video),'w') as out:
        stream=out.add_stream('libx264',rate=25);stream.width=32;stream.height=24;stream.pix_fmt='yuv420p'
        for t in range(8):
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.full((24,32,3),t*10,dtype=np.uint8),format='rgb24')):out.mux(packet)
        for packet in stream.encode():out.mux(packet)
    digest=sha256_file(video);path,pts,w,h=create_video_index(video,tmp_path/'index')
    assert (w,h)==(32,24) and len(pts)==8
    record={'media_path':str(video),'media_sha256':digest,'video_index_path':str(path),'video_index_sha256':sha256_file(path)}
    cached,_,_=record_metadata(record);assert np.array_equal(cached,pts)
    again,*_=create_video_index(video,tmp_path/'index',digest);assert again==path
    raw=path.read_bytes();path.write_bytes(raw+b'tamper')
    with pytest.raises(ValueError,match='index content'):record_metadata(record)
    path.write_bytes(raw);video.write_bytes(video.read_bytes()+b'media change')
    with pytest.raises(ValueError,match='manifest freeze'):record_metadata(record)
