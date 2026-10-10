import json

import av
import numpy as np
import pytest

from signcoord.data.dataset import SSLVideoDataset
from signcoord.provenance import sha256_file
from signcoord.targets.schema import SCHEMA_HASH


@pytest.mark.parametrize('policy',['verified','track_change','unknown','out_of_frame'])
def test_source_gap_pipeline_only_interpolates_verified_continuous_tracks(tmp_path,policy):
    video=tmp_path/'video.mp4'
    with av.open(str(video),'w') as out:
        stream=out.add_stream('libx264',rate=25);stream.width=stream.height=32;stream.pix_fmt='yuv420p'
        for _ in range(8):
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.zeros((32,32,3),dtype=np.uint8),format='rgb24')):out.mux(packet)
        for packet in stream.encode():out.mux(packet)
    xy=np.full((8,133,2),16,dtype=np.float32);xy[:,5]=[8,12];xy[:,6]=[24,12]
    xy[:,9,0]=16+np.arange(8);observed=np.ones((8,133),dtype=bool);observed[3:5,9]=False
    xy[3:5,9]=np.nan;confidence=np.full((8,133),.9,dtype=np.float32);confidence[3:5,9]=0
    status=np.where(observed,0,2).astype(np.uint8)
    ids=np.zeros(8,dtype=np.int64)
    if policy=='track_change':ids[3:5]=-1
    if policy=='out_of_frame':status[3:5,9]=3
    meta={'schema_version':'pose_cache_v1','schema_hash':SCHEMA_HASH,'media_sha256':sha256_file(video),
          'tracking_identity_verified':policy!='unknown'}
    np.savez_compressed(tmp_path/'pose.npz',xy_px=xy,confidence=confidence,observed=observed,status=status,
                        pts_s=np.arange(8)/25,metadata=json.dumps(meta),track_ids=ids)
    record={'schema_version':'video_manifest_v1','sample_id':'example','dataset_id':'fixture','source_video_id':'example',
            'source_group_id':'example','media_path':str(video),'media_sha256':sha256_file(video),
            'start_ms':0,'end_ms':320,'signer_id':None,'split':'ssl_train','pose_cache_key':'pose.npz','license_ref':'fixture'}
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(json.dumps(record)+'\n')
    ds=SSLVideoDataset(manifest,tmp_path,frames=8,spatial_size=(32,32))
    sample=ds[0]
    if policy=='verified':
        assert sample['audit']['interpolated_source_points']==2
        assert float(sample['targets']['reliability'][3,115])==pytest.approx(.45)
        assert float(sample['targets']['global'][3,115,0])==pytest.approx(3/16)
    else:
        assert sample['audit']['interpolated_source_points']==0
        assert sample['targets']['reliability'][3,115]==0
