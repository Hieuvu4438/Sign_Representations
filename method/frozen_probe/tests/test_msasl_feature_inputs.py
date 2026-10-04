"""Independent RGB bytes/window/pooling oracles for actual extraction risks."""
from fractions import Fraction
import hashlib
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from msasl_feature_inputs import collect_native_inputs
from temporal import window_plan
from window_budget import pooled_vector
from pooling import activity_mean


def run_tests():
    raw=list(range(22));raw[6]=3;best=list(range(22));tb=Fraction(1,10)
    ref=dict(header=dict(time_base=str(tb),start_pts=0,width=8,height=4),raw_ticks=raw,best_ticks=best)
    selected=list(range(1,22));payload=b''.join(bytes([i])*(224*224*3) for i in selected)
    item=dict(raw_path='fixture',source_sha256='a'*64,start_time_seconds=.1,end_time_seconds=2.2,box_yxyx=[.25,.25,.75,.75],
              time_base=str(tb),origin_pts=0,selected_native_frames=21,crop_yxyx_pixels=[1,2,3,6],sample_id='fixture',input_contract='MSASL_ESTCLK_v1',
              clock_pts_ticks_sha256=hashlib.sha256(b''.join(i.to_bytes(8,'little',signed=True) for i in selected)).hexdigest(),
              all_resized_RGB_sha256=hashlib.sha256(payload).hexdigest(),
              first_native16_RGB_sha256=hashlib.sha256(b''.join(bytes([i])*(224*224*3) for i in selected[:16])).hexdigest())
    class Frame:
        def __init__(self,i):self.pts=raw[i];self.time_base=tb;self.is_corrupt=False;self.width=8;self.height=4;self.i=i
        def to_ndarray(self,format):
            assert format=='rgb24';return np.full((4,8,3),self.i,dtype=np.uint8)
    class Container:
        def __init__(self):self.streams=SimpleNamespace(video=[SimpleNamespace(time_base=tb,start_time=0,codec_context=SimpleNamespace(thread_count=0))])
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def seek(self,*_,**__):pass
        def decode(self,**_):return iter(Frame(i) for i in range(22))
    def collect(x):
        with patch('msasl_feature_inputs.av.open',return_value=Container()):return collect_native_inputs(x,'fixture_ffprobe',reference=ref)
    frames,pts,info=collect(item)
    assert [int(f[0,0,0]) for f in frames]==selected and pts==[i/10 for i in selected]
    assert len(frames)==21 and info['complete_stride2_windows']==3
    plan=window_plan(pts);assert [x['start_frame'] for x in plan]==[0,2,4]
    assert [x['stop_frame_exclusive'] for x in plan]==[16,18,20] and len(frames)-plan[-1]['stop_frame_exclusive']==1
    for key,value in (('clock_pts_ticks_sha256','0'*64),('all_resized_RGB_sha256','0'*64),('first_native16_RGB_sha256','0'*64),
                      ('selected_native_frames',22),('crop_yxyx_pixels',[0,0,4,8])):
        try:collect(dict(item,**{key:value}))
        except ValueError:pass
        else:raise AssertionError('Input identity mismatch accepted '+key)
    try:collect(dict(item,selected_native_frames=15,end_time_seconds=1.6))
    except ValueError:pass
    else:raise AssertionError('Short input padded')
    x=np.array([[1,0],[0,1]],dtype=np.float32);a=np.array([[-1,.25],[2,0]],dtype=np.float32)
    weighted,metadata=activity_mean(x,a)
    np.testing.assert_allclose(weighted,[.2,.8],atol=1e-15,rtol=0)
    np.testing.assert_allclose(pooled_vector(x,a,'full_activity',[0,1]),[1/np.sqrt(17),4/np.sqrt(17)],atol=1e-15,rtol=0)
    assert metadata['weight_sum']==1.25 and not metadata['fallback_mean']
    value,metadata=activity_mean(x,np.zeros((2,2),dtype=np.float32))
    assert metadata['fallback_mean'] and np.array_equal(value,[.5,.5])
    np.testing.assert_allclose(pooled_vector(x,np.zeros((2,2)),'full_activity',[0,1]),[1/np.sqrt(2)]*2,atol=1e-15,rtol=0)
    for features,activity in ((np.zeros((2,2)),a),(x,np.array([[np.nan,0],[1,0]]))):
        try:pooled_vector(features,activity,'full_activity',[0,1])
        except ValueError:pass
        else:raise AssertionError('Zero/nonfinite output accepted')
    return dict(passed=True,groups=['manual_RGB_clock_ordinals','dense_native16_stride2_odd_tail',
            'input_hash_count_crop_and_short_guards','manual_clipped_max_activity_FP64_L2',
            'zero_weight_fallback_and_nonfinite_zero_norm_guards'],GPU_hours=0,
            scope='Synthetic CPU software; preserve raw activity clipping, no softmax/subsampling/newtaskquality')
