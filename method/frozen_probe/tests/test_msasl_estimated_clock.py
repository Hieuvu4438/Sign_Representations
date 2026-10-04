"""Manual tests for clock boundaries, ordinal/pixel joins and no-padding behavior."""
from fractions import Fraction
import hashlib
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from msasl_estimated_clock import select_estimated_ordinals,scan_estimated_interval


def run_tests():
    pick=select_estimated_ordinals
    # Nonzero origin, backwards preroll allowed only while all times precede start.
    assert pick([100,102,101,104,105,106],[100,102,101,104,105,106],'1/10',100,'.4','.6')==[3,4]
    # Raw duplicate/backwards ticks kept at distinct ordinals; estimated boundary shifts one frame.
    assert pick([100,102,101,104],[100,102,103,104],'1/10',100,'.2','.4')==[1,2]
    assert pick([100,101,102],[100,101,102],'1/10',100,'.1','.2')==[1]
    assert pick([1],[1],'1',0,'2','3')==[]
    for raw,best in (([1,2,3],[1,2,2]),([0,1,2,3],[0,2,0,3]),([0,1,2],[5,1,2])):
        try:pick(raw,best,'1',0,'1','3')
        except ValueError:pass
        else:raise AssertionError('Disorder/reentry/partition accepted')
    for raw,best in (([],[]),([1],[1,2]),([1.0],[1])):
        try:pick(raw,best,'1',0,'0','2')
        except ValueError:pass
        else:raise AssertionError('Bad arrays accepted')
    class Frame:
        def __init__(self,i,pts):self.pts=pts;self.time_base=Fraction(1);self.is_corrupt=False;self.width=8;self.height=4;self.i=i
        def to_ndarray(self,format):
            assert format=='rgb24';return np.full((4,8,3),self.i,dtype=np.uint8)
    class Container:
        def __init__(self,raw):
            self.frames=[Frame(i,t) for i,t in enumerate(raw)]
            self.streams=SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1),start_time=0,codec_context=SimpleNamespace(thread_count=0))])
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def seek(self,*args,**kwargs):pass
        def decode(self,**_):return iter(self.frames)
    raw=list(range(18));raw[5]=3;best=list(range(18))
    row=dict(start_time=1,end_time=8,box=[.25,.25,.75,.75])
    reference=dict(header=dict(time_base='1',start_pts=0,width=8,height=4),raw_ticks=raw,best_ticks=best)
    hashes=[hashlib.sha256(bytes([i])*(4*8*3)).hexdigest() for i in range(18)]
    with patch('msasl_estimated_clock.av.open',return_value=Container(raw)):
        out=scan_estimated_interval('fixture',row,reference,expected_rgb_sha256=hashes)
    assert out['selected_decoded_ordinals']==list(range(1,8)) and out['selected_raw_ticks']==raw[1:8]
    assert out['selected_estimated_ticks']==best[1:8] and out['crop_yxyx_pixels']==(1,2,3,6)
    assert not out['native16_eligible'] and out['complete_stride2_windows']==0 and out['first_native16_RGB_sha256'] is None
    oracle=b''.join(bytes([i])*(224*224*3) for i in range(1,8))
    assert out['all_resized_RGB_sha256']==hashlib.sha256(oracle).hexdigest()
    # Full16 frames with an outside suffix discrepancy must retain every selected ordinal.
    row=dict(start_time=0,end_time=8,box=[.25,.25,.75,.75])
    reference['header']['time_base']='1/2';best=list(range(18));reference['best_ticks']=best
    class HalfContainer(Container):
        def __init__(self,raw):
            super().__init__(raw);self.streams.video[0].time_base=Fraction(1,2)
            for f in self.frames:f.time_base=Fraction(1,2)
    outside=raw.copy();outside[17]=999
    with patch('msasl_estimated_clock.av.open',return_value=HalfContainer(outside)):
        out=scan_estimated_interval('fixture',row,reference,expected_rgb_sha256=hashes)
    assert out['native16_eligible'] and out['selected_native_frames']==16 and out['complete_stride2_windows']==1
    assert out['first_native16_RGB_sha256']==hashlib.sha256(b''.join(bytes([i])*(224*224*3) for i in range(16))).hexdigest()
    wrong=raw.copy();wrong[5]=555
    with patch('msasl_estimated_clock.av.open',return_value=HalfContainer(wrong)):
        try:scan_estimated_interval('fixture',row,reference,expected_rgb_sha256=hashes)
        except ValueError:pass
        else:raise AssertionError('Selected prefix mismatch accepted')
    wrong_hashes=hashes.copy();wrong_hashes[6]='00'*32
    with patch('msasl_estimated_clock.av.open',return_value=HalfContainer(raw)):
        try:scan_estimated_interval('fixture',row,reference,expected_rgb_sha256=wrong_hashes)
        except ValueError:pass
        else:raise AssertionError('Selected RGB mismatch accepted')
    return dict(passed=True,groups=['rational_halfopen_origin_preroll','raw_ordinals_preserved_and_estimated_boundary',
                                  'strict_selected_reentry_partition_type_guards','manual_constant_RGB_crop_hash_and_native_short',
                                  'native16_no_padding_with_outside_suffix','prefix_and_RGB_mismatch_rejects'],GPU_hours=0,
                scope='Synthetic software only; outside suffix exemption explicitly prospective')
