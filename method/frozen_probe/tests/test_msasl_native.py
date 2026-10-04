from fractions import Fraction
import io
import unittest
import av
import numpy as np
from msasl_native import crop_bounds,selected_indices,decode_interval


class NativeContractTests(unittest.TestCase):
    def test_yxyx_axis_and_rounding_oracle(self):
        self.assertEqual(crop_bounds([.25,.1,.75,.8],100,40),(10,10,30,80))
        self.assertEqual(crop_bounds([.11,.21,.61,.71],10,10),(1,2,7,8))
        self.assertEqual(crop_bounds([0.,0.,1.,1.],8,6),(0,0,6,8))
        for box in ([.9,0.,.1,1.],[0.,-.1,1.,1.],[0.,0.,1.,float('nan')]):
            with self.assertRaises(ValueError):crop_bounds(box,100,40)

    def test_halfopen_boundary_and_origin_oracle(self):
        pts=[Fraction(i,4) for i in range(8)]
        self.assertEqual(selected_indices(pts,.25,1.),[1,2,3])
        self.assertEqual(selected_indices(pts,.26,1.01),[2,3,4])
        with self.assertRaises(ValueError):selected_indices(pts,1.,1.)
        with self.assertRaises(ValueError):selected_indices([Fraction(0),Fraction(0)],0.,1.)

    def test_lossless_seek_matches_full_reference(self):
        sink=io.BytesIO()
        with av.open(sink,'w',format='matroska') as encoder:
            stream=encoder.add_stream('ffv1',rate=24);stream.width=32;stream.height=24;stream.pix_fmt='bgr0'
            for i in range(96):
                rgb=np.full((24,32,3),i,dtype=np.uint8);f=av.VideoFrame.from_ndarray(rgb,format='rgb24')
                for packet in stream.encode(f):encoder.mux(packet)
            for packet in stream.encode():encoder.mux(packet)
        raw=sink.getvalue();self.assertLess(len(raw),131072)
        with av.open(io.BytesIO(raw)) as decoder:
            frames=list(decoder.decode(video=0));origin=decoder.streams.video[0].start_time*decoder.streams.video[0].time_base
            times=[f.pts*f.time_base-origin for f in frames]
        indices=selected_indices(times,1.,2.)
        out,pts,meta,_=decode_interval(io.BytesIO(raw),dict(start_time=1.,end_time=2.,box=[0.,0.,1.,1.]))
        self.assertEqual(pts,[times[i] for i in indices]);self.assertEqual(len(out),24)
        self.assertEqual([int(f[0,0,0]) for f in out],indices)
        with self.assertRaises(ValueError):decode_interval(io.BytesIO(raw),dict(start_time=1.,end_time=1.25,box=[0.,0.,1.,1.]))


if __name__=='__main__':unittest.main()
