import hashlib
import io
import unittest
from fractions import Fraction

import av
import numpy as np

from msasl_native import decode_interval
from msasl_native_scan_v2 import scan_interval, ffprobe_interval, EligibilityIssue


class StreamingNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sink = io.BytesIO()
        with av.open(sink, 'w', format='matroska') as encoder:
            s = encoder.add_stream('ffv1', rate=24)
            s.width, s.height, s.pix_fmt = 32, 24, 'bgr0'
            for i in range(96):
                rgb = np.empty((24,32,3), dtype=np.uint8)
                rgb[:,:,0], rgb[:,:,1], rgb[:,:,2] = i, np.arange(32)[None,:], np.arange(24)[:,None]
                for packet in s.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
                    encoder.mux(packet)
            for packet in s.encode():
                encoder.mux(packet)
        cls.raw = sink.getvalue()

    def test_full_decode_oracle_and_crop_hash(self):
        row = dict(start_time=1., end_time=2., box=[.25,.125,.75,.875])
        result, ticks = scan_interval(io.BytesIO(self.raw), row)
        with av.open(io.BytesIO(self.raw)) as d:
            s = d.streams.video[0]
            expected = [int(f.pts) for f in d.decode(video=0) if Fraction(1) <= (f.pts-s.start_time)*s.time_base < 2]
        self.assertEqual(ticks, expected)
        self.assertEqual(result['selected_native_frames'], 24)
        self.assertEqual(result['crop_yxyx_pixels'], (6,4,18,28))
        arrays, times, meta, _ = decode_interval(io.BytesIO(self.raw), row)
        self.assertEqual(result['all_resized_RGB_sha256'], hashlib.sha256(arrays.tobytes()).hexdigest())
        self.assertEqual(result['first_native16_RGB_sha256'], hashlib.sha256(arrays[:16].tobytes()).hexdigest())
        self.assertEqual(times, [(t-result['origin_pts'])*Fraction(result['time_base']) for t in ticks])

    def test_short_boundary_no_padding_and_resource_stop(self):
        row = dict(start_time=1., end_time=1.25, box=[0.,0.,1.,1.])
        result, ticks = scan_interval(io.BytesIO(self.raw), row)
        self.assertEqual(len(ticks), 6)
        self.assertFalse(result['native16_eligible'])
        self.assertIsNone(result['first_native16_RGB_sha256'])
        self.assertEqual(result['complete_stride2_windows'], 0)
        row['end_time'] = 2.
        with self.assertRaises(EligibilityIssue) as e:
            scan_interval(io.BytesIO(self.raw), row, max_frames=16)
        self.assertEqual(e.exception.reason, 'resource_selected_frame_bound')
        with self.assertRaises(EligibilityIssue) as e:
            scan_interval(io.BytesIO(self.raw), row, max_decoded=1)
        self.assertEqual(e.exception.reason, 'resource_seek_frame_bound')

    def test_reference_native_tick_oracles_and_missing_field(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        row = dict(start_time=.25, end_time=1.)
        scan = dict(time_base='1/4', origin_pts=8)
        data = dict(streams=[dict(time_base='1/4', start_pts=8, width=32, height=24)],
                    frames=[dict(pkt_pts=t, best_effort_timestamp=t) for t in [8,9,10,11,12,13]])
        def response():
            return SimpleNamespace(stdout=json.dumps(data).encode(), stderr=b'')
        with patch('msasl_native_scan_v2.subprocess.run', return_value=response()):
            result = ffprobe_interval('fixture.mp4', row, scan, [9,10,11])
            self.assertTrue(result['integer_pts_exact'])
            self.assertEqual(result['selected_native_frames'], 3)
        data['frames'][1]['best_effort_timestamp'] = 99
        with patch('msasl_native_scan_v2.subprocess.run', return_value=response()):
            with self.assertRaises(EligibilityIssue) as e:
                ffprobe_interval('fixture.mp4', row, scan, [9,10,11])
            self.assertEqual(e.exception.reason, 'reference_best_effort_not_native_pts')
        del data['frames'][1]['pkt_pts']
        with patch('msasl_native_scan_v2.subprocess.run', return_value=response()):
            with self.assertRaises(KeyError):
                ffprobe_interval('fixture.mp4', row, scan, [9,10,11])
