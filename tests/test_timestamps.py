import unittest

import numpy as np

from signrepr.timestamps import frame_intervals


class SourceClock(unittest.TestCase):
    def test_variable_presentation_clock_keeps_offset_and_gaps(self):
        frames = [{'best_effort_timestamp_time': '.067'},
                  {'best_effort_timestamp_time': '.1'},
                  {'best_effort_timestamp_time': '.15', 'pkt_duration_time': '.04'}]
        clock, policy = frame_intervals(frames, 3, 30)
        np.testing.assert_allclose(clock, [[.067, .1], [.1, .15], [.15, .19]])
        self.assertEqual(policy, 'reported_last_packet_duration')

    def test_invalid_presentation_clock_cannot_fall_back_to_fps(self):
        for frames, count in [([{'best_effort_timestamp_time': '0'}], 2),
                              ([{}], 1),
                              ([{'best_effort_timestamp_time': 'NaN'}], 1),
                              ([{'best_effort_timestamp_time': '1'},
                                {'best_effort_timestamp_time': '1'}], 2)]:
            with self.assertRaises(ValueError):
                frame_intervals(frames, count, 30)
