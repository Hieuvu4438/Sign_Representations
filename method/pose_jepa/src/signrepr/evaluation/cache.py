"""Persistent calendar feature cache with explicit source-time mapping."""
from pathlib import Path
import numpy as np


def save_sequence_features(path, sequence):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=sequence['features'].numpy(),
                        length=np.array(sequence['length'], dtype=np.int64),
                        timestamps_sec=sequence['timestamps_sec'].numpy(),
                        timing_known=np.array(sequence['timing_known'], dtype=bool),
                        part_valid=sequence['part_valid'].numpy(),
                        source_frame_indices=sequence['source_frame_indices'].numpy(),
                        segment_indices=sequence['segment_indices'].numpy(),
                        observed_time_mask=sequence['observed_time_mask'].numpy(),
                        status=np.array(sequence['status']))
