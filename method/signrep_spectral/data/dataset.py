"""Training clip dataset driven by a precomputed, seeded sample schedule (no labels)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import record_valid_starts
from .pose_reader import read_raw_pose
from .priors import HEADS, build_canonical_targets
from .video import decode_frames


def make_schedule(records: List[dict], num_samples: int, seed: int, clip_length: int) -> np.ndarray:
    """Uniform over eligible videos, then uniform over that video's valid starts."""
    rng = np.random.Generator(np.random.PCG64(seed))
    vids = rng.integers(0, len(records), size=num_samples)
    out = np.zeros((num_samples, 2), np.int64)
    for i, v in enumerate(vids):
        starts = record_valid_starts(records[v], clip_length)
        out[i] = (v, starts[rng.integers(0, len(starts))])
    return out


class TrainClipDataset(Dataset):
    def __init__(self, records: List[dict], schedule: np.ndarray, cfg_t: dict, clip_length: int,
                 max_rel_dev: float):
        self.records = records
        self.schedule = schedule
        self.cfg_t = cfg_t
        self.T = clip_length
        self.max_rel_dev = max_rel_dev

    def __len__(self):
        return len(self.schedule)

    def __getitem__(self, i):
        v, start = (int(x) for x in self.schedule[i])
        rec = self.records[v]
        frames, pts = decode_frames(str(Path(rec["source_root"]) / rec["relative_path"]), start, start + self.T)
        if len(frames) != self.T:  # clock validity of `start` was checked on PyAV PTS at audit
            raise RuntimeError(f"clip decode mismatch {rec['sample_id']}@{start}")
        raw = read_raw_pose(rec["pose_path"], self.cfg_t["confidence_threshold"])
        tb = build_canonical_targets(raw, start, self.T, rec["width"], rec["height"],
                                     self.cfg_t["scale_min_observations"], self.cfg_t["scale_floor"],
                                     self.cfg_t["max_relative_extent"])
        item = {"frames": torch.from_numpy(frames), "index": i, "start": start}
        for h in HEADS:
            item[f"t_{h}"] = torch.from_numpy(tb["targets"][h])
            item[f"m_{h}"] = torch.from_numpy(tb["masks"][h])
        return item
