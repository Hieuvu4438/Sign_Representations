"""Source-uniform sampling and replayable synchronous segment plans."""
from copy import deepcopy
from dataclasses import replace
import numpy as np


def slice_features(bundle, start, stop):
    if not 0 <= start < stop <= len(bundle.timestamps):
        raise ValueError("Invalid feature segment bounds")
    names = ("timestamps", "source_frame_index", "visual", "visual_valid", "pose_xy_body", "pose_observed", "pose_quality", "pose_coordinate_valid")
    meta = {**bundle.meta, "sample_id": f"{bundle.meta['sample_id']}:segment{start}_{stop}",
            "segment_source_sample_id": bundle.meta["sample_id"], "segment_selected_bounds": [start, stop]}
    return replace(bundle, meta=meta, **{name: getattr(bundle, name)[start:stop] for name in names},
                   raw_supports=bundle.raw_supports[start * 7:stop * 7])


class SourceSegmentSampler:
    def __init__(self, source_to_samples, seed, *, epoch_shuffle=False):
        if not source_to_samples or any(not samples for samples in source_to_samples.values()):
            raise ValueError("Sampler requires nonempty source groups")
        self.mapping = {key: tuple(value) for key, value in sorted(source_to_samples.items())}
        self.sources = tuple(self.mapping)
        self.generator = np.random.default_rng(seed)
        self.draws = 0
        self.epoch_shuffle = epoch_shuffle
        self.order, self.cursor = [], 0

    def sample(self, load_features, segment_samples):
        sample_id = self.select_sample()
        return self.segment(load_features(sample_id), segment_samples)

    def select_sample(self):
        if self.epoch_shuffle:
            if self.cursor == len(self.order):
                self.order = self.generator.permutation(len(self.sources)).tolist()
                self.cursor = 0
            source = self.sources[self.order[self.cursor]]
            self.cursor += 1
        else:
            source = self.sources[int(self.generator.integers(len(self.sources)))]
        clips = self.mapping[source]
        return clips[int(self.generator.integers(len(clips)))]

    def segment(self, bundle, segment_samples):
        n = len(bundle.timestamps)
        start = int(self.generator.integers(max(1, n - segment_samples + 1)))
        self.draws += 1
        return slice_features(bundle, start, min(n, start + segment_samples))

    def state_dict(self):
        return {"source_to_samples": self.mapping, "rng": deepcopy(self.generator.bit_generator.state), "draws": self.draws,
                "epoch_shuffle": self.epoch_shuffle, "order": list(self.order), "cursor": self.cursor}

    def load_state_dict(self, state):
        if state["source_to_samples"] != self.mapping:
            raise ValueError("Resume sampler source plan changed")
        if state.get("epoch_shuffle", False) != self.epoch_shuffle:
            raise ValueError("Resume sampler policy changed")
        self.order, self.cursor = list(state.get("order", [])), state.get("cursor", 0)
        self.generator.bit_generator.state = state["rng"]
        self.draws = state["draws"]
