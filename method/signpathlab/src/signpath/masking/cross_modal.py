"""Declared observation hiding with RGB/ROI evidence available, never strict spatial hiding."""
from dataclasses import replace
import numpy as np
from ..contracts import RawMask
from ..geometry.supports import Primitive
from .samplers import MaskSampler

ANATOMICAL_REGIONS = {
    "left_hand": tuple(range(7, 28)), "right_hand": tuple(range(28, 49)),
    "left_arm": (1, 3, 5), "right_arm": (2, 4, 6),
    "left_index": (12, 13, 14, 15), "right_index": (33, 34, 35, 36),
}


def observation_view(bundle, source_frame_index, strategy):
    """The primitive domain is part of the named cross-modal protocol.

    Anatomical masks hide model pose measurements. RGB and ROI extraction remain
    observed from the RGB modality, including their detector-derived ROI priors.
    They are not counterfactually edited when a pose-only measurement is hidden.
    """
    if strategy == "anatomical_pose":
        def pose_support(support):
            return frozenset(Primitive("pose", p.frame, p.index) if p.kind == "joint" else p for p in support)
        tokens = tuple(replace(t, raw_support=pose_support(t.raw_support)) if t.stream >= 4 else t for t in bundle.tokens)
        geometry = tuple(replace(g, support=pose_support(g.support)) for g in bundle.geometry)
    elif strategy in {"stream_temporal", "modality_dropout"}:
        tokens = tuple(replace(t, raw_support=frozenset({Primitive("feature", int(source_frame_index[i // 7]), t.stream)}))
                       for i, t in enumerate(bundle.tokens))
        geometry = bundle.geometry
    else:
        raise ValueError(f"Unimplemented cross-modal observation strategy: {strategy}")
    return replace(bundle, tokens=tokens, geometry=geometry)


class CrossModalSampler:
    def __init__(self, strategy, requested_frame_ratio=.5, span_seconds=(.16, .96)):
        if strategy not in {"anatomical_pose", "stream_temporal", "modality_dropout"}:
            raise ValueError("Legacy SHuBERT flattened masks require the original four-stream adapter, unavailable at this milestone")
        self.strategy = strategy
        self.temporal = MaskSampler(requested_frame_ratio, span_seconds)

    def sample(self, sample, generator):
        temporal = self.temporal.sample(sample, generator)
        frames = {p.frame for p in temporal.cells}
        available = frozenset(p for t in sample.tokens for p in t.raw_support)
        if self.strategy == "anatomical_pose":
            region = tuple(ANATOMICAL_REGIONS)[int(generator.integers(len(ANATOMICAL_REGIONS)))]
            joints = set(ANATOMICAL_REGIONS[region])
            cells = frozenset(p for p in available if p.kind == "pose" and p.frame in frames and p.index in joints)
        else:
            stream = int(generator.integers(7))
            region = f"stream{stream}"
            cells = frozenset(p for p in available if p.kind == "feature" and p.index == stream
                              and (self.strategy == "modality_dropout" or p.frame in frames))
        touched = {p.frame for p in cells}
        source_count = len({p.frame for p in available})
        return RawMask(cells, temporal.requested_ratio, len(touched) / source_count if source_count else 0.0,
                       "cross_modal", self.strategy, region)
