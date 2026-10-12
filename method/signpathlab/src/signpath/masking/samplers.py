import numpy as np
from ..contracts import RawMask


class MaskSampler:
    def __init__(self, requested_frame_ratio=0.5, span_seconds=(0.16, 0.96)):
        if not 0 < requested_frame_ratio < 1 or not 0 < span_seconds[0] <= span_seconds[1]:
            raise ValueError("Invalid mask sampling parameters")
        self.ratio, self.spans = requested_frame_ratio, span_seconds

    def sample(self, sample, generator):
        """Synchronized spans on original observations; generator is an owned NumPy RNG."""
        primitives = set(p for t in sample.tokens for p in t.raw_support)
        primitives.update(p for g in sample.geometry for p in g.support)
        frames = sorted({p.frame for p in primitives})
        if len(frames) < 2:
            return RawMask(frozenset(), self.ratio, 0.0, "strict_temporal")
        # Timestamp is public metadata; never use coordinate/quality/anchor values.
        frame_times = {}
        for token in sample.tokens:
            for p in token.raw_support:
                if p.frame not in frame_times:
                    frame_times[p.frame] = token.timestamp
        if set(frames) - frame_times.keys():
            raise ValueError("Temporal supports outside view require an explicit raw timestamp map")
        frames.sort(key=lambda f: (frame_times[f], f))
        times = np.asarray([frame_times[f] for f in frames])
        target = min(len(frames) - 1, max(1, int(round(len(frames) * self.ratio))))
        selected = set()
        # Finite bounded sampling; fill gaps by singleton spans if overlapping draws stall.
        for _ in range(len(frames) * 4):
            if len(selected) >= target:
                break
            available = [i for i, f in enumerate(frames) if f not in selected]
            start = int(generator.choice(available))
            duration = float(generator.uniform(*self.spans))
            candidates = [i for i in range(start, len(frames)) if times[i] < times[start] + duration and frames[i] not in selected]
            selected.update(frames[i] for i in candidates[:target - len(selected)])
        cells = frozenset(p for p in primitives if p.frame in selected)
        return RawMask(cells, self.ratio, len(selected) / len(frames), "strict_temporal")
