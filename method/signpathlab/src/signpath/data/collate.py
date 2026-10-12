"""Explicit masks and timestamps at the NumPy/torch boundary."""
from dataclasses import dataclass, fields, replace
import numpy as np
import torch

from ..cache.schema import FeatureBundle
from ..contracts import MaskPlan, TokenBundle
from ..geometry.paths import DEFAULT_PATHS
from ..targets.projection import TargetProjectors


@dataclass(frozen=True)
class TrainingExample:
    features: FeatureBundle
    tokens: TokenBundle
    plan: MaskPlan
    projectors: TargetProjectors | None = None
    cluster_ids: np.ndarray | None = None


@dataclass(frozen=True)
class ModelBatch:
    streams: tuple[torch.Tensor, ...]  # seven [B,T,F_type] arrays
    valid: torch.Tensor               # [B,T,7] real observed context
    target_valid: torch.Tensor        # [B,T,7] valid, distinct-source targets
    visible: torch.Tensor             # [B,T,7] closed student context
    semantic_targets: torch.Tensor    # [B,T,7] eligible hidden observations
    relative_seconds: torch.Tensor    # [B,T], shared teacher/student origin
    timestamps: torch.Tensor          # [B,T], original seconds for export
    source_frame_index: torch.Tensor  # [B,T], -1 padding
    lengths: torch.Tensor             # [B], number of real selected observations
    geometry_values: torch.Tensor     # [B,G,Q], deterministic target only by default
    geometry_valid: torch.Tensor      # [B,G]
    geometry_visible: torch.Tensor    # [B,G]
    geometry_targets: torch.Tensor    # [B,G]
    geometry_path: torch.Tensor       # [B,G], public schedule IDs
    geometry_scale: torch.Tensor      # [B,G], public schedule IDs
    geometry_times: torch.Tensor      # [B,G,3], center, requested lo/hi
    cluster_ids: torch.Tensor         # [B,T,7], -100 invalid/unused
    sample_ids: tuple[str, ...]
    diagnostics: tuple[dict, ...]

    def to(self, device):
        changes = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                changes[field.name] = value.to(device, non_blocking=value.is_pinned())
            elif field.name == "streams":
                changes[field.name] = tuple(x.to(device, non_blocking=x.is_pinned()) for x in value)
        return replace(self, **changes)

    def pin_memory(self):
        changes = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                changes[field.name] = value.pin_memory() if value.device.type == "cpu" else value
            elif field.name == "streams":
                changes[field.name] = tuple(x.pin_memory() if x.device.type == "cpu" else x for x in value)
        return replace(self, **changes)

    def select(self, indices):
        indices = list(indices)
        changes = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                changes[field.name] = value[indices]
            elif field.name == "streams":
                changes[field.name] = tuple(x[indices] for x in value)
            else:
                changes[field.name] = tuple(value[i] for i in indices)
        return replace(self, **changes)


def collate_examples(examples, raw_dimensions, geometry_output_dim=32, *, pad_to=None):
    if not examples or len(raw_dimensions) != 7:
        raise ValueError("A batch needs examples and seven raw dimensions")
    b = len(examples)
    t = max(len(e.features.timestamps) for e in examples)
    if pad_to is not None:
        if pad_to < t:
            raise ValueError("pad_to cannot truncate a real observation")
        t = pad_to
    g = max(1, max(len(e.tokens.geometry) for e in examples))
    streams = tuple(torch.zeros(b, t, d) for d in raw_dimensions)
    bool_raw = lambda: torch.zeros(b, t, 7, dtype=torch.bool)
    valid, target_valid, visible, targets = [bool_raw() for _ in range(4)]
    relative, timestamps = torch.zeros(b, t), torch.zeros(b, t, dtype=torch.float64)
    sources, lengths = torch.full((b, t), -1, dtype=torch.int64), torch.zeros(b, dtype=torch.int64)
    geo_values, geo_times = torch.zeros(b, g, geometry_output_dim), torch.zeros(b, g, 3)
    geo_valid, geo_visible, geo_targets = [torch.zeros(b, g, dtype=torch.bool) for _ in range(3)]
    paths, scales = torch.zeros(b, g, dtype=torch.int64), torch.zeros(b, g, dtype=torch.int64)
    clusters = torch.full((b, t, 7), -100, dtype=torch.int64)
    path_ids = {group.name: i for i, group in enumerate(DEFAULT_PATHS)}
    for i, example in enumerate(examples):
        f, bundle, plan = example.features, example.tokens, example.plan
        n = len(f.timestamps)
        if len(bundle.tokens) != n * 7 or bundle.sample_id != f.meta["sample_id"]:
            raise ValueError("Token/feature identity or time-major indexing mismatch")
        lengths[i] = n
        timestamps[i, :n] = torch.from_numpy(f.timestamps.copy())
        relative[i, :n] = torch.from_numpy((f.timestamps - f.timestamps[0]).astype(np.float32))
        sources[i, :n] = torch.from_numpy(f.source_frame_index.copy())
        seen = set()
        unique = []
        for source in f.source_frame_index:
            unique.append(int(source) not in seen)
            seen.add(int(source))
        for slot, token in enumerate(bundle.tokens):
            frame, stream = divmod(slot, 7)
            if token.stream != stream or token.values.shape != (raw_dimensions[stream],):
                raise ValueError("Token stream/dimension mismatch")
            streams[stream][i, frame] = torch.from_numpy(token.values.copy())
            valid[i, frame, stream] = token.valid
            target_valid[i, frame, stream] = token.target_valid and unique[frame]
        visible[i].view(-1)[list(plan.visible)] = True
        targets[i].view(-1)[list(plan.semantic_targets)] = True
        if (visible[i] & ~valid[i]).any() or (targets[i] & visible[i]).any():
            raise ValueError("Invalid context/target masks")
        targets[i] &= target_valid[i]
        for j, geo in enumerate(bundle.geometry):
            paths[i, j], scales[i, j] = path_ids[geo.group], geo.scale
            geo_times[i, j] = torch.tensor([geo.timestamp - f.timestamps[0], geo.bounds[0] - f.timestamps[0], geo.bounds[1] - f.timestamps[0]])
            if geo.valid:
                if example.projectors is None:
                    raise ValueError("Valid geometry requires fitted train-only target projectors")
                key = f"{geo.group}/{geo.scale}"
                if key not in example.projectors.projections:
                    raise ValueError(f"No registered train projection for {key}")
                projected = example.projectors.projections[key].apply(geo.descriptor)
                if projected.shape != (geometry_output_dim,):
                    raise ValueError("Geometry projection dimension mismatch")
                geo_values[i, j] = torch.from_numpy(projected)
                geo_valid[i, j] = True
        geo_visible[i, list(plan.geometry_visible)] = True
        geo_targets[i, list(plan.geometry_targets)] = True
        if ((geo_visible[i] | geo_targets[i]) & ~geo_valid[i]).any() or (geo_visible[i] & geo_targets[i]).any():
            raise ValueError("Invalid geometry visibility/eligibility")
        if example.cluster_ids is not None:
            if example.cluster_ids.shape != (n, 7) or example.cluster_ids.dtype != np.int64:
                raise ValueError("Cluster IDs require int64 [T,7]")
            clusters[i, :n] = torch.from_numpy(example.cluster_ids.copy())
            clusters[i, :n][~target_valid[i, :n]] = -100
    return ModelBatch(streams, valid, target_valid, visible, targets, relative, timestamps, sources, lengths,
        geo_values, geo_valid, geo_visible, geo_targets, paths, scales, geo_times, clusters,
        tuple(e.tokens.sample_id for e in examples), tuple(e.plan.diagnostics for e in examples))
