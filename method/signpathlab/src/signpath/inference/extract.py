"""Unmasked full-clip inference with declared tiling and timestamp-aligned averaging."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from ..cache.hashing import file_hash
from ..data.sampling import slice_features
from ..models.teacher_student import ResearchModel
from ..training.runner import full_clip_batch


@dataclass(frozen=True)
class ClipEmbedding:
    dense: torch.Tensor
    timestamps: torch.Tensor
    valid: torch.Tensor
    clip: torch.Tensor


def window_starts(length, window, overlap):
    if not length or not 0 <= overlap < window:
        raise ValueError("Invalid inference length/window/overlap")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, window - overlap))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def load_encoder(checkpoint, device):
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ResearchModel(value["config"]).to(device)
    model.load_state_dict(value["trainer"]["model"], strict=True)
    model.eval()
    identity = value["identities"].get("cache_identity", value["identities"].get("feature_identity"))
    if identity is None:
        raise ValueError("Checkpoint has no registered real-cache input identity")
    return model, identity, value["config"], file_hash(checkpoint)


def temporal_embedding(model, bundle, config, *, window_samples=48, overlap_samples=16, device="cpu"):
    n, d = len(bundle.timestamps), config["model"]["embed_dim"]
    sums = torch.zeros(n, d, device=device)
    counts = torch.zeros(n, 1, device=device)
    valid = torch.zeros(n, dtype=torch.bool, device=device)
    for start in window_starts(n, window_samples, overlap_samples):
        stop = min(n, start + window_samples)
        segment = slice_features(bundle, start, stop)
        batch = full_clip_batch([segment], config).to(device)
        output = model.extract_embeddings(batch)
        # Fixed original sample slots, never a loose join on rounded timestamps.
        sums[start:stop] = sums[start:stop] + output.dense[0] * output.valid[0, :, None]
        counts[start:stop] += output.valid[0, :, None]
        valid[start:stop] |= output.valid[0]
    dense = sums / counts.clamp_min(1)
    first = np.unique(bundle.source_frame_index, return_index=True)[1]
    unique = torch.zeros(n, dtype=torch.bool, device=device)
    unique[first] = True
    selected = unique & valid
    if not selected.any():
        raise ValueError("Clip has no valid encoder output")
    clip = dense[selected].mean(dim=0)
    return ClipEmbedding(dense, torch.tensor(bundle.timestamps.copy(), device=device), valid, clip)


def frozen_visual_embedding(bundle):
    values = torch.from_numpy(bundle.visual.astype(np.float32))
    valid = torch.from_numpy(bundle.visual_valid.copy())
    dense = (values * valid[..., None]).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
    frame_valid = valid.any(dim=1)
    unique = torch.zeros(len(values), dtype=torch.bool)
    unique[np.unique(bundle.source_frame_index, return_index=True)[1]] = True
    selected = unique & frame_valid
    if not selected.any():
        raise ValueError("No valid frozen visual observation")
    return ClipEmbedding(dense, torch.from_numpy(bundle.timestamps.copy()), frame_valid, dense[selected].mean(dim=0))
