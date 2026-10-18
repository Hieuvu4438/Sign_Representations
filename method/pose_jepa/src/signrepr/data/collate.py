"""Zero tail padding; missing calendar frames remain in the sequence."""
import torch
from .schema import PART_ORDER


def collate_pose(samples):
    if not samples:
        raise ValueError("Empty batch")
    t = (max(s["length"] for s in samples) + 1) // 2 * 2
    b = len(samples)
    parts = {}
    for p in PART_ORDER:
        parts[p] = {}
        for k, value in samples[0]["parts"][p].items():
            tensor = value.new_zeros((b, t, *value.shape[1:]))
            for i, s in enumerate(samples):
                tensor[i, :s["length"]] = s["parts"][p][k]
            parts[p][k] = tensor
    out = {"parts": parts, "lengths": torch.tensor([s["length"] for s in samples]),
           "timing_known": torch.tensor([s["timing_known"] for s in samples]),
           "sample_ids": [s["sample_id"] for s in samples], "statuses": [s["status"] for s in samples]}
    for k in ("timestamps_sec", "position", "frame_indices", "track_continuity"):
        size = t - 1 if k == "track_continuity" else t
        tensor = samples[0][k].new_zeros((b, size))
        for i, s in enumerate(samples):
            tensor[i, :len(s[k])] = s[k]
        out[k] = tensor
    out["real"] = torch.arange(t)[None] < out["lengths"][:, None]
    if "label" in samples[0]:
        out["labels"] = [s["label"] for s in samples]
    return out


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value
