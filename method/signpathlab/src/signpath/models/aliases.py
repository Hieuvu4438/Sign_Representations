"""Select one attention key per observed source frame, retaining dense output slots."""
import torch


def source_aliases(sources, real):
    """Return first real slot and uniqueness mask; -1 padding is never an observation."""
    t = sources.shape[1]
    slots = torch.arange(t, device=sources.device)
    same = (sources[:, :, None] == sources[:, None, :]) & real[:, None, :] & (sources[:, :, None] >= 0)
    first = torch.where(same, slots[None, None, :], t).amin(dim=2).clamp_max(t - 1)
    return first, real & (sources >= 0) & (first == slots[None])
