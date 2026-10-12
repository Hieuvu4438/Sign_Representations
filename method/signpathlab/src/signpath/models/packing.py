from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PackedTokens:
    values: torch.Tensor
    valid: torch.Tensor
    padding: torch.Tensor  # True = ignored by MHA; safe dummy key for empty rows
    slots: torch.Tensor    # -1 for padding


def pack_tokens(values, keep):
    if values.ndim != 3 or keep.shape != values.shape[:2] or keep.dtype != torch.bool:
        raise ValueError("Expected values[B,N,D] and boolean keep[B,N]")
    width = max(1, int(keep.sum(dim=1).max().item()))
    rows, slots, masks = [], [], []
    for row in range(len(values)):
        ids = torch.nonzero(keep[row], as_tuple=False).flatten()
        # An all-empty rank still has a valid autograd graph and a finite dummy key.
        selected = values[row, ids]
        rows.append(F.pad(selected, (0, 0, 0, width - len(ids))))
        slots.append(F.pad(ids, (0, width - len(ids)), value=-1))
        masks.append(torch.arange(width, device=values.device) < len(ids))
    real = torch.stack(masks)
    padding = ~real
    padding[~real.any(dim=1), 0] = False
    return PackedTokens(torch.stack(rows), real, padding, torch.stack(slots))


def scatter_packed(values, packed, num_slots):
    result = values.new_zeros(len(values), num_slots, values.shape[-1])
    for b in range(len(values)):
        real = packed.valid[b]
        result[b, packed.slots[b, real]] = values[b, real]
    return result
