"""Fixed readouts (§11): only state tokens of the intact encoder are pooled.

state_part_concat1024_l2: per part valid-time mean -> L2 -> zero if absent -> concat [body, LH, RH, face] -> L2.
state_balanced_mean256_l2 (diagnostic): mean of the present per-part L2 vectors -> L2.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from asm_jepa.data.schema import N_PARTS

FEATURE_KEYS = ("state_part_concat1024_l2", "state_balanced_mean256_l2")


def part_means(x_state: torch.Tensor, valid_state: torch.Tensor, n_patches: int):
    """x_state [B, 4*n_p, D], valid [B, 4*n_p] -> per-part L2 means [B,4,D] and presence [B,4]."""
    B, _, D = x_state.shape
    x = x_state.reshape(B, N_PARTS, n_patches, D).float()
    v = valid_state.reshape(B, N_PARTS, n_patches, 1).float()
    cnt = v.sum(2)
    mean = (x * v).sum(2) / cnt.clamp_min(1.0)
    present = cnt[..., 0] > 0
    return F.normalize(mean, dim=-1) * present[..., None], present


def pool_state(x_state, valid_state, n_patches):
    parts, present = part_means(x_state, valid_state, n_patches)
    concat = F.normalize(parts.flatten(1), dim=-1)
    concat = concat * present.any(1, keepdim=True)
    bal = parts.sum(1) / present.sum(1, keepdim=True).clamp_min(1)
    bal = F.normalize(bal, dim=-1) * present.any(1, keepdim=True)
    return {"state_part_concat1024_l2": concat, "state_balanced_mean256_l2": bal}, present

