"""Auxiliary heads: state anchor (§10.2), coordinate MAE (R1 control), global projection G (§10.3), classifier (§11)."""
from __future__ import annotations

from torch import nn

from asm_jepa.data.schema import PART_ORDER
from asm_jepa.features.state import MAE_DIMS

ANCHOR_DIMS = {"body": 24, "lh": 132, "rh": 132, "face": 180}


class PartHeads(nn.Module):
    """Linear 256 -> target_dim per part; hands share one head when share_hand."""

    def __init__(self, dim: int, dims: dict, share_hand: bool = True):
        super().__init__()
        self.share_hand = share_hand
        self.heads = nn.ModuleDict({p: nn.Linear(dim, dims[p]) for p in PART_ORDER if not (share_hand and p == "rh")})

    def forward(self, part: str, x):
        key = "lh" if (self.share_hand and part == "rh") else part
        return self.heads[key](x)


def anchor_heads(dim, share_hand):
    return PartHeads(dim, ANCHOR_DIMS, share_hand)


def mae_heads(dim, share_hand):
    return PartHeads(dim, MAE_DIMS, share_hand)


class GlobalHead(nn.Sequential):
    def __init__(self, in_dim=1024, hidden=512, out_dim=256):
        super().__init__(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class LinearHead(nn.Linear):
    pass


class FinetuneHead(nn.Sequential):
    def __init__(self, in_dim, n_classes, dropout):
        super().__init__(nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, n_classes))
