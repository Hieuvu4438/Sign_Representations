"""Pre-LN Transformer (SDPA) with key-validity masks and stochastic depth; predictor decoder blocks."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(keep)
        return x * mask / keep


class Attention(nn.Module):
    def __init__(self, dim, heads, attn_drop=0.0, kv_dim=None):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(kv_dim or dim, 2 * dim)
        self.o = nn.Linear(dim, dim)
        self.attn_drop = attn_drop

    def forward(self, x, ctx=None, key_valid=None):
        ctx = x if ctx is None else ctx
        B, L, D = x.shape
        q = self.q(x).reshape(B, L, self.h, D // self.h).transpose(1, 2)
        k, v = self.kv(ctx).reshape(B, ctx.shape[1], 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        mask = None if key_valid is None else key_valid[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.attn_drop if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, L, D))


class Mlp(nn.Sequential):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, dim), nn.Dropout(drop))


class Block(nn.Module):
    def __init__(self, dim, heads, ffn, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(dim), Attention(dim, heads, attn_drop)
        self.n2, self.mlp = nn.LayerNorm(dim), Mlp(dim, ffn, drop)
        self.dp = DropPath(drop_path)
        self.drop = nn.Dropout(drop)

    def forward(self, x, key_valid):
        x = x + self.dp(self.drop(self.attn(self.n1(x), key_valid=key_valid)))
        return x + self.dp(self.mlp(self.n2(x)))


class Encoder(nn.Module):
    def __init__(self, dim=256, depth=6, heads=8, ffn=1024, drop=0.0, attn_drop=0.0, drop_path=0.1):
        super().__init__()
        rates = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([Block(dim, heads, ffn, drop, attn_drop, r) for r in rates])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_valid):
        key_valid = ensure_any(key_valid)
        for blk in self.blocks:
            x = blk(x, key_valid)
        return self.norm(x)


class DecoderBlock(nn.Module):
    """Pre-LN: query self-attention, cross-attention to visible context, MLP."""

    def __init__(self, dim, heads, ffn):
        super().__init__()
        self.n1, self.self_attn = nn.LayerNorm(dim), Attention(dim, heads)
        self.n2, self.cross = nn.LayerNorm(dim), Attention(dim, heads)
        self.nc = nn.LayerNorm(dim)
        self.n3, self.mlp = nn.LayerNorm(dim), Mlp(dim, ffn)

    def forward(self, q, q_valid, ctx, ctx_valid):
        q = q + self.self_attn(self.n1(q), key_valid=q_valid)
        q = q + self.cross(self.n2(q), ctx=self.nc(ctx), key_valid=ctx_valid)
        return q + self.mlp(self.n3(q))


def ensure_any(valid: torch.Tensor) -> torch.Tensor:
    """Rows without any valid key would make SDPA return NaN; let them attend to slot 0 (their outputs are
    ignored by every consumer because the row has no valid token). Sync-free."""
    first = torch.zeros_like(valid)
    first[:, 0] = True
    return valid | (first & ~valid.any(1, keepdim=True))


def pack(select: torch.Tensor):
    """select [B,K] bool -> (idx [B,L] slot ids in ascending order, valid [B,L]); L = max count (>=1)."""
    counts = select.sum(1)
    L = max(int(counts.max().item()) if select.numel() else 0, 1)
    order = torch.argsort((~select).to(torch.int8), dim=1, stable=True)[:, :L]
    valid = torch.arange(L, device=select.device)[None, :] < counts[:, None]
    return order, valid


def gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))
