"""Predictor (§8.5): 2 decoder blocks at d=128. Queries = learned mask token + metadata embeddings of the target
slots (type/part/family/scale/view/time); never target descriptor values."""
from __future__ import annotations

import torch
from torch import nn

from asm_jepa.features.tokens import TokenLayout
from asm_jepa.models.encoder import DecoderBlock, ensure_any
from asm_jepa.models.tokenizer import MetaEmbedding


class Predictor(nn.Module):
    def __init__(self, layout: TokenLayout, enc_dim=256, dim=128, depth=2, heads=4, ffn=512, out_dim=None):
        super().__init__()
        self.ctx_proj = nn.Linear(enc_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.meta = MetaEmbedding(layout, dim)
        self.ctx_meta = MetaEmbedding(layout, dim)
        self.blocks = nn.ModuleList([DecoderBlock(dim, heads, ffn) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, out_dim or enc_dim)

    def forward(self, ctx, ctx_idx, ctx_valid, q_idx, q_valid, token_time, log_duration):
        """ctx [B,Lc,256] online context features at slots ctx_idx; q_idx [B,Lq] target slots."""
        c = self.ctx_proj(ctx) + self.ctx_meta(ctx_idx, torch.gather(token_time, 1, ctx_idx), log_duration)
        q = self.mask_token + self.meta(q_idx, torch.gather(token_time, 1, q_idx), log_duration)
        cv, qv = ensure_any(ctx_valid), ensure_any(q_valid)
        for blk in self.blocks:
            q = blk(q, qv, c, cv)
        return self.out(self.norm(q))
