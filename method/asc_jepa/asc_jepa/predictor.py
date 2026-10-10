"""Predictor (spec §8.1–8.3): shared context-memory transformer + per-stage dense query decoders.
Inputs are ONLY visible online tokens + canonical positions/timestamps + stage id (no pose, no
teacher features). Queries = full canonical grid positions; cross-attention without query
self-attention, so query chunking is exact."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sincos(x, nfreq=8, fmin=1.0, fmax=64.0):
    """x [...]: scalar coordinate -> [..., 2*nfreq] fixed-frequency features."""
    f = torch.exp(torch.linspace(math.log(fmin), math.log(fmax), nfreq, device=x.device))
    a = x[..., None] * f * math.pi
    return torch.cat([a.sin(), a.cos()], -1)


def grid_coords(T, H, W, frame_ms, frames_per_token, device):
    t = (torch.arange(T, device=device).float() * frames_per_token + (frames_per_token - 1) / 2) * frame_ms / 1000.0
    y = (torch.arange(H, device=device).float() + 0.5) / H
    x = (torch.arange(W, device=device).float() + 0.5) / W
    tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
    return torch.stack([tt, yy, xx], -1).reshape(-1, 3)  # [T*H*W, 3] (seconds, y, x)


class Predictor(nn.Module):
    def __init__(self, stage_dims: dict, d=256, heads=8, mem_blocks=2, mlp_ratio=4.0):
        super().__init__()
        self.names = list(stage_dims)
        self.inp = nn.ModuleDict({k: nn.Linear(c, d) for k, c in stage_dims.items()})
        self.out = nn.ModuleDict({k: nn.Linear(d, c) for k, c in stage_dims.items()})
        self.stage_emb = nn.Embedding(len(self.names), d)
        self.pos = nn.Linear(3 * 16, d)
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        el = nn.TransformerEncoderLayer(d, heads, int(d * mlp_ratio), 0.0, batch_first=True, norm_first=True)
        self.mem = nn.TransformerEncoder(el, mem_blocks)
        self.xattn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.q_norm, self.m_norm, self.f_norm = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, int(d * mlp_ratio)), nn.GELU(), nn.Linear(int(d * mlp_ratio), d))

    def posenc(self, coords):
        return self.pos(torch.cat([sincos(coords[..., i]) for i in range(3)], -1))

    def forward(self, ctx: dict, coords: dict, query_chunk=512):
        """ctx[name]: StageOut (student). coords[name]: [N_full,3]. Returns {name: [B,N_full,C]}."""
        mems, pads = [], []
        for si, k in enumerate(self.names):
            so = ctx[k]
            m = self.inp[k](so.tokens.float()) + self.posenc(coords[k][so.flat_idx]) + self.stage_emb.weight[si]
            mems.append(m)
            pads.append(torch.zeros(m.shape[:2], dtype=torch.bool, device=m.device))
        mem = self.mem(torch.cat(mems, 1), src_key_padding_mask=torch.cat(pads, 1))
        memn = self.m_norm(mem)
        out = {}
        B = mem.shape[0]
        for si, k in enumerate(self.names):
            q_all = self.posenc(coords[k]) + self.stage_emb.weight[si] + self.query[0]
            chunks = []
            for s in range(0, q_all.shape[0], query_chunk):
                q = q_all[s:s + query_chunk][None].expand(B, -1, -1)
                h = q + self.xattn(self.q_norm(q), memn, memn, need_weights=False)[0]
                h = h + self.ffn(self.f_norm(h))
                chunks.append(self.out[k](h))
            out[k] = torch.cat(chunks, 1)
        return out
