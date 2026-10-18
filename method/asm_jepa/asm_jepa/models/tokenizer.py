"""Token embeddings (§8.1-8.3). Every token embedding reads only its own slot's features + public metadata,
so embeddings can be computed for all slots and then gathered (locality)."""
from __future__ import annotations

import math

import torch
from torch import nn

from asm_jepa.data.schema import N_CHANNELS, N_PARTS, N_RELATIONS, PART_ORDER, PART_SIZES, PART_SLICES, RELATION_CHANNELS
from asm_jepa.features.tokens import TYPE_RELATION, TYPE_STATE, TokenLayout


class Fourier(nn.Module):
    """Fixed sinusoidal features of a continuous scalar followed by a linear map (no learned length table)."""

    def __init__(self, dim: int, n_freq: int = 16, max_freq: float = 64.0):
        super().__init__()
        self.register_buffer("freqs", torch.logspace(0, math.log10(max_freq), n_freq) * math.pi, persistent=False)
        self.proj = nn.Linear(2 * n_freq + 1, dim)

    def forward(self, x):
        a = x[..., None].float() * self.freqs
        return self.proj(torch.cat([torch.sin(a), torch.cos(a), x[..., None].float()], -1))


class MetaEmbedding(nn.Module):
    """type + sub-id (part / relation / path family) + scale + view + time + clip-duration embeddings."""

    def __init__(self, layout: TokenLayout, dim: int):
        super().__init__()
        A = layout.arrays
        t = torch.as_tensor(A["type"])
        sub = torch.as_tensor(A["sub"])
        sub_idx = torch.where(t == TYPE_STATE, sub, torch.where(t == TYPE_RELATION, N_PARTS + sub,
                                                                N_PARTS + N_RELATIONS + sub))
        self.register_buffer("type_idx", t, persistent=False)
        self.register_buffer("sub_idx", sub_idx, persistent=False)
        self.register_buffer("scale_idx", torch.as_tensor(A["scale"]) + 1, persistent=False)
        self.register_buffer("view_idx", torch.as_tensor(A["view"]) + 1, persistent=False)
        self.type_emb = nn.Embedding(4, dim)
        self.sub_emb = nn.Embedding(N_PARTS + N_RELATIONS + layout.n_families, dim)
        self.scale_emb = nn.Embedding(layout.n_scales + 1, dim)
        self.view_emb = nn.Embedding(3, dim)
        self.time = Fourier(dim)
        self.duration = Fourier(dim, n_freq=8, max_freq=8.0)
        for emb in (self.type_emb, self.sub_emb, self.scale_emb, self.view_emb):
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, slot_idx, token_time, log_duration):
        """slot_idx [B,L] layout slot ids; token_time [B,L] in [0,1]; log_duration [B]."""
        e = (self.type_emb(self.type_idx[slot_idx]) + self.sub_emb(self.sub_idx[slot_idx])
             + self.scale_emb(self.scale_idx[slot_idx]) + self.view_emb(self.view_idx[slot_idx]))
        return e + self.time(token_time) + self.duration(log_duration)[:, None, :]


def _proj(i, d):
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.LayerNorm(d))


class Tokenizer(nn.Module):
    def __init__(self, layout: TokenLayout, dim: int, patch_frames: int, share_hand: bool = True):
        super().__init__()
        self.layout, self.P, self.dim = layout, patch_frames, dim
        P = patch_frames
        self.state = nn.ModuleDict({
            "body": _proj(P * PART_SIZES["body"] * N_CHANNELS, dim),
            "lh": _proj(P * PART_SIZES["lh"] * N_CHANNELS, dim),
            "face": _proj(P * PART_SIZES["face"] * N_CHANNELS, dim),
        })
        self.share_hand = share_hand
        if not share_hand:
            self.state["rh"] = _proj(P * PART_SIZES["rh"] * N_CHANNELS, dim)
        if layout.relations:
            self.relation = _proj(P * RELATION_CHANNELS, dim)
        if layout.geometry:
            self.geometry = nn.Linear(3, dim)
        if layout.timing:
            self.timing = nn.Linear(7, dim)
        self.meta = MetaEmbedding(layout, dim)
        self.register_buffer("all_slots", torch.arange(layout.n_tokens), persistent=False)

    def content(self, joints, rel, motion_geom, motion_time):
        """joints [B,T,114,7], rel [B,T,3,5], motion_* [B,F*C*S,d] -> content embeddings [B,K,D]."""
        B, T = joints.shape[:2]
        n_p = T // self.P
        parts = []
        for part in PART_ORDER:
            x = joints[:, :, PART_SLICES[part]].reshape(B, n_p, -1)
            key = "lh" if (part == "rh" and self.share_hand) else part
            parts.append(self.state[key](x))
        out = [torch.cat(parts, 1)]
        if self.layout.relations:
            r = rel.reshape(B, n_p, self.P, N_RELATIONS, RELATION_CHANNELS).permute(0, 3, 1, 2, 4)
            out.append(self.relation(r.reshape(B, N_RELATIONS * n_p, -1)))
        if self.layout.motion_enabled:
            views = []
            if self.layout.geometry:
                views.append(self.geometry(motion_geom))
            if self.layout.timing:
                views.append(self.timing(motion_time))
            out.append(torch.stack(views, 2).reshape(B, -1, self.dim))
        x = torch.cat(out, 1)
        assert x.shape[1] == self.layout.n_tokens, (x.shape, self.layout.n_tokens)
        return x

    def forward(self, joints, rel, motion_geom, motion_time, token_time, log_duration):
        B = joints.shape[0]
        slots = self.all_slots[None].expand(B, -1)
        return self.content(joints, rel, motion_geom, motion_time) + self.meta(slots, token_time, log_duration)
