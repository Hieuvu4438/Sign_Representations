"""PhonoJEPA model (proposal12 v2.1 §6.5): channel tokenizers, one transformer encoder (+CLS), narrow predictor with one
query slot per target cell, contextual / intrinsic / global heads, EMA teacher.

Token layout: n = b * NC + c (bin-major), 32 bins x 8 channels = 256 tokens; time position = bin index b (1-D RoPE on
q/k, CLS gets no rotation). The student encoder only attends to VISIBLE tokens (key padding); non-visible positions carry a
constant zero input and their outputs are never used, so hidden content cannot reach visible outputs or the CLS.
The predictor sequence is [CLS ctx | 256 context slots (key-valid = visible) | 256 query slots (key-valid = target)];
a query slot = mask embedding + channel embedding + RoPE(bin) — never coordinates.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from s2.features import TYPE_TARGET_DIMS, TYPE_TOKEN_DIMS
from s2.joints import NC, TOKEN_TYPE

TB = 32
N_TOK = TB * NC


def rope(x, pos):
    """x [B,H,N,D], pos [N] or [B,N] (float); rotary embedding on interleaved pairs (fixed variant of V-JEPA 2's
    rotate_queries_or_keys: frequencies are repeat_interleave'd, see facebookresearch/vjepa2 PR #15)."""
    D = x.shape[-1]
    omega = 1.0 / (10000 ** (torch.arange(D // 2, device=x.device, dtype=torch.float32) / (D / 2)))
    if pos.dim() == 1:
        pos = pos[None]
    freq = pos.float()[:, None, :, None] * omega                              # [B,1,N,D/2]
    sin, cos = freq.sin().repeat_interleave(2, -1), freq.cos().repeat_interleave(2, -1)
    y = x.float().unflatten(-1, (-1, 2))
    y = torch.stack((-y[..., 1], y[..., 0]), -1).flatten(-2)
    return (x.float() * cos + y * sin).to(x.dtype)


class Block(nn.Module):
    def __init__(self, d, heads, mlp=4.0, drop_path=0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, int(mlp * d)), nn.GELU(), nn.Linear(int(mlp * d), d))
        self.h = heads

    def forward(self, x, pos, rot, key_ok):
        """x [B,N,d]; pos [N]; rot [N] bool (apply RoPE); key_ok [B,N] bool."""
        B, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        qr, kr = rope(q, pos), rope(k, pos)
        r = rot[None, None, :, None]
        q, k = torch.where(r, qr, q), torch.where(r, kr, k)
        mask = key_ok[:, None, None, :]                                             # True = may attend
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, d))
        return x + self.mlp(self.n2(x))


class Tokenizer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(i, 256), nn.GELU(), nn.Linear(256, d), nn.LayerNorm(d))
                                   for i in TYPE_TOKEN_DIMS])
        self.ch = nn.Parameter(torch.randn(NC, d) * 0.02)
        self.absent = nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, ins, present):
        """ins list[NC] [B,Tb,Din]; present [B,NC,Tb] -> tokens [B, Tb*NC, d] (bin-major)."""
        out = []
        for c in range(NC):
            t = self.mlps[TOKEN_TYPE[c]](ins[c])
            t = torch.where(present[:, c, :, None], t, self.absent.to(t.dtype).expand_as(t))
            out.append(t + self.ch[c])
        return torch.stack(out, 2).flatten(1, 2)                                     # [B,Tb,NC,d] -> [B,N,d]


def _positions(device):
    return torch.arange(TB, device=device).repeat_interleave(NC).float()           # bin of each token


class Encoder(nn.Module):
    def __init__(self, d=256, depth=6, heads=4, descriptor_input='none'):
        super().__init__()
        self.tok = Tokenizer(d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.d = d
        from s2.interval_heads import DescriptorTokenizer
        self.descriptor = DescriptorTokenizer(d,descriptor_input) if descriptor_input!='none' else None

    def forward(self, ins, present, visible, return_layers=0, descriptors=None):
        """visible [B,NC,Tb] -> (tokens [B,N,d], cls [B,d], last-k layer outputs list)."""
        x = self.tok(ins, present)
        vis = visible.transpose(1, 2).flatten(1)                                    # [B, Tb*NC] bin-major
        x = x * vis[..., None]                                                      # non-visible inputs -> constant 0
        B = x.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1).to(x.dtype), x], 1)
        key_ok = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), vis], 1)
        pos = torch.cat([torch.zeros(1, device=x.device), _positions(x.device)])
        rot = torch.cat([torch.zeros(1, dtype=torch.bool, device=x.device), torch.ones(N_TOK, dtype=torch.bool, device=x.device)])
        if self.descriptor is not None:
            if descriptors is None:raise ValueError('Descriptor-input encoder requires support-checked descriptors')
            extra,ok,epos=self.descriptor(descriptors)
            x=torch.cat([x,extra.to(x.dtype)],1)
            key_ok=torch.cat([key_ok,ok],1)
            pos=torch.cat([pos,epos]);rot=torch.cat([rot,torch.ones_like(epos,dtype=torch.bool)])
        layers = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, pos, rot, key_ok)
            if return_layers and i >= len(self.blocks) - return_layers:
                layers.append(x[:, 1:1+N_TOK])
        x = self.norm(x)
        return x[:, 1:1+N_TOK], x[:, 0], layers


class Predictor(nn.Module):
    def __init__(self, d=256, dp=128, depth=4, heads=4):
        super().__init__()
        self.inp = nn.Linear(d, dp)
        self.mask = nn.Parameter(torch.randn(dp) * 0.02)
        self.ch = nn.Parameter(torch.randn(NC, dp) * 0.02)
        self.blocks = nn.ModuleList([Block(dp, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dp)

    def forward(self, h, cls, visible, target):
        """h [B,N,d] student tokens, cls [B,d]; visible/target [B,NC,Tb] -> query outputs [B,N,dp] (bin-major)."""
        B = h.shape[0]
        vis = visible.transpose(1, 2).flatten(1); tgt = target.transpose(1, 2).flatten(1)
        ctx = self.inp(h) * vis[..., None]
        chq = self.ch[None].expand(TB, -1, -1).reshape(N_TOK, -1)                   # channel emb per bin-major slot
        q = (self.mask + chq)[None].expand(B, -1, -1).to(ctx.dtype)
        x = torch.cat([self.inp(cls)[:, None], ctx, q], 1)
        p = _positions(h.device)
        pos = torch.cat([torch.zeros(1, device=h.device), p, p])
        rot = torch.cat([torch.zeros(1, dtype=torch.bool, device=h.device), torch.ones(2 * N_TOK, dtype=torch.bool, device=h.device)])
        key_ok = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=h.device), vis, tgt], 1)
        for blk in self.blocks:
            x = blk(x, pos, rot, key_ok)
        return self.norm(x[:, 1 + N_TOK:])


class Heads(nn.Module):
    def __init__(self, d=256, dp=128, n_proto=0):
        super().__init__()
        self.ctx = nn.Linear(dp, d)
        self.int = nn.ModuleList([nn.Linear(dp, k) for k in TYPE_TARGET_DIMS])
        self.n_proto = n_proto

    def intrinsic(self, q):
        """q [B,N,dp] bin-major -> list[NC] [B,Tb,Dt]."""
        qq = q.view(q.shape[0], TB, NC, -1)
        return [self.int[TOKEN_TYPE[c]](qq[:, :, c]) for c in range(NC)]


class Projector(nn.Module):
    """global (CLS) / prototype projector (EMA-copied to the teacher)."""

    def __init__(self, d, out, hidden=512, bn=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.BatchNorm1d(hidden) if bn else nn.Identity(), nn.GELU(), nn.Linear(hidden, out))

    def forward(self, x):
        shp = x.shape
        return self.net(x.reshape(-1, shp[-1])).reshape(*shp[:-1], -1)


class PhonoJEPA(nn.Module):
    def __init__(self, d=256, depth=6, heads=4, dp=128, pdepth=4, n_proto=0, glob=False, sigreg=False, descriptor_input='none'):
        super().__init__()
        self.student = Encoder(d, depth, heads,descriptor_input)
        self.predictor = Predictor(d, dp, pdepth, heads)
        self.heads = Heads(d, dp)
        self.g_cls = Projector(d, d) if glob else None
        self.proto = nn.utils.parametrizations.weight_norm(nn.Linear(d, n_proto, bias=False)) if n_proto else None
        if self.proto is not None:                                   # DINO: weight-norm gain fixed at 1
            self.proto.parametrizations.weight.original0.data.fill_(1.0)
            self.proto.parametrizations.weight.original0.requires_grad_(False)
        self.sig_tok = Projector(d, 128, bn=True) if sigreg else None
        self.teacher = copy.deepcopy(self.student)
        self.t_cls = copy.deepcopy(self.g_cls) if glob else None
        self.t_proto = copy.deepcopy(self.proto) if n_proto else None
        for m in (self.teacher, self.t_cls, self.t_proto):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(False)

    def ema_pairs(self):
        pairs = [(self.student, self.teacher)]
        if self.g_cls is not None:
            pairs.append((self.g_cls, self.t_cls))
        if self.proto is not None:
            pairs.append((self.proto, self.t_proto))
        return pairs

    @torch.no_grad()
    def ema_update(self, m):
        for s, t in self.ema_pairs():
            for ps, pt in zip(s.parameters(), t.parameters()):
                pt.mul_(m).add_(ps.detach(), alpha=1 - m)
            for bs, bt in zip(s.buffers(), t.buffers()):
                bt.copy_(bs)

    @torch.no_grad()
    def copy_student_to_teacher(self):
        for s, t in self.ema_pairs():
            t.load_state_dict(s.state_dict())


def momentum(step, total, m0=0.996, m1=0.9999):
    return m1 - (m1 - m0) * (math.cos(math.pi * min(step, total) / total) + 1) / 2
