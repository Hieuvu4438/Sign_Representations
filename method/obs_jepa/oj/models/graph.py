"""Proposal B: graph-JEPA on J = 67 landmarks (proposal11 §8.1-8.6).

Token = (2-frame bin, joint). Hidden or unobserved cells are replaced by a learned missing token and are excluded
as keys/neighbours everywhere (spatial message passing, temporal attention, part pooling), so no information
flows from them into context states (receptive field = visible observations only).
  tokenizer   : 2 x 19 node slots -> 256, + joint / part / real-time embeddings
  local branch: 2 spatial graph blocks on hand (21, shared weights + side embedding) and face (18) nodes,
                inputs = per-bin local articulation features; projected and added to main node embeddings
  main block  : spatial graph (normalised anatomical adjacency over visible nodes) -> temporal self-attention per
                joint (8 heads) -> cross-part self-attention over 4 visible-pooled part summaries -> MLP
  predictor   : 4 transformer blocks (256, 8 heads) over [visible context ; queries(joint id, time)]
  heads       : observable anchors (shape/orient from local branch; loc/motion/rel/face from part summaries),
                raw-reconstruction head (B10)
Teacher = EMA of tokenizer + local branch + encoder (every module producing latent targets, §8.4).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from oj.data.landmarks import EDGES, FACE, J, LH, PARENT, PART_OF, RH
from oj.features.geometry import ANCHOR_DIMS, LOCAL_SLOTS, NODE_SLOTS

D = 256


def adjacency(n, edges):
    A = torch.zeros(n, n)
    for a, b in edges:
        A[a, b] = A[b, a] = 1.0
    return A + torch.eye(n)


def local_edges(nodes):
    pos = {int(j): i for i, j in enumerate(nodes)}
    return [(pos[a], pos[b]) for a, b in EDGES if a in pos and b in pos]


def time_embedding(t_sec, d):
    """sinusoidal embedding of real time (seconds) -> [..., d]."""
    half = d // 2
    f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t_sec.device) / half) * 2 * math.pi
    a = t_sec[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1)


class SpatialGraph(nn.Module):
    """h <- h + MLP-free update: W_self LN(h) + W_nb (A_vis-normalised neighbour mean of LN(h)); masked rows untouched."""

    def __init__(self, A, d=D):
        super().__init__()
        self.register_buffer("A", A, persistent=False)
        self.norm = nn.LayerNorm(d); self.w_self = nn.Linear(d, d); self.w_nb = nn.Linear(d, d, bias=False)

    def forward(self, h, vis):
        """h [..., N, d], vis [..., N] bool."""
        x = self.norm(h)
        A = self.A * vis[..., None, :].float()  # neighbours must be visible
        deg = A.sum(-1, keepdim=True).clamp(min=1.0)
        nb = (A / deg) @ x
        upd = F.gelu(self.w_self(x) + self.w_nb(nb))
        return h + upd * vis[..., None].float()


def masked_attention(x, vis, attn, norm):
    """self-attention over the second-to-last axis with key padding = ~vis; rows without visible keys get 0 update."""
    shp = x.shape
    xx = norm(x).reshape(-1, shp[-2], shp[-1]); vv = vis.reshape(-1, shp[-2])
    has = vv.any(-1)
    out = torch.zeros_like(xx)
    if has.any():
        o, _ = attn(xx[has], xx[has], xx[has], key_padding_mask=~vv[has], need_weights=False)
        out[has] = o.to(out.dtype)
    return x + out.reshape(shp) * vis[..., None].float()


class LocalBranch(nn.Module):
    def __init__(self, d=D, blocks=2):
        super().__init__()
        self.inp_hand = nn.Linear(2 * LOCAL_SLOTS, d); self.inp_face = nn.Linear(2 * LOCAL_SLOTS, d)
        self.side = nn.Embedding(2, d); self.node_hand = nn.Embedding(21, d); self.node_face = nn.Embedding(18, d)
        Ah = adjacency(21, local_edges(LH)); Af = adjacency(18, local_edges(FACE))
        self.hand = nn.ModuleList([SpatialGraph(Ah, d) for _ in range(blocks)])
        self.face = nn.ModuleList([SpatialGraph(Af, d) for _ in range(blocks)])

    def forward(self, loc, vis):
        """loc: {'lh','rh': [B,Tb,21,16], 'face': [B,Tb,18,16]}, vis: [B,Tb,J] -> dict of [B,Tb,n,d]."""
        out = {}
        for s, (name, nodes) in enumerate((("lh", LH), ("rh", RH))):
            h = self.inp_hand(loc[name]) + self.side.weight[s] + self.node_hand.weight
            v = vis[:, :, nodes]
            for blk in self.hand:
                h = blk(h, v)
            out[name] = h * v[..., None].float()
        h = self.inp_face(loc["face"]) + self.node_face.weight; v = vis[:, :, FACE]
        for blk in self.face:
            h = blk(h, v)
        out["face"] = h * v[..., None].float()
        return out


class MainBlock(nn.Module):
    def __init__(self, A, d=D, heads=8):
        super().__init__()
        self.spatial = SpatialGraph(A, d)
        self.t_attn = nn.MultiheadAttention(d, heads, batch_first=True); self.t_norm = nn.LayerNorm(d)
        self.p_attn = nn.MultiheadAttention(d, heads, batch_first=True); self.p_norm = nn.LayerNorm(d)
        self.p_out = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.register_buffer("part_of", torch.as_tensor(PART_OF), persistent=False)

    def forward(self, h, vis):
        """h [B,Tb,J,d], vis [B,Tb,J]."""
        h = self.spatial(h, vis)
        ht = masked_attention(h.transpose(1, 2), vis.transpose(1, 2), self.t_attn, self.t_norm).transpose(1, 2)
        oh = F.one_hot(self.part_of, 4).float()  # [J,4]
        w = vis.float()[..., None] * oh  # [B,Tb,J,4]
        summ = torch.einsum("btjp,btjd->btpd", w, ht) / w.sum(2).clamp(min=1.0)[..., None]
        pv = w.sum(2) > 0  # [B,Tb,4]
        s2 = masked_attention(summ, pv, self.p_attn, self.p_norm)
        upd = self.p_out(s2 - summ)  # cross-part message
        h = ht + torch.einsum("jp,btpd->btjd", oh, upd) * vis[..., None].float()
        return h + self.mlp(h) * vis[..., None].float(), summ, pv


class GraphEncoder(nn.Module):
    def __init__(self, d=D, blocks=6, heads=8):
        super().__init__()
        self.inp = nn.Linear(2 * NODE_SLOTS, d); self.node = nn.Embedding(J, d); self.part = nn.Embedding(4, d)
        self.t_proj = nn.Linear(d, d); self.missing = nn.Parameter(torch.zeros(d)); nn.init.normal_(self.missing, std=0.02)
        self.local = LocalBranch(d)
        self.local_proj = nn.ModuleDict({k: nn.Linear(d, d) for k in ("lh", "rh", "face")})
        A = adjacency(J, EDGES)
        self.blocks = nn.ModuleList([MainBlock(A, d, heads) for _ in range(blocks)])
        self.norm = nn.LayerNorm(d)
        self.register_buffer("part_of", torch.as_tensor(PART_OF), persistent=False)

    def forward(self, node, loc, vis, t_bin):
        """node [B,Tb,J,38], loc dict, vis [B,Tb,J] (visible cell), t_bin [B,Tb] seconds ->
        h [B,Tb,J,d], part summaries [B,Tb,4,d], part-visibility [B,Tb,4], local feats dict."""
        d = self.node.weight.shape[-1]
        te = self.t_proj(time_embedding(t_bin, d))[:, :, None]
        h = self.inp(node) + self.node.weight + self.part.weight[self.part_of] + te
        lf = self.local(loc, vis)
        for name, nodes in (("lh", LH), ("rh", RH), ("face", FACE)):
            h[:, :, nodes] = (h[:, :, nodes] + self.local_proj[name](lf[name])).to(h.dtype)
        h = torch.where(vis[..., None], h, self.missing + self.node.weight + te)
        summ, pv = None, None
        for blk in self.blocks:
            h, summ, pv = blk(h, vis)
        return self.norm(h), summ, pv, lf


class GraphPredictor(nn.Module):
    def __init__(self, d=D, depth=4, heads=8, out=D):
        super().__init__()
        self.node = nn.Embedding(J, d); self.t_proj = nn.Linear(d, d); self.query = nn.Parameter(torch.zeros(d))
        self.ctx_proj = nn.Linear(d, d)
        el = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(el, depth); self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, out)

    def forward(self, h, vis, hid, t_bin):
        """context = visible cells of h; queries = hidden cells (joint id + time only, no coordinates).
        Returns predictions [B,Tb,J,out] (valid where hid)."""
        B, Tb, Jn, d = h.shape
        pos = self.node.weight[None, None] + self.t_proj(time_embedding(t_bin, d))[:, :, None]  # [B,Tb,J,d]
        ctx = (self.ctx_proj(h) + pos).reshape(B, Tb * Jn, d)
        q = (self.query + pos).reshape(B, Tb * Jn, d)
        x = torch.cat([ctx, q], 1)
        keep = torch.cat([vis.reshape(B, -1), hid.reshape(B, -1)], 1)
        y = self.blocks(x, src_key_padding_mask=~keep)
        return self.out(self.norm(y[:, Tb * Jn:])).reshape(B, Tb, Jn, -1)


class AnchorHeads(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.shape = nn.Linear(d, ANCHOR_DIMS["shape"] // 2); self.orient = nn.Linear(d, ANCHOR_DIMS["orient"] // 2)
        self.loc = nn.Linear(d, ANCHOR_DIMS["loc"] // 2); self.motion = nn.Linear(d, ANCHOR_DIMS["motion"] // 2)
        self.rel = nn.Linear(2 * d, ANCHOR_DIMS["rel"]); self.face = nn.Linear(d, ANCHOR_DIMS["face"])

    def forward(self, lf, vis, summ):
        """local feats (pooled over visible hand nodes) -> shape/orient; part summaries -> loc/motion/rel/face."""
        def pool(x, v):
            return (x * v[..., None].float()).sum(2) / v.float().sum(2).clamp(min=1.0)[..., None]
        hl, hr = pool(lf["lh"], vis[:, :, LH]), pool(lf["rh"], vis[:, :, RH])
        sb, sf, sl, sr = summ[:, :, 0], summ[:, :, 1], summ[:, :, 2], summ[:, :, 3]
        return {"shape": torch.cat([self.shape(hl), self.shape(hr)], -1), "orient": torch.cat([self.orient(hl), self.orient(hr)], -1),
                "loc": torch.cat([self.loc(sl), self.loc(sr)], -1), "motion": torch.cat([self.motion(sl), self.motion(sr)], -1),
                "rel": self.rel(torch.cat([sl, sr], -1)), "face": self.face(sf)}


class GraphJepa(nn.Module):
    def __init__(self, objective="anchors", d=D):
        """objective: 'latent' (B12), 'anchors' (B13 core), 'cce' (B11, S-JEPA centred CE), 'raw' (B10)."""
        super().__init__()
        self.objective = objective
        self.student = GraphEncoder(d)
        self.teacher = copy.deepcopy(self.student).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        out = 8 if objective == "raw" else d
        self.predictor = GraphPredictor(d, out=out)
        self.anchors = AnchorHeads(d) if objective == "anchors" else None

    @torch.no_grad()
    def ema_update(self, m):
        for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)

    @torch.no_grad()
    def embed(self, node, loc, vis, t_bin, use_teacher=True):
        """downstream window embedding: concat of 4 part means over visible nodes and bins (fixed pooling)."""
        enc = self.teacher if use_teacher else self.student
        h, summ, pv, lf = enc(node, loc, vis, t_bin)
        w = vis.float()[..., None] * F.one_hot(enc.part_of, 4).float()
        num = torch.einsum("btjp,btjd->bpd", w, h); den = w.sum((1, 2)).clamp(min=1.0)[..., None]
        return (num / den).flatten(1)
