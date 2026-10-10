"""CIP-Sign forecasters G0 / G1 / Gsh (proposal5 §9) on cached geometry states.

Identical architecture/init/budget for the three types. One example = (video, anchor t,
ordered edge a->b, horizon δ). History = frames t-7..t of own part b (slot 0) and of a
second slot: G0 -> own b again (part id b), G1 -> peer a of the same video, Gsh -> part a
of a *different* video (same fold, random valid anchor). Target = state of b at t+δ.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

K = 8
HORIZONS = (2, 4, 8)
EDGES = [(a, b) for a in range(4) for b in range(4) if a != b]


class Forecaster(nn.Module):
    def __init__(self, D=102, d=128, heads=4, ffn=512, enc_layers=2, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(2 * D + 1, d)
        self.part = nn.Embedding(4, d)
        self.role = nn.Embedding(2, d)
        self.time = nn.Embedding(K, d)
        self.edge = nn.Embedding(len(EDGES), d)
        self.hz = nn.Embedding(len(HORIZONS), d)
        el = nn.TransformerEncoderLayer(d, heads, ffn, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, enc_layers)
        dl = nn.TransformerDecoderLayer(d, heads, ffn, dropout, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dl, 1)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, D))
        t = torch.arange(K).repeat_interleave(2)
        self.register_buffer("tok_time", t)
        self.register_buffer("tok_role", torch.arange(2).repeat(K))
        self.register_buffer("attn_mask", t[None, :] > t[:, None])  # key time > query time masked

    def forward(self, y0, m0, y1, m1, part0, part1, edge_idx, hz_idx):
        """y*/m*: [B,K,D] own/second-slot history; part*: [B] part ids."""
        q0 = m0.float().mean(-1, keepdim=True)
        q1 = m1.float().mean(-1, keepdim=True)
        x0 = self.inp(torch.cat([y0 * m0, m0.float(), q0], -1)) + self.part(part0)[:, None]
        x1 = self.inp(torch.cat([y1 * m1, m1.float(), q1], -1)) + self.part(part1)[:, None]
        x = torch.stack([x0, x1], 2).flatten(1, 2)  # B, 2K, d interleaved by time
        x = x + self.time(self.tok_time)[None] + self.role(self.tok_role)[None]
        h = self.enc(x, mask=self.attn_mask)
        q = (self.edge(edge_idx) + self.hz(hz_idx))[:, None]
        return self.out(self.dec(q, h)[:, 0])


BLOCKS = None  # e.g. [(0,128),(128,224)] for proposal5 states: block-weighted MSE (§11.1)


def masked_mse(pred, y, m):
    m = m.float()
    if BLOCKS is None:
        err = ((pred - y) ** 2 * m).sum(-1) / m.sum(-1).clamp(min=1)
        return err, (m.sum(-1) > 0)
    errs, oks = [], []
    for a, b in BLOCKS:
        mb = m[..., a:b]
        cnt = mb.sum(-1)
        errs.append(((pred[..., a:b] - y[..., a:b]) ** 2 * mb).sum(-1) / cnt.clamp(min=1))
        oks.append((cnt > 0).float())
    E, O = torch.stack(errs, -1), torch.stack(oks, -1)
    return (E * O).sum(-1) / O.sum(-1).clamp(min=1), O.sum(-1) > 0


class Sampler:
    """Draws examples from cached states for a set of videos."""

    def __init__(self, S, V, off, vids, seed, min_hist=6):
        self.S, self.V, self.off = S, V, off
        self.vids = np.asarray(vids)
        self.rng = np.random.default_rng(seed)
        self.min_hist = min_hist
        lens = off[self.vids + 1] - off[self.vids]
        self.ok = lens >= K + max(HORIZONS)
        self.vids = self.vids[self.ok]

    def _hist(self, v, t, p):
        s = self.off[v] + t - K + 1
        return self.S[s : s + K, p], self.V[s : s + K, p]

    def draw(self, n, mode, edge_hz=None):
        Y0, M0, Y1, M1, P0, P1, E, H, YT, MT = ([] for _ in range(10))
        tries = 0
        while len(E) < n and tries < 50 * n:
            tries += 1
            v = self.vids[self.rng.integers(len(self.vids))]
            T = self.off[v + 1] - self.off[v]
            if edge_hz is None:
                ei, hi = self.rng.integers(len(EDGES)), self.rng.integers(len(HORIZONS))
            else:
                ei, hi = edge_hz
            a, b = EDGES[ei]
            d = HORIZONS[hi]
            t = int(self.rng.integers(K - 1, T - d))
            yb, mb = self._hist(v, t, b)
            if mb.any(-1).sum() < self.min_hist:
                continue
            ya, ma = self._hist(v, t, a)
            if ma.any(-1).sum() < self.min_hist:  # common support: real peer history valid for all modes
                continue
            if mode == "g0":
                y1, m1, p1 = yb, mb, b
            elif mode == "g1":
                y1, m1, p1 = ya, ma, a
            else:  # shuffled peer: other video, random anchor, same part a
                u = self.vids[self.rng.integers(len(self.vids))]
                if u == v:
                    continue
                Tu = self.off[u + 1] - self.off[u]
                y1, m1 = self._hist(u, int(self.rng.integers(K - 1, Tu)), a); p1 = a
                if m1.any(-1).sum() < self.min_hist:
                    continue
            s = self.off[v] + t + d
            if not self.V[s, b].any():
                continue
            Y0.append(yb); M0.append(mb); Y1.append(y1); M1.append(m1); P0.append(b); P1.append(p1)
            E.append(ei); H.append(hi); YT.append(self.S[s, b]); MT.append(self.V[s, b])
        t_ = lambda x, dt=torch.float32: torch.as_tensor(np.stack(x)).to(dt)
        return (t_(Y0), t_(M0, torch.bool), t_(Y1), t_(M1, torch.bool), t_(P0, torch.long), t_(P1, torch.long),
                t_(E, torch.long), t_(H, torch.long), t_(YT), t_(MT, torch.bool))
