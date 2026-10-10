"""Hiera (SignRep) backbone adapter for ASC-JEPA (spec §7).

Canonical layout: stage grids [B, T, H, W, C], flat order t-major then h then w.
Native masking: Hiera mask units over an (8,7,7) grid (1x8x8 tokens at stage 0); visible units are
kept in ascending unit-id order by the native path, and at stages 2/3 (14x14 spatial) each unit
contributes a 1x2x2 token block. We scatter visible blocks back to the canonical grid.
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
P4 = REPO / "method" / "p4_signssl"
if str(P4) not in sys.path:
    sys.path.insert(0, str(P4))
from p4 import signrep as S  # noqa: E402  (vendored SignRep loader, read-only)

STAGES = {"mid": 2, "final": 3}  # Hiera stage indices (stage_ends [1,4,20,23])
MU_GRID = (8, 7, 7)


@dataclass
class StageOut:
    grid: torch.Tensor          # [B,T,H,W,C] (zeros at hidden positions for student)
    visible: torch.Tensor       # [B,T,H,W] bool
    tokens: torch.Tensor        # [B,Nv,C] visible tokens (student) or all tokens (teacher)
    flat_idx: torch.Tensor      # [B,Nv] canonical flat indices of `tokens`


def load_signrep_backbone(device):
    model, info = S.load_signrep(device)
    bb = model.backbone
    from models.hiera.hiera_clstoken import Hiera  # vendored, on path after load
    return model, bb, Hiera, info


class HieraAdapter(nn.Module):
    def __init__(self, backbone, hiera_cls):
        super().__init__()
        self.bb = backbone
        self._hiera = hiera_cls

    def random_mask(self, B, hidden_ratio, generator=None, device="cuda"):
        n = MU_GRID[0] * MU_GRID[1] * MU_GRID[2]
        keep = max(1, min(n - 1, int(round(n * (1 - hidden_ratio)))))
        noise = torch.rand(B, n, generator=generator, device="cpu").to(device)
        idx = noise.argsort(1)[:, :keep]
        m = torch.zeros(B, n, dtype=torch.bool, device=device)
        m.scatter_(1, idx, True)
        return m  # True = keep (visible)

    def forward_full(self, x, stages=("mid", "final")):
        out, inter, _ = self._hiera.forward(self.bb, x, mask=None, return_intermediates=True)
        res = {}
        for name in stages:
            g = inter[STAGES[name]]  # B,T,H,W,C
            B, T, H, W, C = g.shape
            res[name] = StageOut(g, torch.ones(B, T, H, W, dtype=torch.bool, device=g.device),
                                 g.reshape(B, -1, C), torch.arange(T * H * W, device=g.device).expand(B, -1))
        return res, out  # out = mean of final tokens (pre-fc latent)

    def forward_masked(self, x, mask, stages=("mid", "final")):
        out, inter, _ = self._hiera.forward(self.bb, x, mask=mask, return_intermediates=True)
        res = {}
        B = x.shape[0]
        units = torch.stack([m.nonzero().squeeze(1) for m in mask])  # [B,Nmu] ascending
        ut = units // (MU_GRID[1] * MU_GRID[2]); uy = (units % (MU_GRID[1] * MU_GRID[2])) // MU_GRID[2]; ux = units % MU_GRID[2]
        for name in stages:
            blk = inter[STAGES[name]]  # B,Nmu,mt,my,mx,C
            _, nmu, mt, my, mx, C = blk.shape
            T, H, W = MU_GRID[0] * mt, MU_GRID[1] * my, MU_GRID[2] * mx
            dt, dy, dx = torch.meshgrid(torch.arange(mt, device=x.device), torch.arange(my, device=x.device),
                                        torch.arange(mx, device=x.device), indexing="ij")
            tt = ut[:, :, None, None, None] * mt + dt; yy = uy[:, :, None, None, None] * my + dy; xx = ux[:, :, None, None, None] * mx + dx
            flat = (tt * H + yy) * W + xx  # B,Nmu,mt,my,mx
            flat = flat.reshape(B, -1)
            tok = blk.reshape(B, -1, C)
            grid = torch.zeros(B, T * H * W, C, dtype=tok.dtype, device=tok.device)
            grid.scatter_(1, flat[..., None].expand(-1, -1, C), tok)
            vis = torch.zeros(B, T * H * W, dtype=torch.bool, device=tok.device)
            vis.scatter_(1, flat, True)
            res[name] = StageOut(grid.view(B, T, H, W, C), vis.view(B, T, H, W), tok, flat)
        return res, out  # out = [B,Nvis_final,C] final tokens (not pooled) when masked


def build_online_and_teacher(device):
    model, bb, hiera_cls, info = load_signrep_backbone(device)
    online = HieraAdapter(bb, hiera_cls)
    teacher = HieraAdapter(copy.deepcopy(bb), hiera_cls)
    for p in teacher.parameters():
        p.requires_grad_(False)
    for p in online.parameters():
        p.requires_grad_(True)
    teacher.eval()
    return online, teacher, model.fc, model.active_head, info
