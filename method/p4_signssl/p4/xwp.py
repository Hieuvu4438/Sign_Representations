"""Cross-window latent prediction (XWP) continuation of SignRep — H002 MVP.

Student  s(.) : SignRep backbone + fc, initialised from the released checkpoint, trained
               with the original masked input (mask_ratio, drop_path as in pretraining).
Teacher  T(.) : frozen copy of the released SignRep, full (unmasked) 16-frame windows.
Predictor P   : small MLP conditioned on the signed offset Δ (in frames).

For a window A starting at t and an offset Δ (Δ=0 for the matched control):
    L = (1 - cos(s(A), T(A)))                      # anchor: stay close to the original space
      + λ (1 - cos(P(s(A), Δ), T(A shifted by Δ))) # predict the teacher embedding of another window
Targets come from a frozen network, so the objective cannot collapse; P is discarded at
inference (retrieval uses s(.) features exactly like SignRep).
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import signrep as S
from .data import decode_resized


class DeltaPredictor(nn.Module):
    def __init__(self, deltas, dim=768, hidden=1536):
        super().__init__()
        self.deltas = list(deltas)
        self.emb = nn.Embedding(len(self.deltas), dim)
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.emb.weight)

    def forward(self, z, delta_idx):
        return z + self.net(z + self.emb(delta_idx))


class PairDataset(torch.utils.data.Dataset):
    """Unlabelled clips (labels never read). Returns two 16-frame windows A and A+Δ."""

    def __init__(self, rows, deltas, length=16, seed=0):
        self.rows, self.deltas, self.L = rows, list(deltas), length
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rng = random.Random(hash((self.seed, i, random.random())))
        arr, _ = decode_resized(self.rows[i]["path"])
        n = arr.shape[0]
        if n < self.L:
            arr = np.concatenate([arr, np.repeat(arr[-1:], self.L - n, 0)])
            n = self.L
        di = rng.randrange(len(self.deltas))
        d = self.deltas[di]
        lo, hi = max(0, -d), min(n - self.L, n - self.L - d)
        if hi < lo:  # clip too short for this offset -> fall back to Δ index of 0 if present
            di = self.deltas.index(0) if 0 in self.deltas else di
            d = self.deltas[di] if 0 in self.deltas else 0
            lo, hi = 0, n - self.L
        t = rng.randint(lo, hi)
        a = arr[t : t + self.L]
        b = arr[t + d : t + d + self.L]
        return torch.from_numpy(a), torch.from_numpy(b), di


def build_models(device, mask_ratio=0.8):
    teacher, info = S.load_signrep(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = copy.deepcopy(teacher)
    student.mask_ratio = mask_ratio
    for p in student.parameters():  # deepcopy inherits the teacher's frozen flags
        p.requires_grad_(True)
    # heads not used by the objective are frozen so they stay identical to the original
    for p in list(student.cls_head.parameters()) + list(student.active_head.parameters()):
        p.requires_grad_(False)
    return teacher.eval(), student, info


def to_input(u8, device):
    """[B,L,H,W,3] uint8 -> [B,3,L,H,W] normalized."""
    B, L = u8.shape[:2]
    x = S.normalize_uint8(u8.reshape(B * L, *u8.shape[2:]), device)
    return x.view(B, L, 3, 224, 224).permute(0, 2, 1, 3, 4).contiguous()


def param_groups(student, predictor, lr, lr_pred, layer_decay=0.85, wd=0.05):
    blocks = student.backbone.blocks
    nb = len(blocks)
    groups = []
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone.blocks."):
            lid = int(name.split(".")[2]) + 1
        elif name.startswith("backbone."):
            lid = 0
        else:
            lid = nb + 1
        scale = layer_decay ** (nb + 1 - lid)
        groups.append({"params": [p], "lr": lr * scale, "weight_decay": 0.0 if p.ndim == 1 else wd, "base_lr": lr * scale})
    groups.append({"params": list(predictor.parameters()), "lr": lr_pred, "weight_decay": wd, "base_lr": lr_pred})
    return groups


def xwp_loss(student, teacher, predictor, a, b, di, lam, delta_zero_idx=None):
    with torch.no_grad():
        ta = F.normalize(teacher(a)["features"].float(), dim=-1)
        tb = F.normalize(teacher(b)["features"].float(), dim=-1)
    sa = student(a)["features"].float()
    anchor = 1 - (F.normalize(sa, dim=-1) * ta).sum(-1)
    pred = F.normalize(predictor(sa, di).float(), dim=-1)
    pl = 1 - (pred * tb).sum(-1)
    loss = anchor.mean() + lam * pl.mean()
    return loss, {"anchor": anchor.mean().item(), "pred": pl.mean().item(),
                  "t_ab_cos": (ta * tb).sum(-1).mean().item()}
