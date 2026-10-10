"""H007: 32-frame SignRep student (whole-sign context in one window).

Student: Hiera-B built for input (32,224,224) (same architecture otherwise), initialised from the
released SignRep backbone; temporal position embedding linearly interpolated 8 -> 16 tokens
(the vendor code carries a commented-out interpolation for longer inputs). `fc` copied.
Targets (from the frozen 16-frame SignRep teacher) for a 32-frame window [t, t+32):
    a = T([t, t+16)), b = T([t+16, t+32))   (L2-normalised `features`)
  H7 (ordered):   head_A(z) -> a,  head_B(z) -> b
  C7 (control):   head_A(z) -> m,  head_B(z) -> m,  m = L2N(a + b)  (order-free, same compute)
Retrieval uses the student's `features` z (heads discarded).
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


def build_student32(device, mask_ratio=0.8):
    teacher, info = S.load_signrep(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    _, _, hiera_mod = S._import_vendor()
    import models.hiera.hiera_nodec_mae as nodec
    bb = nodec.MaskedAutoencoderHiera(
        num_classes=400, input_size=(32, 224, 224), q_stride=(1, 2, 2), mask_unit_size=(1, 8, 8),
        patch_kernel=(3, 7, 7), patch_stride=(2, 4, 4), patch_padding=(1, 3, 3), sep_pos_embed=True, q_pool=2,
        drop_path_rate=0.1, num_cls_tokens=0)
    sd = {k[len("backbone."):]: v for k, v in teacher.state_dict().items() if k.startswith("backbone.")}
    pt = sd["pos_embed_temporal"]  # 1,8,96
    sd["pos_embed_temporal"] = F.interpolate(pt.permute(0, 2, 1), size=16, mode="linear", align_corners=True).permute(0, 2, 1)
    res = bb.load_state_dict(sd, strict=True)
    student = nn.Module()
    student.backbone = bb
    student.fc = copy.deepcopy(teacher.fc)
    for p in student.parameters():
        p.requires_grad_(True)
    student.mask_ratio = mask_ratio
    info["student32_load"] = {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys)}
    return teacher.eval(), student.to(device), info


def student_features(student, x, train=True):
    latent, _, _ = student.backbone(x, mask_ratio=student.mask_ratio if train else 0.0)
    return student.fc(latent)


class Heads(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.a = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.b = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, z):
        return self.a(z), self.b(z)


class Win32Dataset(torch.utils.data.Dataset):
    """Unlabelled clips -> one random 32-frame window (padded by repeating the last frame)."""

    def __init__(self, rows, seed=0):
        self.rows, self.seed = rows, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rng = random.Random(hash((self.seed, i, random.random())))
        arr, _ = decode_resized(self.rows[i]["path"])
        if len(arr) < 32:
            arr = np.concatenate([arr, np.repeat(arr[-1:], 32 - len(arr), 0)])
        t = rng.randint(0, len(arr) - 32)
        return torch.from_numpy(arr[t:t + 32])


def loss_fn(student, teacher, heads, x32, ordered):
    with torch.no_grad():
        ta = F.normalize(teacher(x32[:, :, :16].contiguous())["features"].float(), dim=-1)
        tb = F.normalize(teacher(x32[:, :, 16:].contiguous())["features"].float(), dim=-1)
    z = student_features(student, x32, train=True).float()
    pa, pb = heads(z)
    pa, pb = F.normalize(pa, dim=-1), F.normalize(pb, dim=-1)
    if ordered:
        la, lb = 1 - (pa * ta).sum(-1), 1 - (pb * tb).sum(-1)
    else:
        m = F.normalize(ta + tb, dim=-1)
        la, lb = 1 - (pa * m).sum(-1), 1 - (pb * m).sum(-1)
    return (la + lb).mean() / 2, {"la": la.mean().item(), "lb": lb.mean().item(), "t_ab_cos": (ta * tb).sum(-1).mean().item()}
