"""Proposal A (and C's RGB part): V-JEPA 2.1 ViT-B student + EMA teacher + NEW 4-block predictor (proposal11 §7.2-7.3).

recipe "AR": target = final encoder layer (n_out = 1)
recipe "AD": targets = blocks 3, 6, 9, 12 (n_out = 4; the encoder's own norms_block layers from the checkpoint);
             predictor trunk is shared, its projection has n_out x 768 outputs (4 output heads), plus a context head.
The official released predictor is NOT loaded (it projects to ViT-G features); the 4-block predictor is new.
"""
from __future__ import annotations

import copy
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from oj.paths import VENDOR_VJEPA, VJEPA_CKPT

if str(VENDOR_VJEPA) not in sys.path:
    sys.path.insert(0, str(VENDOR_VJEPA))
from app.vjepa_2_1.models import predictor as vit_pred  # noqa: E402
from app.vjepa_2_1.models import vision_transformer as video_vit  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
D = 768


def normalize_video(x_uint8):
    """[B,T,H,W,3] uint8 -> [B,3,T,H,W] float with the checkpoint's ImageNet mean/std."""
    x = x_uint8.permute(0, 4, 1, 2, 3).float().div_(255.0)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def build_encoder(size=224, frames=16, n_out=1, pretrained=True, checkpointing=True):
    enc = video_vit.vit_base(img_size=(size, size), num_frames=frames, patch_size=16, tubelet_size=2, use_rope=True,
                             use_sdpa=True, use_activation_checkpointing=checkpointing, uniform_power=True,
                             img_temporal_dim_size=1, interpolate_rope=True, n_output_distillation=n_out,
                             modality_embedding=True)
    if pretrained:
        sd = torch.load(VJEPA_CKPT, map_location="cpu", weights_only=False)["ema_encoder"]
        sd = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in sd.items()}
        sd.pop("pos_embed", None)
        enc.load_state_dict(sd, strict=True)
    return enc


def build_predictor(size=224, frames=16, n_out=1, depth=4, width=384, heads=6, checkpointing=True):
    return vit_pred.vit_predictor(img_size=(size, size), use_mask_tokens=True, patch_size=16, num_frames=frames,
                                  tubelet_size=2, embed_dim=D, predictor_embed_dim=width, depth=depth, num_heads=heads,
                                  uniform_power=True, num_mask_tokens=1, zero_init_mask_tokens=True, use_rope=True,
                                  use_sdpa=True, use_activation_checkpointing=checkpointing, return_all_tokens=True,
                                  interpolate_rope=True, modality_embedding=True, img_temporal_dim_size=1,
                                  out_embed_dim=D, n_output_distillation=n_out)


def momentum(step, total, m0=0.996, m1=0.9999):
    """§11.2: m_s = m1 - (m1 - m0) (1 + cos(pi s / S)) / 2."""
    return m1 - (m1 - m0) * (1 + math.cos(math.pi * min(step, total) / total)) / 2


class RGBJepa(nn.Module):
    def __init__(self, size=224, frames=16, recipe="AR", pretrained=True):
        super().__init__()
        assert recipe in ("AR", "AD")
        self.n_out = 1 if recipe == "AR" else 4
        self.student = build_encoder(size, frames, self.n_out, pretrained)
        self.teacher = copy.deepcopy(self.student).eval()  # EMA initialised from the LOADED student (§11.2)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.predictor = build_predictor(size, frames, self.n_out)

    @torch.no_grad()
    def targets(self, x):
        h = self.teacher(x, training=True).float()  # [B, N, n_out*768]
        return torch.cat([F.layer_norm(h[..., i * D:(i + 1) * D], (D,)) for i in range(self.n_out)], -1)  # LN without affine

    def forward(self, x, ctx, tgt):
        """x [B,3,T,H,W]; ctx [B,K] / tgt [B,M] token indices -> (pred_tgt [B,M,n_out*768], pred_ctx [B,K,n_out*768], z_ctx)."""
        z = self.student(x, masks=[ctx], training=True)
        p_t, p_c = self.predictor(z, [ctx], [tgt], mask_index=0)
        return p_t, p_c, z

    @torch.no_grad()
    def ema_update(self, m):
        for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)


def gather(x, idx):
    return torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))
