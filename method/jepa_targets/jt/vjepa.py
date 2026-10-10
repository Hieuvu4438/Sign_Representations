"""V-JEPA 2.1 ViT-B (distilled from ViT-G) encoder + predictor for continued pretraining on sign video.

All arms share: the official encoder weights (`ema_encoder` key), the official 12-block predictor body
(predictor_embed, blocks, norm, mask tokens) and the official multi-block mask generators. Only the
predictor's output projection is new, sized to the arm's target (EMA latents, pixels, ...), because
the released predictor projects to ViT-G teacher features (4 x 416 = 1664) that we do not have.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "vjepa2_src"
CKPT = ROOT.parents[0] / "signrel_jepa" / "assets" / "vjepa2_1_vitb_dist_vitG_384.pt"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from app.vjepa_2_1.models import predictor as vit_pred  # noqa: E402
from app.vjepa_2_1.models import vision_transformer as video_vit  # noqa: E402
from src.masks.multiseq_multiblock3d import _MaskGenerator  # noqa: E402
from src.masks.utils import apply_masks  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
# official vitb16 2.1 pretrain masks (configs/train_2_1/vitb16/pretrain-256px-16f.yaml)
MASK_CFG = [dict(spatial_pred_mask_scale=(0.15, 0.15), aspect_ratio=(0.75, 1.5), npred=8),
            dict(spatial_pred_mask_scale=(0.7, 0.7), aspect_ratio=(0.75, 1.5), npred=2)]


def _strip(sd):
    return {k.removeprefix("module.").removeprefix("backbone."): v for k, v in sd.items()}


def build_encoder(crop=224, frames=16, ckpt_key="ema_encoder", pretrained=True, checkpointing=True):
    enc = video_vit.vit_base(img_size=(crop, crop), num_frames=frames, patch_size=16, tubelet_size=2,
                             use_rope=True, use_sdpa=True, use_activation_checkpointing=checkpointing,
                             uniform_power=True, img_temporal_dim_size=1, interpolate_rope=True,
                             n_output_distillation=1, modality_embedding=True)
    if pretrained:
        sd = _strip(torch.load(CKPT, map_location="cpu", weights_only=False)[ckpt_key])
        sd.pop("pos_embed", None)
        enc.load_state_dict(sd, strict=True)
    return enc


def build_predictor(target_dim, crop=224, frames=16, pretrained=True, checkpointing=True, context_head=True):
    """n_output_distillation=1 as in the released checkpoint (single-level projection, 384 -> target_dim)."""
    pred = vit_pred.vit_predictor(img_size=(crop, crop), use_mask_tokens=True, patch_size=16, num_frames=frames,
                                  tubelet_size=2, embed_dim=768, predictor_embed_dim=384, depth=12, num_heads=12,
                                  uniform_power=True, num_mask_tokens=8, zero_init_mask_tokens=True, use_rope=True,
                                  use_sdpa=True, use_activation_checkpointing=checkpointing,
                                  return_all_tokens=context_head, interpolate_rope=True, modality_embedding=True,
                                  img_temporal_dim_size=1, out_embed_dim=target_dim,
                                  n_output_distillation=1)
    if pretrained:
        sd = _strip(torch.load(CKPT, map_location="cpu", weights_only=False)["predictor"])
        sd = {k: v for k, v in sd.items() if not k.startswith("predictor_proj")}  # new output heads
        missing, unexpected = pred.load_state_dict(sd, strict=False)
        assert all(k.startswith("predictor_proj") for k in missing), missing
        assert not unexpected, unexpected
    return pred


class Masks:
    """Official multi-block masks: returns ([enc idx per generator], [pred idx per generator])."""

    def __init__(self, crop=224, frames=16, seed=0):
        self.gens = [_MaskGenerator(crop_size=(crop, crop), num_frames=frames, spatial_patch_size=(16, 16),
                                    temporal_patch_size=2, temporal_pred_mask_scale=(1.0, 1.0),
                                    max_context_frames_ratio=1.0, max_keep=None, full_complement=False, **c)
                     for c in MASK_CFG]
        for i, g in enumerate(self.gens):  # block-size seed stream (block positions use the global torch RNG)
            g._itr_counter.value = seed * 1_000_003 + i * 100_003 - 1

    def __call__(self, B, device):
        enc, pred = [], []
        for g in self.gens:
            e, p = g(B)
            enc.append(e.to(device)); pred.append(p.to(device))
        return enc, pred


def normalize_video(x_uint8):
    """[B,T,H,W,3] uint8 -> [B,3,T,H,W] float, ImageNet mean/std (V-JEPA convention)."""
    x = x_uint8.permute(0, 4, 1, 2, 3).float().div_(255.0)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


__all__ = ["build_encoder", "build_predictor", "Masks", "apply_masks", "normalize_video", "nn"]
