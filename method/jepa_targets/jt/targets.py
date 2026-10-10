"""Dense per-token targets (token order = V-JEPA patch_embed order: t, h, w with tubelet 2, patch 16)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def pixel_targets(v):
    """v [B,3,16,H,W] normalised video -> [B, N, 2*16*16*3] per-patch normalised pixels (VideoMAE norm_pix)."""
    B, C, T, H, W = v.shape
    x = v.reshape(B, C, T // 2, 2, H // 16, 16, W // 16, 16)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, (T // 2) * (H // 16) * (W // 16), -1)
    mu, var = x.mean(-1, keepdim=True), x.var(-1, keepdim=True)
    return (x - mu) / (var + 1e-6).sqrt()


_SOBEL = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def hog_targets(v, nbins=9, cell=8):
    """MaskFeat-style HOG on grayscale frames: 9 unsigned orientation bins, 8x8 cells, 2x2 cells per 16x16
    patch, L2-normalised per patch and frame; the two frames of a tubelet are concatenated -> [B, N, 72]."""
    B, C, T, H, W = v.shape
    g = (0.299 * v[:, 0] + 0.587 * v[:, 1] + 0.114 * v[:, 2]).reshape(B * T, 1, H, W).float()
    kx = _SOBEL.to(g.device).view(1, 1, 3, 3); ky = kx.transpose(2, 3)
    gx, gy = F.conv2d(g, kx, padding=1), F.conv2d(g, ky, padding=1)
    mag = (gx ** 2 + gy ** 2).sqrt()
    ang = torch.atan2(gy, gx) % math.pi
    b = (ang / math.pi * nbins).long().clamp_(max=nbins - 1)
    hist = torch.zeros(B * T, nbins, H, W, device=g.device).scatter_(1, b, mag)
    hist = F.avg_pool2d(hist, cell) * cell * cell  # [BT, 9, H/8, W/8]
    hp, wp = H // 16, W // 16
    hist = hist.reshape(B * T, nbins, hp, 2, wp, 2).permute(0, 2, 4, 3, 5, 1).reshape(B * T, hp, wp, 4 * nbins)
    hist = F.normalize(hist, dim=-1) * math.sqrt(4 * nbins)  # ~unit per-dim scale (balances the latent L1)
    hist = hist.reshape(B, T // 2, 2, hp, wp, 4 * nbins).permute(0, 1, 3, 4, 2, 5).reshape(B, (T // 2) * hp * wp, 8 * nbins)
    return hist
