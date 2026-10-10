"""Read-only SignRep Hiera backend with explicit temporal geometry rebuilding.

No dependency on any other method package and no mutation of vendor source.
"""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from signcoord.provenance import sha256_file
from .backbone_adapter import pooled_valid
from .attention import configure_attention


def import_vendor(vendor_root):
    root = Path(vendor_root).resolve()
    if not (root/"models/hiera/hiera_clstoken.py").is_file():
        raise ValueError(f"SignRep checkout not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("models.hiera.hiera_clstoken")
    if root not in Path(module.__file__).resolve().parents:
        raise RuntimeError("conflicting 'models' package; use an isolated process")
    return module.Hiera


def visible_indices(keep_mask, mu_grid, unit_shape):
    """Indices of rerolled final visible blocks in canonical time,y,x grid."""
    counts = keep_mask.sum(1)
    if not (counts == counts[0]).all() or counts[0] <= 0:
        raise ValueError("equal nonzero keep counts required")
    if keep_mask.dtype != torch.bool or keep_mask.shape[1] != math.prod(mu_grid):
        raise ValueError("incorrect mask-unit geometry")
    units = torch.stack([m.nonzero().flatten() for m in keep_mask])
    ut = units//(mu_grid[1]*mu_grid[2])
    uy = (units//mu_grid[2]) % mu_grid[1]
    ux = units % mu_grid[2]
    mt, my, mx = unit_shape
    t, h, w = tuple(a*b for a, b in zip(mu_grid, unit_shape))
    dt, dy, dx = torch.meshgrid(*(torch.arange(n, device=keep_mask.device) for n in unit_shape), indexing="ij")
    tt = ut[..., None, None, None]*mt+dt
    yy = uy[..., None, None, None]*my+dy
    xx = ux[..., None, None, None]*mx+dx
    flat = ((tt*h+yy)*w+xx).reshape(len(keep_mask), -1)
    coords = torch.stack((tt.expand_as(flat.reshape(len(keep_mask), -1, mt, my, mx)),
                          yy.expand_as(flat.reshape(len(keep_mask), -1, mt, my, mx)),
                          xx.expand_as(flat.reshape(len(keep_mask), -1, mt, my, mx))), -1).reshape(len(keep_mask), -1, 3)
    return flat, coords, (t, h, w)


class HieraAdapter(nn.Module):
    def __init__(self, vendor_root, input_frames=32, checkpoint=None, initialization_mode="signrep_pretrained", attention_kernel="vendor_5d"):
        super().__init__()
        if input_frames not in (16, 32, 64):
            raise ValueError("validated temporal geometry: 16/32/64 frames")
        if initialization_mode not in ("signrep_pretrained", "random_init"):
            raise ValueError("this backend loads SignRep weights only; generic MAE needs upstream backend")
        self.hiera_cls = import_vendor(vendor_root)
        self.bb = self.hiera_cls(input_size=(input_frames, 224, 224), num_classes=400,
                  q_stride=(1, 2, 2), mask_unit_size=(1, 8, 8), patch_kernel=(3, 7, 7),
                  patch_stride=(2, 4, 4), patch_padding=(1, 3, 3), sep_pos_embed=True,
                  q_pool=2, num_cls_tokens=0, drop_path_rate=0.1)
        # Match the public nodec-MAE encoder: norm/head are Identity, no classifier output.
        self.bb.norm, self.bb.head = nn.Identity(), nn.Identity()
        self.dim, self.input_frames = 768, input_frames
        self.mu_grid = tuple(self.bb.mask_spatial_shape)
        self.num_mask_units = math.prod(self.mu_grid)
        self.load_report = {"initialization_mode": initialization_mode, "adapted_keys": []}
        if initialization_mode == "signrep_pretrained":
            if checkpoint is None:
                raise ValueError("SignRep initialization requires an explicit checkpoint")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if "model" not in state:
                raise ValueError("expected official SignRep checkpoint['model']")
            original = state["model"]
            state = {k.removeprefix("backbone."): v for k, v in original.items() if k.startswith("backbone.")}
            excluded = [k for k in original if not k.startswith("backbone.")]
            if any(not k.startswith(("fc.", "active_head.", "cls_head.")) for k in excluded):
                raise ValueError("unrecognized non-encoder keys in checkpoint")
            if state["pos_embed_temporal"].shape != self.bb.pos_embed_temporal.shape:
                state["pos_embed_temporal"] = F.interpolate(state["pos_embed_temporal"].transpose(1, 2),
                       size=self.bb.pos_embed_temporal.shape[1], mode="linear", align_corners=False).transpose(1, 2)
                self.load_report["adapted_keys"].append("pos_embed_temporal")
            result = self.bb.load_state_dict(state, strict=True)
            self.load_report.update(checkpoint_sha256=sha256_file(checkpoint),
                 missing_keys=list(result.missing_keys), unexpected_keys=list(result.unexpected_keys),
                 excluded_training_keys=excluded)
        assert tuple(self.bb.patch_stride) == (2, 4, 4)
        configure_attention(self.bb,attention_kernel)
        self.load_report["attention_kernel"]=attention_kernel

    def random_mask(self, batch, ratio, device,generator=None):
        n = self.num_mask_units
        k = int(n*(1-ratio))
        if not 0 < k <= n:
            raise ValueError("mask leaves zero visible units")
        idx = torch.rand(batch, n, device=device,generator=generator).argsort(1)[:, :k]
        return torch.zeros(batch, n, dtype=torch.bool, device=device).scatter_(1, idx, True)

    def _forward_grid(self, rgb, keep_mask):
        if rgb.shape[2:] != (self.input_frames, 224, 224):
            raise ValueError("Hiera input geometry differs from rebuilt adapter")
        _, inter, _ = self.bb(rgb, mask=keep_mask, return_intermediates=True)
        return inter[-1]

    def forward_visible(self, rgb, keep_mask, frame_valid, times):
        if keep_mask.dtype != torch.bool:
            raise ValueError("True means keep; use a boolean mask")
        # Validate counts before invoking native Hiera's boolean-index reshape.
        if not keep_mask.any(1).all() or not (keep_mask.sum(1) == keep_mask.sum(1)[0]).all():
            raise ValueError("equal nonzero visible counts required")
        blocks = self._forward_grid(rgb, keep_mask)
        _, _, mt, my, mx, d = blocks.shape
        _, indices, grid = visible_indices(keep_mask, self.mu_grid, (mt, my, mx))
        tokens = blocks.reshape(len(rgb), -1, d)
        center_idx = (indices[..., 0]*self.bb.patch_stride[0]).clamp_max(self.input_frames-1)
        valid = frame_valid.gather(1, center_idx)
        coords = indices.to(tokens.dtype)/tokens.new_tensor([max(n-1, 1) for n in grid])
        return tokens, coords, valid, pooled_valid(tokens, valid)

    def encode_dense(self, rgb, frame_valid, times):
        grid = self._forward_grid(rgb, None)  # B,T_enc,H,W,D; pre-classifier
        h = grid.mean((2, 3))
        centers = torch.arange(h.shape[1], device=rgb.device)*self.bb.patch_stride[0]
        valid, h_times = frame_valid[:, centers], times[:, centers]
        return h, h_times, valid, pooled_valid(h, valid)
