"""RGB encoder adapter (vendored Hiera backbone + fc) and the new temporal decoder."""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .vendor import VENDOR_MODEL_PARAMS, build_vendor_backbone, get_projection_model_class

OUT_DIM = 768
T_FRAMES = 16


class RGBEncoder(nn.Module):
    """RGB -> backbone -> fc. Inputs are BCTHW. Never calls any decoder/activity head."""

    def __init__(self, backbone: nn.Module, fc: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.fc = fc

    def num_mask_units(self) -> int:
        return math.prod(self.backbone.mask_spatial_shape)

    def encode(self, rgb: torch.Tensor, mask_ratio: float = 0.0, mask: Optional[torch.Tensor] = None):
        if rgb.dim() != 5 or rgb.shape[1] != 3 or rgb.shape[2] != T_FRAMES:
            raise ValueError(f"expected BCTHW [B,3,16,224,224], got {tuple(rgb.shape)}")
        latent, _, _ = self.backbone(rgb, mask_ratio=mask_ratio, mask=mask, eval=False)
        features = self.fc(latent)
        if latent.shape[1:] != (OUT_DIM,) or features.shape[1:] != (OUT_DIM,):
            raise RuntimeError(f"unexpected latent/features shapes {latent.shape} {features.shape}")
        return {"features": features, "latent": latent}

    forward = encode


def make_explicit_mask(num_units: int, batch: int, mask_ratio: float, generator: torch.Generator) -> torch.Tensor:
    """Same semantics as Hiera.get_random_mask (1 = keep), but from an explicit CPU generator."""
    len_keep = int(num_units * (1 - mask_ratio))
    noise = torch.rand(batch, num_units, generator=generator)
    ids_restore = torch.argsort(torch.argsort(noise, dim=1), dim=1)
    mask = torch.zeros(batch, num_units)
    mask[:, :len_keep] = 1
    return torch.gather(mask, 1, ids_restore).bool()


class TemporalDecoder(nn.Module):
    """New decoder: upstream ProjectionModel (read-only class) with canonical heads only."""

    def __init__(self, vendor_root, heads: Dict[str, int], hidden_dim=512, latent_dim=384, total_frames=T_FRAMES):
        super().__init__()
        projection_model = get_projection_model_class(vendor_root)
        self.heads = dict(heads)
        self.net = projection_model(
            latents_params={"hidden_dim": hidden_dim, "out_dim": latent_dim},
            projection_params={k: {"hidden_dim": None, "output_dim": d} for k, d in heads.items()},
            total_frames=total_frames,
            in_dim=OUT_DIM,
        )

    def reset_parameters(self, seed: int) -> None:
        """xavier_uniform_zero_bias_v1, seeded independently of the global RNG stream."""
        g = torch.Generator().manual_seed(seed)
        for m in self.net.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = math.sqrt(6.0 / (fan_in + fan_out))
                with torch.no_grad():
                    m.weight.copy_(torch.rand(m.weight.shape, generator=g) * 2 * bound - bound)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.net(features)
        priors = {k: out[k] for k in self.heads}
        for k, d in self.heads.items():
            if priors[k].shape[1:] != (T_FRAMES, d):
                raise RuntimeError(f"head {k} shape {tuple(priors[k].shape)}")
        return priors


class TrainingModel(nn.Module):
    def __init__(self, encoder: RGBEncoder, decoder: TemporalDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, rgb, mask_ratio: float = 0.0, mask=None):
        enc = self.encoder.encode(rgb, mask_ratio=mask_ratio, mask=mask)
        return {**enc, "priors": self.decoder(enc["features"])}


def encoder_from_vendor_model(base_model) -> RGBEncoder:
    return RGBEncoder(base_model.backbone, base_model.fc)


def build_empty_encoder(vendor_root) -> RGBEncoder:
    """Architecture-only encoder for loading an exported encoder state dict (strict)."""
    backbone = build_vendor_backbone(vendor_root)
    fc = nn.Sequential(nn.LayerNorm(OUT_DIM), nn.Linear(OUT_DIM, OUT_DIM, bias=False))
    return RGBEncoder(backbone, fc)


def _sdpa_4d(q, k, v, *args, **kwargs):
    """Hiera passes 5-D [B,H,W,N,D] tensors to SDPA, which forces the slow math kernel.

    Folding (H,W) into one batch-like dim is mathematically identical attention
    (each (head, window) attends independently) and enables fused kernels.
    """
    import torch.nn.functional as TF

    if q.dim() == 5:
        B, H, W, N, D = q.shape
        o = TF.scaled_dot_product_attention(
            q.reshape(B, H * W, N, D), k.reshape(B, H * W, k.shape[3], D), v.reshape(B, H * W, v.shape[3], D),
            *args, **kwargs,
        )
        return o.reshape(B, H, W, N, D)
    return TF.scaled_dot_product_attention(q, k, v, *args, **kwargs)


class _FunctionalShim:
    def __getattr__(self, name):
        import torch.nn.functional as TF

        return getattr(TF, name)

    scaled_dot_product_attention = staticmethod(_sdpa_4d)


class attention_kernel:
    """Scoped swap of the vendored Hiera attention module's ``F`` symbol (restored on exit).

    Only affects ``models.hiera.hiera_clstoken`` as imported from the vendor root in
    this process; vendor source files are untouched. ``vendor_default`` is a no-op.
    """

    def __init__(self, kind: str):
        if kind not in ("vendor_default", "sdpa_4d_reshape"):
            raise ValueError(kind)
        self.kind = kind
        self._saved = None

    def __enter__(self):
        if self.kind == "sdpa_4d_reshape":
            import sys

            mod = sys.modules["models.hiera.hiera_clstoken"]
            self._mod, self._saved = mod, mod.F
            mod.F = _FunctionalShim()
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            self._mod.F = self._saved
            self._saved = None
        return False


def state_hash(module: nn.Module) -> str:
    import hashlib

    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() if v.numel() else b"")
    return h.hexdigest()


def check_constructor_literal() -> None:
    assert VENDOR_MODEL_PARAMS["out_dim"] == OUT_DIM
    assert VENDOR_MODEL_PARAMS["cls_head_params"]["total_frames"] == T_FRAMES
