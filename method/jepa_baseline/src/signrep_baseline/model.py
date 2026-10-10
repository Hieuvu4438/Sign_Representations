"""Checkpoint-compatible SignRep; architecture reused from pinned author source."""

import math
from pathlib import Path
import torch
from torch import nn
from .schema import PRIOR_SHAPES
from .upstream_signrep.models.hiera.hiera_clstoken import Hiera
from .upstream_signrep.models.hiera.hiera_nodec_mae import mae_hiera_base_16x224
from .upstream_signrep.models.head_models.dict_head_v6_FINAL_thoughtful import ProjectionModel


class SignRep(nn.Module):
    def __init__(self, mask_ratio=0.8, drop_path_rate=0.1):
        super().__init__()
        if not 0 <= mask_ratio < 1:
            raise ValueError("Mask ratio must leave visible tokens")
        self.mask_ratio = mask_ratio
        self.frames = 16
        self.backbone = mae_hiera_base_16x224(
            num_classes=400, pretrained=False, drop_path_rate=drop_path_rate, num_cls_tokens=0
        )
        self.cls_head = ProjectionModel(
            latents_params={"hidden_dim": 512, "out_dim": 384},
            projection_params={key: {"hidden_dim": None, "output_dim": math.prod(shape)}
                               for key, shape in PRIOR_SHAPES.items()},
            total_frames=16, in_dim=768,
        )
        self.fc = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 768, bias=False))
        self.active_head = nn.Sequential(nn.Linear(768, 128), nn.GELU(), nn.Linear(128, 2))

    def encode(self, images, mask=None, mask_ratio=None):
        if images.ndim != 5 or tuple(images.shape[1:]) != (3, self.frames, 224, 224):
            raise ValueError(f"Expected B×3×{self.frames}×224×224")
        ratio = self.mask_ratio if self.training else 0.0
        if mask_ratio is not None:
            ratio = mask_ratio
        if not 0 <= ratio < 1:
            raise ValueError("Mask ratio must be in [0,1)")
        if mask is None:
            mask = self.backbone.get_random_mask(images, ratio)
        expected = (images.shape[0], math.prod(self.backbone.mask_spatial_shape))
        if mask.shape != expected or mask.dtype != torch.bool:
            raise ValueError(f"Expected boolean keep mask {expected}")
        if (mask.sum(1) == 0).any() or not (mask.sum(1) == mask.sum(1)[0]).all():
            raise ValueError("All items need the same positive number of visible mask units")
        # Passing an all-keep mask matches the released nodec encoder's zero-mask path.
        tokens, intermediates, cls = Hiera.forward(
            self.backbone, images, mask=mask, return_intermediates=True
        )
        latent = tokens.mean(dim=1)
        return {"features": self.fc(latent), "latent": latent, "tokens": tokens,
                "intermediates_feature_map": intermediates,
                "intermediate_features_cls": cls, "mask": mask}

    def forward(self, images, mask=None, mask_ratio=None):
        out = self.encode(images, mask, mask_ratio)
        if self.frames != 16:
            raise ValueError("Prior decoder is defined for 16-frame pretraining; use encode for recognition")
        out["cls_head"] = self.cls_head(out["features"])
        out["cls_active"] = self.active_head(out["features"])
        return out

    def load_released(self, checkpoint):
        obj = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if not isinstance(obj, dict) or "model" not in obj:
            raise ValueError("Expected released checkpoint with 'model' state dictionary")
        return self.load_state_dict(obj["model"], strict=True)

    def initialize_backbone(self, checkpoint):
        """Load Kinetics MAE weights, rejecting every missing encoder parameter."""
        obj = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        state = obj.get("model_state", obj.get("model", obj))
        required = self.backbone.state_dict()
        if all(k.startswith("backbone.") for k in state):
            state = {k.removeprefix("backbone."): v for k, v in state.items()}
        missing = sorted(set(required) - set(state))
        if missing:
            raise ValueError(f"Missing Kinetics encoder tensors: {missing}")
        self.backbone.load_state_dict({k: state[k] for k in required}, strict=True)
        return {"loaded": sorted(required), "excluded_pretraining_heads": sorted(set(state) - set(required))}

    def inflate_for_recognition(self):
        """Supplement A6: kernel indices 0/1/2 -> 1/3/5, retaining 8 temporal tokens."""
        if self.frames != 16:
            raise ValueError("Inflation is only valid on an uninflated 16-frame model")
        old = self.backbone.patch_embed.proj
        if old.kernel_size != (3, 7, 7) or old.stride != (2, 4, 4):
            raise ValueError("Unexpected source patch embedding")
        new = nn.Conv3d(old.in_channels, old.out_channels, (7, 7, 7),
                        stride=(8, 4, 4), padding=(3, 3, 3), bias=old.bias is not None,
                        device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :, [1, 3, 5]] = old.weight
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.backbone.patch_embed.proj = new
        self.backbone.patch_stride = (8, 4, 4)
        # Unroll/reroll depend on the output token grid, which remains unchanged.
        self.frames = 64
        return self


class SignRecognizer(nn.Module):
    def __init__(self, encoder: SignRep, classes: int):
        super().__init__()
        if classes < 2 or encoder.frames != 64:
            raise ValueError("Recognition needs an inflated encoder and at least two classes")
        self.encoder = encoder
        # Paper §7.2 discards prior decoders for recognition.
        self.encoder.cls_head = nn.Identity()
        self.encoder.active_head = nn.Identity()
        self.classifier = nn.Linear(768, classes)

    def forward(self, images):
        return self.classifier(self.encoder.encode(images, mask_ratio=0.0)["features"])
