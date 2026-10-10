"""Pinned upstream video MAE encoder; final stage before MAE fusion/norm.

MAE fusion and reconstruction weights are deliberately excluded. This is a
shared final-stage representation protocol, not the upstream MAE readout.
"""
import importlib
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from signcoord.provenance import sha256_file
from .hiera_adapter import HieraAdapter
from .attention import configure_attention

UPSTREAM_REVISION = "b12b842542ee5c757fcfec8c41f6b56fcbe89b65"
SOURCE_HASHES = {
    "hiera/hiera.py": "44b7efc44ca86e9df3b0682450bd2d98d6fb10d8be9d89be61ca492cc87dec94",
    "hiera/hiera_utils.py": "2e256cc22b27990a2b54e6a1bde1b9560eefef8dce67cf9b3f42bcab1953f3b6",
}
OFFICIAL_WEIGHT_SHA256 = "dd80be81ca35a575511e7b5204cac07368c3e497c79dbe0da9fe339fb86f6dea"


def import_upstream(root):
    root = Path(root).resolve()
    if not (root / "hiera/hiera.py").is_file():
        raise ValueError(f"upstream Hiera checkout not found: {root}")
    for relative, expected in SOURCE_HASHES.items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"upstream source differs from pinned revision: {relative}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("hiera.hiera")
    if root not in Path(module.__file__).resolve().parents:
        raise RuntimeError("conflicting 'hiera' package; use an isolated process")
    return module.Hiera


def encoder_state_from_mae(original, expected, temporal_size):
    """Strict boundary: every encoder key required, only known MAE keys omitted."""
    excluded = sorted(set(original) - set(expected))
    allowed = ("encoder_norm.", "multi_scale_fusion_heads.", "decoder_embed.",
               "decoder_blocks.", "decoder_norm.", "decoder_pred.")
    if any(k not in ("mask_token", "decoder_pos_embed") and not k.startswith(allowed) for k in excluded):
        raise ValueError("unrecognized non-encoder keys in generic MAE checkpoint")
    missing = sorted(set(expected) - set(original))
    if missing:
        raise ValueError(f"missing generic encoder keys: {missing}")
    state = {k: original[k] for k in expected}
    adapted = []
    key = "pos_embed_temporal"
    if state[key].shape != expected[key].shape:
        if state[key].shape != (1, 8, 96):
            raise ValueError("generic checkpoint must have official T16 temporal positions")
        state[key] = F.interpolate(state[key].transpose(1, 2), size=temporal_size,
                                   mode="linear", align_corners=False).transpose(1, 2)
        adapted.append(key)
    return state, excluded, adapted


class GenericHieraAdapter(HieraAdapter):
    def __init__(self, vendor_root, input_frames=32, checkpoint=None,
                 initialization_mode="generic_hiera_mae",attention_kernel="vendor_5d"):
        nn.Module.__init__(self)
        if input_frames not in (16, 32, 64):
            raise ValueError("validated temporal geometry: 16/32/64 frames")
        if initialization_mode not in ("generic_hiera_mae", "random_init"):
            raise ValueError("generic backend requires generic MAE or explicit random init")
        core = import_upstream(vendor_root)
        self.bb = core(input_size=(input_frames, 224, 224), embed_dim=96,
                       num_heads=1, stages=(2, 3, 16, 3), num_classes=400,
                       q_stride=(1, 2, 2), mask_unit_size=(1, 8, 8),
                       patch_kernel=(3, 7, 7), patch_stride=(2, 4, 4),
                       patch_padding=(1, 3, 3), sep_pos_embed=True,
                       q_pool=2, drop_path_rate=0.1)
        self.bb.norm, self.bb.head = nn.Identity(), nn.Identity()
        self.dim, self.input_frames = 768, input_frames
        self.mu_grid = tuple(self.bb.mask_spatial_shape)
        self.num_mask_units = math.prod(self.mu_grid)
        self.load_report = {"initialization_mode": initialization_mode,
                            "source_revision": UPSTREAM_REVISION,
                            "feature_layer": "final_stage_before_MAE_fusion_and_norm",
                            "adapted_keys": []}
        if initialization_mode == "generic_hiera_mae":
            if checkpoint is None:
                raise ValueError("generic initialization requires an explicit video MAE checkpoint")
            checkpoint_hash = sha256_file(checkpoint)
            if checkpoint_hash != OFFICIAL_WEIGHT_SHA256:
                raise ValueError("generic initialization requires the pinned official video MAE weights")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if "model_state" not in saved:
                raise ValueError("expected official generic MAE checkpoint['model_state']")
            state, excluded, adapted = encoder_state_from_mae(
                saved["model_state"], self.bb.state_dict(), input_frames // 2)
            result = self.bb.load_state_dict(state, strict=True)
            self.load_report.update(checkpoint_sha256=checkpoint_hash,
                                    missing_keys=list(result.missing_keys),
                                    unexpected_keys=list(result.unexpected_keys),
                                    excluded_training_keys=excluded, adapted_keys=adapted)
        assert tuple(self.bb.patch_stride) == (2, 4, 4)
        assert self.bb.q_pool == 2
        configure_attention(self.bb,attention_kernel)
        self.load_report["attention_kernel"]=attention_kernel

    def _forward_grid(self, rgb, keep_mask):
        if rgb.shape[2:] != (self.input_frames, 224, 224):
            raise ValueError("Hiera input geometry differs from rebuilt adapter")
        _, intermediates = self.bb(rgb, mask=keep_mask, return_intermediates=True)
        return intermediates[-1]
