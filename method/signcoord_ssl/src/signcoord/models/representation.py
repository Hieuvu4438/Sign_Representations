import torch
from torch import nn
from torch.nn import functional as F

from .decoder import PartTimeDecoder
from .projector import RegularizationProjector
from .pooled_prior import PooledPriorDecoder


class SignCoordModel(nn.Module):
    def __init__(self, encoder, decoder_config=None, projector_config=None, pooled_prior_frames=None):
        super().__init__()
        self.encoder = encoder
        config=dict(decoder_config or {})
        if pooled_prior_frames is None:
            self.decoder = PartTimeDecoder(encoder.dim, **config)
        else:
            config.pop("heads",None)
            self.decoder = PooledPriorDecoder(encoder.dim,pooled_prior_frames,**config)
        self.projector = RegularizationProjector(encoder.dim, **(projector_config or {}))

    def forward(self, rgb, keep_mask, frame_valid, times):
        memory, coords, valid, pooled = self.encoder.forward_visible(rgb, keep_mask, frame_valid, times)
        return {"pred_standardized": self.decoder(memory, coords, valid, times), "projected": self.projector(pooled)}

    def encode(self, rgb, frame_valid, times):
        return encode(self.encoder, rgb, frame_valid, times)


def encode(encoder, rgb, frame_valid, times):
    h, h_times, valid, pooled = encoder.encode_dense(rgb, frame_valid, times)
    if (pooled.norm(dim=-1) <= 1e-12).any():
        raise ValueError("zero-norm encoder representation")
    return {"sequence": h, "sequence_times": h_times, "sequence_valid": valid,
            "clip_raw": pooled, "clip": F.normalize(pooled, dim=-1)}
