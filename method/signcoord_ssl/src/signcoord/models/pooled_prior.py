"""A1-inspired: reconstruct the shared geometry sequence from one pooled vector.

This uses the proposal's 2D schema/loss, not unpublished SignRep prior targets.
"""
import torch
from torch import nn

from .backbone_adapter import pooled_valid


class PooledPriorDecoder(nn.Module):
    def __init__(self, encoder_dim, input_frames, width=384, blocks=2, mlp_ratio=4, dropout=0.):
        super().__init__()
        self.input_frames = input_frames
        layers = [nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim,width), nn.GELU()]
        for _ in range(blocks):
            layers += [nn.Linear(width,width*mlp_ratio),nn.GELU(),nn.Dropout(dropout),
                       nn.Linear(width*mlp_ratio,width),nn.GELU()]
        layers += [nn.Linear(width,input_frames*119*2)]
        self.head = nn.Sequential(*layers)

    def forward(self, memory, coords, valid, times):
        if times.shape[1] != self.input_frames:
            raise ValueError("pooled prior output geometry differs from input context")
        return self.head(pooled_valid(memory,valid)).reshape(len(memory),self.input_frames,119,2)
