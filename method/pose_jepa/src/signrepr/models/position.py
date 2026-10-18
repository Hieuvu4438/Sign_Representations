import math
import torch
from torch import nn


class PositionEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.part = nn.Embedding(4, dim)
        self.register_buffer("frequencies", torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim)))
        self.dim = dim

    def forward(self, position, indices):
        angles = position.float()[..., None] * self.frequencies
        result = torch.zeros((*position.shape, self.dim), device=position.device)
        result[..., 0::2] = angles.sin()
        result[..., 1::2] = angles[..., :self.dim//2].cos()
        return result + self.part(indices % 4)
