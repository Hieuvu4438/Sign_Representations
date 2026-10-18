from torch import nn
import torch
from ..data.schema import PART_ORDER, PART_INDICES
from .pose_encoder import initialize


class PhysicalHeads(nn.Module):
    def __init__(self, dim):
        super().__init__()
        def head(j):
            return nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 2*j*2))
        hand = head(21)
        self.parts = nn.ModuleDict({"body": head(9), "left_hand": hand, "right_hand": hand, "face": head(18)})
        for module in self.modules():
            initialize(module)

    def forward(self, hidden):
        b, k, _, _ = hidden.shape
        return {p: self.parts[p](hidden[:, :, i]).reshape(b, k*2, len(PART_INDICES[p]), 2) for i, p in enumerate(PART_ORDER)}


class MetricHead(nn.Module):
    def __init__(self, raw_dim=1024, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(raw_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, output_dim))
        self.apply(initialize)

    def forward(self, x):
        return torch.nn.functional.normalize(self.net(x), dim=-1)
