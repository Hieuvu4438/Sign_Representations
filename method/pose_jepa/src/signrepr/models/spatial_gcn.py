import torch
from torch import nn
from .graphs import adjacency


class SpatialBlock(nn.Module):
    def __init__(self, cin, cout, a):
        super().__init__()
        self.register_buffer("adjacency", a)
        self.self_linear = nn.Linear(cin, cout)
        self.neighbor_linear = nn.Linear(cin, cout)
        self.residual = nn.Identity() if cin == cout else nn.Linear(cin, cout)
        self.norm = nn.LayerNorm(cout)

    def forward(self, x, valid):
        x = torch.where(valid[..., None], x, 0)
        neighbors = torch.einsum("vu,btuc->btvc", self.adjacency, self.neighbor_linear(x) * valid[..., None])
        message = self.self_linear(x) + neighbors + self.residual(x)
        return torch.where(valid[..., None], torch.nn.functional.gelu(self.norm(message)), 0)


class SpatialGCN(nn.Module):
    """Every operation is confined to one frame and one part."""
    def __init__(self, joints, edges, channels, dim, anchor_dim, aggregation="flatten_project", use_anchor=True):
        super().__init__()
        self.input = nn.Linear(3, channels[0])
        self.blocks = nn.ModuleList([SpatialBlock(a, b, adjacency(joints, edges)) for a, b in zip(channels[:-1], channels[1:])])
        self.aggregation = aggregation
        if aggregation not in ("flatten_project", "mean"):
            raise ValueError("Unknown joint aggregation")
        self.aggregate = nn.Sequential(nn.Linear(channels[-1] * (joints if aggregation == "flatten_project" else 1), dim), nn.LayerNorm(dim), nn.GELU())
        self.anchor = nn.Sequential(nn.Linear(3, anchor_dim), nn.GELU(), nn.Linear(anchor_dim, anchor_dim)) if use_anchor else None
        self.anchor_dim = anchor_dim
        self.fuse = nn.Linear(dim + anchor_dim, dim)

    def forward(self, stream):
        valid = stream["joint_valid"] & stream["frame_valid"][..., None]
        x = torch.cat([stream["xy"], stream["confidence"][..., None]], -1)
        x = self.input(torch.where(valid[..., None], x, 0)) * valid[..., None]
        for block in self.blocks:
            x = block(x, valid)
        if self.aggregation == "flatten_project":
            x = x.flatten(-2)
        else:
            x = x.sum(-2) / valid.sum(-1, keepdim=True).clamp_min(1)
        x = self.aggregate(x)
        anchor = self.anchor(stream["anchor"]) if self.anchor is not None else x.new_zeros((*x.shape[:-1], self.anchor_dim))
        return torch.where(stream["frame_valid"][..., None], self.fuse(torch.cat([x, anchor], -1)), 0)
