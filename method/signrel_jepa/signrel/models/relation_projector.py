import torch
from torch import nn


class RelationProjector(nn.Module):
    def __init__(self, dim, relation_dim=256, mode="concat_diff_product"):
        super().__init__()
        if mode not in {"concat","concat_diff_product"}:
            raise ValueError(mode)
        self.mode = mode
        n = dim*(2 if mode == "concat" else 4)
        self.net = nn.Sequential(nn.LayerNorm(n),nn.Linear(n,2*relation_dim),nn.GELU(),
                                 nn.Linear(2*relation_dim,relation_dim),nn.LayerNorm(relation_dim))

    def forward(self, a, b):
        u = [a,b] if self.mode == "concat" else [a,b,a-b,a*b]
        return self.net(torch.cat(u,-1))
