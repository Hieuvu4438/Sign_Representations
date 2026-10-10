import torch
from torch import nn


class StateHeads(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.hands = nn.ModuleList([nn.Sequential(nn.Linear(width,512),nn.GELU(),nn.Linear(512,384)) for _ in range(2)])
        self.body = nn.Sequential(nn.Linear(width,128),nn.GELU(),nn.Linear(128,14))

    def forward(self,context):
        return torch.stack([head(context[:,:,a]) for a,head in enumerate(self.hands)],dim=2),self.body(context[:,:,2])
