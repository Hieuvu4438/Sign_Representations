import math
import torch
from torch import nn

from signcoord.targets.schema import PART_RAW


def continuous_position(coords, width):
    """Sin/cos continuous coordinates; no target-derived positions."""
    axis = coords.shape[-1]
    bands = max(1, width//(2*axis))
    freq = torch.exp(torch.arange(bands, device=coords.device, dtype=coords.dtype)*(-math.log(10000)/max(bands-1, 1)))
    x = coords[..., None]*freq
    pos = torch.cat((x.sin(), x.cos()), -1).flatten(-2)
    return torch.nn.functional.pad(pos, (0, width-pos.shape[-1]))


class DecoderBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.self_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, width*mlp_ratio), nn.GELU(), nn.Linear(width*mlp_ratio, width))

    def forward(self, query, memory, token_valid):
        q = self.norms[0](query)
        query = query+self.self_attn(q, q, q, need_weights=False)[0]
        q = self.norms[1](query)
        query = query+self.cross_attn(q, memory, memory, key_padding_mask=~token_valid, need_weights=False)[0]
        return query+self.ff(self.norms[2](query))


class PartTimeDecoder(nn.Module):
    def __init__(self, encoder_dim, width=384, blocks=2, heads=6, mlp_ratio=4, dropout=0.0):
        super().__init__()
        if width % heads:
            raise ValueError("decoder width must divide attention heads")
        self.width = width
        self.memory_projection = nn.Linear(encoder_dim, width)
        self.part_embedding = nn.Parameter(torch.randn(4, width)*0.02)
        self.query_base = nn.Parameter(torch.randn(width)*0.02)
        self.blocks = nn.ModuleList([DecoderBlock(width, heads, mlp_ratio, dropout) for _ in range(blocks)])
        self.heads = nn.ModuleDict({p: nn.Linear(width, len(js)*2) for p, js in PART_RAW.items()})

    def forward(self, memory, coords, token_valid, times):
        if not token_valid.any(1).all():
            raise ValueError("decoder requires at least one valid memory token per sample")
        b, t = times.shape
        relative_times = (times-times[:, :1]).to(memory.dtype)
        time_emb = continuous_position(relative_times[..., None], self.width)
        query = time_emb[:, :, None, :]+self.part_embedding[None, None]+self.query_base
        query = query.reshape(b, t*4, self.width)
        mem = self.memory_projection(memory)+continuous_position(coords, self.width)
        for block in self.blocks:
            query = block(query, mem, token_valid)
        hidden = query.reshape(b, t, 4, self.width)
        pred = torch.cat([self.heads[part](hidden[:, :, i]).reshape(b, t, -1, 2)
                          for i, part in enumerate(PART_RAW)], dim=2)
        return pred
