import torch
from torch import nn
from .part_tokenizer import PartTokenizer
from .position import PositionEncoding


def initialize(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        nn.init.trunc_normal_(module.in_proj_weight, std=0.02)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)


class TemporalTransformer(nn.Module):
    def __init__(self, dim, heads, ffn, layers):
        super().__init__()
        # Construct distinct layers instead of cloning one initialization.
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, ffn, dropout=0.0, activation="gelu", batch_first=True, norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, valid):
        active = valid.any(1)
        out = x * 0
        if bool(active.any()):
            h = x[active]
            mask = valid[active]
            for layer in self.layers:
                h = layer(h, src_key_padding_mask=~mask)
                h = torch.where(mask[..., None], h, 0)
            out = out.index_copy(0, active.nonzero().flatten(), torch.where(mask[..., None], self.norm(h), 0))
        else:
            # Keep configured parameters connected on a zero-support DDP rank.
            out = out + sum(p.sum() * 0 for p in self.parameters())
        return out


def pack_visible(tokens, visible, meta):
    b, _, d = tokens.shape
    counts = visible.sum(1)
    cap = max(1, int(counts.max()))
    indices = torch.zeros((b, cap), dtype=torch.long, device=tokens.device)
    valid = torch.arange(cap, device=tokens.device)[None] < counts[:, None]
    for i in range(b):
        ids = visible[i].nonzero().flatten()
        indices[i, :len(ids)] = ids
    x = tokens.gather(1, indices[..., None].expand(b, cap, d))
    return {"features": torch.where(valid[..., None], x, 0), "indices": indices, "valid": valid,
            "position": meta["position"].gather(1, indices)}


class PoseEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.tokenizer = PartTokenizer(cfg)
        self.position = PositionEncoding(cfg["encoder_dim"])
        self.temporal = TemporalTransformer(cfg["encoder_dim"], cfg["encoder_heads"], cfg["encoder_ffn_dim"], cfg["encoder_layers"])
        # Deduplicate shared hand modules during initialization as well as EMA.
        for module in self.modules():
            initialize(module)

    def encode_full(self, batch):
        x, meta = self.tokenizer(batch)
        x = x + self.position(meta["position"], meta["indices"])
        return {"features": self.temporal(x, meta["valid"]), **meta}

    def encode_visible(self, batch, visible):
        x, meta = self.tokenizer(batch)
        if visible.shape != meta["valid"].shape or bool((visible & ~meta["valid"]).any()):
            raise ValueError("Visible mask contains invalid tokens")
        packed = pack_visible(x, visible, meta)
        x = packed["features"] + self.position(packed["position"], packed["indices"])
        packed["features"] = self.temporal(x, packed["valid"])
        return packed

    def forward(self, batch):
        return self.encode_full(batch)


def pool_clip(encoded):
    f, v = encoded["features"], encoded["valid"]
    b, _, d = f.shape
    f, v = f.reshape(b, -1, 4, d), v.reshape(b, -1, 4)
    mean = (f * v[..., None]).sum(1) / v.sum(1).clamp_min(1)[..., None]
    return mean.flatten(1), v.any(1)
