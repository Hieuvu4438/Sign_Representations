import torch
from torch import nn
from .pose_encoder import TemporalTransformer, initialize
from .position import PositionEncoding


class Predictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["predictor_dim"]
        self.project = nn.Linear(cfg["encoder_dim"], d)
        self.mask_embedding = nn.Embedding(4, d)
        self.position = PositionEncoding(d)
        self.temporal = TemporalTransformer(d, cfg["predictor_heads"], cfg["predictor_ffn_dim"], cfg["predictor_layers"])
        self.latent = nn.Linear(d, cfg["latent_dim"])
        self.apply(initialize)

    def forward(self, context, meta, target_mask):
        b, n = meta["valid"].shape
        if bool((target_mask & ~meta["valid"]).any()):
            raise ValueError("Targets must be naturally valid")
        h = self.project(context["features"])
        full = h.new_zeros((b, n, h.shape[-1]))
        # scatter_add handles packed pad indices without overwriting real index 0.
        full = full.scatter_add(1, context["indices"][..., None].expand_as(h), h * context["valid"][..., None])
        full = torch.where(target_mask[..., None], self.mask_embedding(meta["indices"] % 4), full)
        full = full + self.position(meta["position"], meta["indices"])
        hidden = self.temporal(full, meta["valid"])
        logits = torch.where(meta["valid"][..., None], self.latent(hidden), 0)
        return {"hidden": hidden.reshape(b, meta["segments"], 4, -1), "latent": logits}
