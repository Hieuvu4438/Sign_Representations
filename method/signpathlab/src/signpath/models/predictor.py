"""Decoder queries are functions of declared public schedule metadata only."""
from dataclasses import dataclass
import torch
from torch import nn

from ..geometry.paths import DEFAULT_PATHS
from .encoder import time_encoding
from .packing import pack_tokens, scatter_packed


@dataclass(frozen=True)
class PredictionOutput:
    semantic: torch.Tensor
    geometry: torch.Tensor
    cluster: tuple[torch.Tensor, ...]
    query_metadata: torch.Tensor


class QueryPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        model, pred = config["model"], config["model"]["predictor"]
        self.dimension = d = pred["embed_dim"]
        self.query_types = nn.Embedding(7 + len(DEFAULT_PATHS), d)
        self.scales = nn.Embedding(len(config["geometry"]["window_seconds"]), d)
        self.mask_token = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.mask_token, std=.02)
        self.blocks = nn.ModuleList(nn.TransformerDecoderLayer(d, pred["heads"], d * model["encoder"]["mlp_ratio"],
            model["encoder"]["dropout"], activation="gelu", batch_first=True, norm_first=model["encoder"]["norm_first"]) for _ in range(pred["layers"]))
        self.semantic_head = nn.Linear(d, d)
        self.geometry_head = nn.Linear(d, pred["geometry_output_dim"])
        self.cluster_heads = nn.ModuleList(nn.Linear(d, 256) for _ in range(7)) if config["loss"]["cluster_weight"] else nn.ModuleList()

    def query_metadata(self, batch):
        raw = self.query_types.weight[:7][None, None] + time_encoding(batch.relative_seconds, self.dimension)[:, :, None]
        raw = raw.flatten(1, 2)
        geo = self.query_types(batch.geometry_path + 7) + self.scales(batch.geometry_scale)
        for i in range(3):
            geo = geo + time_encoding(batch.geometry_times[..., i], self.dimension) / 3
        return torch.cat([raw, geo], dim=1) + self.mask_token

    def forward(self, context, batch, *, geometry_enabled=True):
        raw_slots = batch.valid.shape[1] * 7
        queries = self.query_metadata(batch)
        geo_keep = batch.geometry_targets if geometry_enabled else torch.zeros_like(batch.geometry_targets)
        keep = torch.cat([batch.semantic_targets.flatten(1), geo_keep], dim=1)
        packed = pack_tokens(queries, keep)
        x = packed.values
        for block in self.blocks:
            x = block(x, context.context, tgt_key_padding_mask=packed.padding,
                      memory_key_padding_mask=context.context_padding)
        x = torch.where(packed.valid[..., None], x, 0)
        expanded = scatter_packed(x, packed, queries.shape[1])
        semantic = self.semantic_head(expanded[:, :raw_slots])
        geometry = self.geometry_head(expanded[:, raw_slots:])
        clusters = tuple(head(expanded[:, stream:raw_slots:7]) for stream, head in enumerate(self.cluster_heads))
        return PredictionOutput(semantic, geometry, clusters, queries)
