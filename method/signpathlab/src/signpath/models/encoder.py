"""Per-stream projectors and a bidirectional encoder over packed valid tokens."""
from dataclasses import dataclass
import math
import torch
from torch import nn

from .packing import pack_tokens, scatter_packed
from .aliases import source_aliases


def time_encoding(seconds, dimension):
    # Seconds are relative to the original segment start, including after masking.
    frequencies = torch.exp(torch.arange(0, dimension, 2, device=seconds.device, dtype=torch.float32) * (-math.log(10000.0) / dimension))
    angle = seconds.float()[..., None] * frequencies
    encoded = torch.stack((angle.sin(), angle.cos()), dim=-1).flatten(-2)
    return encoded[..., :dimension]


@dataclass(frozen=True)
class EncoderOutput:
    tokens: torch.Tensor  # [B,T*7,D], zero outside retained slots
    context: torch.Tensor
    context_padding: torch.Tensor
    context_valid: torch.Tensor
    layers: dict[int, torch.Tensor]


class SignEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        model, enc = config["model"], config["model"]["encoder"]
        self.dimension = d = model["embed_dim"]
        hidden = model["projection_hidden_dim"]
        self.projectors = nn.ModuleList(nn.Sequential(nn.Linear(f, hidden), nn.GELU(), nn.Linear(hidden, d), nn.LayerNorm(d)) for f in model["raw_dimensions"])
        self.type_embeddings = nn.Embedding(7, d) if model["type_embeddings"] else None
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(d, enc["heads"], d * enc["mlp_ratio"], enc["dropout"],
            activation="gelu", batch_first=True, norm_first=enc["norm_first"]) for _ in range(enc["layers"]))
        self.norm = nn.LayerNorm(d)
        self.return_layers = set(enc["return_layers"])

    def forward(self, batch, keep, return_layers=False):
        if keep.shape != batch.valid.shape or (keep & ~batch.valid).any():
            raise ValueError("Encoder may retain only real valid observations")
        first, unique = source_aliases(batch.source_frame_index, keep.any(dim=2))
        requested = keep
        keep = keep & unique[..., None]
        streams = []
        time = time_encoding(batch.relative_seconds, self.dimension)
        for stream, projector in enumerate(self.projectors):
            # Hidden/padded values never enter a learned projector.
            values = torch.where(keep[:, :, stream, None], batch.streams[stream], 0)
            projected = projector(values) + time
            if self.type_embeddings is not None:
                projected = projected + self.type_embeddings.weight[stream]
            streams.append(projected)
        flat = torch.stack(streams, dim=2).flatten(1, 2)
        packed = pack_tokens(flat, keep.flatten(1))
        def restore_aliases(values):
            dense = scatter_packed(values, packed, flat.shape[1]).reshape(*keep.shape, self.dimension)
            dense = dense.gather(1, first[..., None, None].expand_as(dense))
            return torch.where(requested[..., None], dense, 0).flatten(1, 2)
        x, layers = packed.values, {}
        for number, block in enumerate(self.blocks, 1):
            x = block(x, src_key_padding_mask=packed.padding)
            if return_layers and number in self.return_layers:
                normalized = torch.where(packed.valid[..., None], self.norm(x), 0)
                layers[number] = restore_aliases(normalized)
        x = torch.where(packed.valid[..., None], self.norm(x), 0)
        return EncoderOutput(restore_aliases(x), x, packed.padding, packed.valid, layers)
