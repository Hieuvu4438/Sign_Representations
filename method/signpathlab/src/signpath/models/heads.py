"""Downstream heads have trainable objectives, distinct from SSL prediction heads."""
import torch
from torch import nn


class LinearHead(nn.Module):
    def __init__(self, dimension, classes):
        super().__init__()
        if classes <= 0:
            raise ValueError("Classification head needs a nonempty taxonomy")
        self.classifier = nn.Linear(dimension, classes)

    def forward(self, clip_embeddings):
        return self.classifier(clip_embeddings)


class AttentiveHead(nn.Module):
    def __init__(self, dimension, classes, heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dimension) * .02)
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.classifier = nn.Linear(dimension, classes)

    def forward(self, dense, valid):
        if not valid.any(dim=1).all():
            raise ValueError("Attentive probe requires at least one valid frame per clip")
        memory = torch.where(valid[..., None], dense, 0)
        pooled, _ = self.attention(self.query.expand(len(dense), -1, -1), memory, memory,
                                   key_padding_mask=~valid, need_weights=False)
        return self.classifier(pooled[:, 0])
