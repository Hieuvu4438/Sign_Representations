"""Supplement A4 style matching network, safe pair construction and EMA state."""

from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm


def gram_style(tokens, normalization="tokens_channels"):    
    if tokens.ndim != 3 or min(tokens.shape[1:]) == 0:
        raise ValueError("Expected B×K×D final encoder tokens")
    tokens = tokens.float()
    gram = tokens.transpose(1, 2) @ tokens
    if normalization == "tokens_channels":
        gram = gram / (tokens.shape[1] * tokens.shape[2])
    elif normalization == "tokens":
        gram = gram / tokens.shape[1]
    elif normalization != "none":
        raise ValueError("Unknown Gram normalization")
    return gram.mean(dim=-1)


def style_pairs(video_ids, generator=None, device=None):
    """Return per-anchor other-view and random different-video indices."""
    ids = list(video_ids)
    if len(set(ids)) < 2:
        raise ValueError("Unmatched styles require at least two different videos")
    matched, unmatched = [], []
    for anchor, video in enumerate(ids):
        positive = [i for i, value in enumerate(ids) if value == video and i != anchor]
        if len(positive) != 1:
            raise ValueError("Batch must contain exactly two views of every unique video")
        negatives = [i for i, value in enumerate(ids) if value != video]
        picked = torch.randint(len(negatives), (1,), generator=generator).item()
        matched.append(positive[0])
        unmatched.append(negatives[picked])
    return (torch.tensor(matched, device=device), torch.tensor(unmatched, device=device))


class StyleDiscriminator(nn.Module):
    def __init__(self, dimension=768):
        super().__init__()
        self.style = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(),
                                   nn.Linear(dimension, dimension), nn.GELU(), nn.LayerNorm(dimension))
        layers = []
        for index, (inputs, outputs) in enumerate([(2 * dimension, dimension), (dimension, dimension),
                                                  (dimension, dimension), (dimension, 1)]):
            layers.append(spectral_norm(nn.Linear(inputs, outputs)))
            if index < 3:
                layers.append(nn.GELU())
        self.match = nn.Sequential(*layers)

    def forward(self, content, style):
        return self.match(torch.cat([content, self.style(style * 100)], dim=-1)).squeeze(-1)

    def paired_logits(self, content, style, matched, unmatched):
        # One forward for both labels keeps spectral normalization identical.
        size = len(content)
        logits = self(torch.cat([content, content]), torch.cat([style[matched], style[unmatched]]))
        return logits[:size], logits[size:]


class MatchEMA(nn.Module):
    def __init__(self, momentum=.1):
        super().__init__()
        if not 0 < momentum <= 1:
            raise ValueError("Invalid EMA update momentum")
        self.momentum = momentum
        self.register_buffer("matched", torch.tensor(0.))
        self.register_buffer("unmatched", torch.tensor(0.))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, matched, unmatched):
        for target, scores in [(self.matched, matched), (self.unmatched, unmatched)]:
            if not torch.isfinite(scores).all():
                raise ValueError("Cannot update EMA from nonfinite probabilities")
            target.lerp_(scores.detach().float().mean(), self.momentum)
        self.updates.add_(1)


def discriminator_loss(matched_logits, unmatched_logits):
    return F.binary_cross_entropy_with_logits(
        torch.cat([matched_logits, unmatched_logits]),
        torch.cat([torch.ones_like(matched_logits), torch.zeros_like(unmatched_logits)]))


@contextmanager
def frozen_discriminator(model):
    parameters = list(model.parameters())
    requires = [parameter.requires_grad for parameter in parameters]
    training = model.training
    try:
        model.eval()
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        model.train(training)
        for parameter, flag in zip(parameters, requires):
            parameter.requires_grad_(flag)
