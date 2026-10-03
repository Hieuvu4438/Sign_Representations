"""Capacity-matched frozen readouts of positively recorded functional intervals.

Boundary distributions rank candidate positions; they are not probabilities of
phenomenon presence or verified absence. Gold classes/intervals never enter the
forward signature. These standard readouts carry no method-novelty claim.
"""
import numpy as np
import torch
from torch import nn


class IntervalReadout(nn.Module):
    def __init__(self, dimension=768, classes=3, temporal=False):
        super().__init__()
        self.classes = classes
        self.temporal = temporal
        self.classifier = nn.Linear(dimension, classes)
        # Omit boundary biases on BOTH controls: temporal softmax would cancel
        # them. Every counted boundary parameter affects the respective output.
        self.boundaries = nn.Linear(dimension, 2 * classes, bias=False)

    def forward(self, tokens, mask, positions):
        if not mask.any(dim=1).all():
            raise ValueError('Cannot predict a sample with no supported windows')
        masked = tokens * mask.unsqueeze(-1)
        pooled = masked.sum(1) / mask.sum(1, keepdim=True)
        logits = self.classifier(pooled)
        if self.temporal:
            scores = self.boundaries(masked).masked_fill(~mask.unsqueeze(-1), -torch.inf)
            distribution = scores.softmax(dim=1)
            endpoints = (distribution * positions.unsqueeze(-1)).sum(1)
        else:
            endpoints = self.boundaries(pooled).sigmoid()
            distribution = None
        ordered, order = endpoints.reshape(-1, self.classes, 2).sort(dim=-1)
        if distribution is not None:
            shape = distribution.shape
            distribution = distribution.reshape(shape[0], shape[1], self.classes, 2).gather(
                -1, order.unsqueeze(1).expand(-1, shape[1], -1, -1)).reshape(shape)
        return logits, ordered, distribution


def interval_iou(predicted, target):
    predicted, target = np.asarray(predicted, float), np.asarray(target, float)
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError('Interval arrays must have equal [N,2] shapes')
    if (not np.isfinite(predicted).all() or not np.isfinite(target).all()
            or (predicted[:, 1] < predicted[:, 0]).any()
            or (target[:, 1] <= target[:, 0]).any()):
        raise ValueError('Invalid or reversed intervals')
    intersection = np.maximum(0, np.minimum(predicted[:, 1], target[:, 1])
                              - np.maximum(predicted[:, 0], target[:, 0]))
    union = predicted[:, 1] - predicted[:, 0] + target[:, 1] - target[:, 0] - intersection
    return intersection / union


def macro_values(gold, values, weights=None):
    """Mean over seeds and classes with weighted support; no best-seed choice."""
    gold, values = np.asarray(gold), np.asarray(values, float)
    if values.ndim == 1:
        values = values[None, :]
    weights = np.ones(len(gold)) if weights is None else np.asarray(weights, float)
    supported = [c for c in np.unique(gold) if weights[gold == c].sum() > 0]
    if not supported:
        raise ValueError('No supported classes')
    per_seed = np.mean([np.average(values[:, gold == c], axis=1, weights=weights[gold == c])
                        for c in supported], axis=0)
    return per_seed


def cluster_interval_bootstrap(gold, groups, values, reference=None, resamples=2000, seed=42):
    gold, values = np.asarray(gold), np.asarray(values, float)
    if values.ndim != 2 or values.shape[1] != len(gold) or len(groups) != len(gold):
        raise ValueError('Seed/row dimensions disagree')
    if resamples < 2000 or not np.isfinite(values).all():
        raise ValueError('Invalid bootstrap inputs')
    _, inverse = np.unique(groups, return_inverse=True)
    n_groups = len(np.unique(inverse))
    if n_groups < 2:
        raise ValueError('Cannot estimate uncertainty from fewer than two groups')
    if reference is not None:
        reference = np.asarray(reference, float)
        if reference.shape != values.shape or not np.isfinite(reference).all():
            raise ValueError('Paired reference shape or finiteness differs')
    point = macro_values(gold, values)
    delta = None if reference is None else point - macro_values(gold, reference)
    rng, samples, deltas, missing = np.random.default_rng(seed), [], [], 0
    classes = np.unique(gold)
    for _ in range(resamples):
        weights = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)[inverse]
        missing += any(weights[gold == c].sum() == 0 for c in classes)
        value = macro_values(gold, values, weights)
        samples.append(float(value.mean()))
        if reference is not None:
            deltas.append(float((value - macro_values(gold, reference, weights)).mean()))
    result = {'point': float(point.mean()), 'per_seed': point.tolist(),
              'training_seed_sd': float(point.std(ddof=1)) if len(point) > 1 else None,
              'ci95': np.quantile(samples, [.025, .975]).tolist(),
              'samples': len(gold), 'groups': n_groups, 'resamples': resamples,
              'resamples_missing_class': int(missing),
              'interval_scope': 'Conditional on trained heads and observed source-sequence groups; one test signer. Not signer-population or retraining uncertainty.'}
    if delta is not None:
        result.update(delta=float(delta.mean()), delta_ci95=np.quantile(deltas, [.025, .975]).tolist(),
                      reference_point=float(macro_values(gold, reference).mean()))
    return result
