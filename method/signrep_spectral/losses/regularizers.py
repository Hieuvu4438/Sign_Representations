"""Explicit variance / covariance regularizers on `features` (version var_cov_v1, spec 8.1).

Normalization is defined by this implementation; it is not the original SignRep loss.
"""
from __future__ import annotations

import torch

REGULARIZER_VERSION = "var_cov_v1"


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    if z.shape[0] < 2:
        raise ValueError("variance regularizer needs batch >= 2")
    std = torch.sqrt(z.var(dim=0, unbiased=True) + eps)
    return torch.clamp(gamma - std, min=0).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        raise ValueError("covariance regularizer needs batch >= 2")
    zc = z - z.mean(0, keepdim=True)
    c = zc.T @ zc / (z.shape[0] - 1)
    off = c - torch.diag(torch.diag(c))
    return off.pow(2).sum() / z.shape[1]
