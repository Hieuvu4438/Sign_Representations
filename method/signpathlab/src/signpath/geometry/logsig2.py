"""Ordered [level1, upper Lie-level2]; no scalar level0."""
import numpy as np


def logsig2_reference(points):
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("Need >=2 finite observations [T,D]")
    d = x.shape[1]
    a, b = np.zeros(d), np.zeros((d, d))
    for delta in np.diff(x, axis=0):
        b += np.outer(a, delta) + 0.5 * np.outer(delta, delta)
        a += delta
    lie2 = b - 0.5 * np.outer(a, a)
    return np.concatenate([a, lie2[np.triu_indices(d, 1)]])


def logsig2(points):
    """Differentiable batched Torch backend [...,T,D], accumulated outside AMP."""
    import torch
    if points.ndim < 2 or points.shape[-2] < 2 or points.shape[-1] < 1 or not torch.isfinite(points).all():
        raise ValueError("Need >=2 finite observations [...,T,D]")
    if not points.is_floating_point():
        raise ValueError("Torch paths must have a floating dtype")
    with torch.autocast(device_type=points.device.type, enabled=False):
        x = points if points.dtype == torch.float64 else points.float()
        increments = x.diff(dim=-2)
        prefix = increments.cumsum(dim=-2) - increments
        i, j = torch.triu_indices(x.shape[-1], x.shape[-1], offset=1, device=x.device)
        area = 0.5 * (prefix[..., i] * increments[..., j] - prefix[..., j] * increments[..., i]).sum(dim=-2)
        return torch.cat([increments.sum(dim=-2), area], dim=-1)
