"""Level-2 log-signature without external dependencies (§8.4).

For increments delta_i and prefix a_i = sum_{j<i} delta_j:
  S1 = sum_i delta_i
  S2 = sum_i a_i (x) delta_i + 1/2 delta_i (x) delta_i        (tensor basis)
  logsig2 = [S1, upper triangle (i<j) of A],  A = 1/2 (S2 - S2^T)  (Levy area)
Basis/order: first the d level-1 coefficients, then area_{ij} for i<j in row-major order.
`inc_mask` zeroes increments outside the path support (zero increments leave the signature unchanged).
"""
from __future__ import annotations

import numpy as np


def logsig2_dim(d: int) -> int:
    return d + d * (d - 1) // 2


def logsig2_np(path: np.ndarray, inc_mask: np.ndarray | None = None) -> np.ndarray:
    """path [..., N, d] -> [..., d + d(d-1)/2]"""
    path = np.asarray(path, np.float64)
    delta = np.diff(path, axis=-2)
    if inc_mask is not None:
        delta = delta * np.asarray(inc_mask, np.float64)[..., None]
    s1 = delta.sum(-2)
    prefix = np.cumsum(delta, axis=-2) - delta
    s2 = np.einsum("...ni,...nj->...ij", prefix, delta) + 0.5 * np.einsum("...ni,...nj->...ij", delta, delta)
    area = 0.5 * (s2 - np.swapaxes(s2, -1, -2))
    d = path.shape[-1]
    iu, ju = np.triu_indices(d, k=1)
    return np.concatenate([s1, area[..., iu, ju]], axis=-1)


def logsig2_torch(path, inc_mask=None):
    import torch

    with torch.no_grad():
        delta = path[..., 1:, :] - path[..., :-1, :]
        if inc_mask is not None:
            delta = delta * inc_mask[..., None].to(delta.dtype)
        s1 = delta.sum(-2)
        prefix = delta.cumsum(-2) - delta
        s2 = torch.einsum("...ni,...nj->...ij", prefix, delta) + 0.5 * torch.einsum("...ni,...nj->...ij", delta, delta)
        area = 0.5 * (s2 - s2.transpose(-1, -2))
        d = path.shape[-1]
        iu, ju = torch.triu_indices(d, d, offset=1)
        return torch.cat([s1, area[..., iu, ju]], dim=-1)
