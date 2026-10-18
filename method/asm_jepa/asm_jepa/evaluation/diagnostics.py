"""Representation diagnostics (§12.3): embedding std, effective rank, mean pairwise cosine."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def embedding_diagnostics(x, eps: float = 1e-12) -> dict:
    x = torch.as_tensor(np.asarray(x, np.float32)).double()
    if x.shape[0] < 2:
        return {"n": int(x.shape[0])}
    xc = x - x.mean(0)
    sv = torch.linalg.svdvals(xc)
    p = (sv**2) / (sv**2).sum().clamp_min(eps)
    p = p.clamp_min(eps)
    erank = float(torch.exp(-(p * p.log()).sum()))
    xn = torch.nn.functional.normalize(x, dim=-1)
    cos = xn @ xn.T
    n = x.shape[0]
    mean_cos = float((cos.sum() - cos.diagonal().sum()) / (n * (n - 1)))
    return {"n": int(n), "std_mean": float(x.std(0).mean()), "effective_rank": erank, "mean_pairwise_cos": mean_cos}
