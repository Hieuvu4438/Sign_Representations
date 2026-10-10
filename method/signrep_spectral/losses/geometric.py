"""Framewise masked SmoothL1 geometric reconstruction (L_base_v1)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .stats import LossStat


def geometric_term_stats(pred, target, observed, beta: float = 1.0, name: str = "geom") -> LossStat:
    if pred.shape != target.shape or pred.shape != observed.shape:
        raise ValueError("geometric shape mismatch")
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Non-finite prediction")
    target = target.detach()
    if (observed & ~torch.isfinite(target)).any():
        raise ValueError("Declared-observed target is non-finite")
    safe_target = torch.where(observed, target, torch.zeros_like(target))
    count = int(observed.sum())
    if count == 0:
        return LossStat(name, pred.sum() * 0.0, 0)
    elem = F.smooth_l1_loss(pred, safe_target, beta=beta, reduction="none")
    return LossStat(name, torch.where(observed, elem, torch.zeros_like(elem)).sum(), count)


def geometric_loss_stats(
    heads: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    beta: float,
    weights: List[float],
) -> List[LossStat]:
    out = []
    device_type = next(iter(heads.values()))[0].device.type
    with torch.autocast(device_type=device_type, enabled=False):
        for (hname, (pred, target, obs)), w in zip(heads.items(), weights):
            st = geometric_term_stats(pred.float(), target.float(), obs, beta, name=f"geom/{hname}")
            st.coefficient = float(w)
            out.append(st)
    return out
