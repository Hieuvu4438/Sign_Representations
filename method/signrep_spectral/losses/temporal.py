"""Velocity / acceleration controls (per-frame differences in native cadence)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .stats import LossStat


def difference(a: torch.Tensor, valid: torch.Tensor, order: int):
    if order == 1:
        return a[:, 1:] - a[:, :-1], valid[:, 1:] & valid[:, :-1]
    if order == 2:
        return a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2], valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    raise ValueError(order)


def difference_term_stats(pred, target, valid, order: int, name: str) -> LossStat:
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Non-finite prediction")
    target = target.detach()
    if (valid & ~torch.isfinite(target)).any():
        raise ValueError("Declared-valid target is non-finite")
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    dp, v = difference(pred, valid, order)
    dt, _ = difference(safe_target, valid, order)
    count = int(v.sum())
    if count == 0:
        return LossStat(name, pred.sum() * 0.0, 0)
    err = (dp - dt).abs()
    return LossStat(name, torch.where(v, err, torch.zeros_like(err)).sum(), count)


def difference_loss_stats(
    groups: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], order: int
) -> List[LossStat]:
    kind = {1: "velocity", 2: "acceleration"}[order]
    device_type = next(iter(groups.values()))[0].device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return [
            difference_term_stats(p.float(), t.float(), v, order, name=f"{kind}/{g}")
            for g, (p, t, v) in groups.items()
        ]
