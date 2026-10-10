"""Masked multiscale log-magnitude spectral loss on temporal trajectories.

Conventions (spec section 7): clip-observed-mean centering inside the spectral
branch only, strict window-channel validity (no padding / interpolation),
periodic Hann window, rfft(norm="backward") divided by window.sum(), one-sided
bins with equal weight, L1 between log1p magnitudes, fixed denominator over
configured (group, scale) terms. Native PyTorch only (torch.fft / Tensor.unfold).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from .stats import LossStat


def _validate(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
    if pred.dim() != 3:
        raise ValueError(f"pred must be [B,T,D], got {tuple(pred.shape)}")
    if pred.shape != target.shape or pred.shape != valid.shape:
        raise ValueError(
            f"shape mismatch pred={tuple(pred.shape)} target={tuple(target.shape)} "
            f"valid={tuple(valid.shape)}"
        )
    if valid.dtype != torch.bool:
        raise TypeError("valid must be bool")
    if not pred.is_floating_point() or pred.dtype != target.dtype:
        raise TypeError(f"pred/target dtype mismatch {pred.dtype} vs {target.dtype}")
    if pred.device != target.device or pred.device != valid.device:
        raise ValueError("pred/target/valid on different devices")
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Non-finite prediction")
    if (valid & ~torch.isfinite(target)).any():
        raise ValueError("Declared-valid target is non-finite")


def center_clip(pred, target, valid):
    """Center each channel by its mean over target-valid times (same mask for both)."""
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    counts = valid.sum(dim=1, keepdim=True).clamp_min(1).to(pred.dtype)
    target_mean = safe_target.sum(dim=1, keepdim=True) / counts
    pred_mean = torch.where(valid, pred, torch.zeros_like(pred)).sum(dim=1, keepdim=True) / counts
    return pred - pred_mean, safe_target - target_mean


def windowed_rows(pred_c, target_c, valid, n: int, hop: int):
    """Strictly valid windows as rows [U,N] (Tensor.unfold appends N last: [B,M,D,N])."""
    if pred_c.shape[1] < n:
        raise ValueError("No implicit padding or silent scale dropping")
    row_valid = valid.unfold(1, n, hop).all(dim=-1)  # [B,M,D]
    pred_rows = pred_c.unfold(1, n, hop)[row_valid]
    target_rows = target_c.unfold(1, n, hop)[row_valid]
    return pred_rows, target_rows, row_valid


def hann_periodic(n: int, device, dtype) -> torch.Tensor:
    return torch.hann_window(n, periodic=True, device=device, dtype=dtype)


def spectrum(rows: torch.Tensor, n: int) -> torch.Tensor:
    window = hann_periodic(n, rows.device, rows.dtype)
    return torch.fft.rfft(rows * window, n=n, dim=-1, norm="backward") / window.sum()


def spectral_term_stats(pred, target, valid, n: int, hop: int, name: str = "spec") -> LossStat:
    _validate(pred, target, valid)
    target = target.detach()
    valid = valid.detach()
    pred_c, target_c = center_clip(pred, target, valid)
    pred_rows, target_rows, _ = windowed_rows(pred_c, target_c, valid, n, hop)
    if pred_rows.shape[0] == 0:
        return LossStat(name, pred.sum() * 0.0, 0)
    err = (torch.log1p(spectrum(pred_rows, n).abs()) - torch.log1p(spectrum(target_rows, n).abs())).abs()
    return LossStat(name, err.sum(), int(err.numel()))


def window_time_term_stats(pred, target, valid, n: int, hop: int, name: str = "wtime") -> LossStat:
    """Control B4: same centering/windows/Hann/window-sum norm, L1 in time domain."""
    _validate(pred, target, valid)
    target = target.detach()
    valid = valid.detach()
    pred_c, target_c = center_clip(pred, target, valid)
    pred_rows, target_rows, _ = windowed_rows(pred_c, target_c, valid, n, hop)
    if pred_rows.shape[0] == 0:
        return LossStat(name, pred.sum() * 0.0, 0)
    window = hann_periodic(n, pred.device, pred.dtype)
    err = ((pred_rows - target_rows) * window / window.sum()).abs()
    return LossStat(name, err.sum(), int(err.numel()))


def grouped_term_stats(
    groups: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    windows: Sequence[Dict[str, int]],
    kind: str = "spectral",
) -> List[LossStat]:
    """One LossStat per configured (group, window scale); computed in FP32 w/o autocast."""
    fn = {"spectral": spectral_term_stats, "window_time": window_time_term_stats}[kind]
    out: List[LossStat] = []
    device_type = next(iter(groups.values()))[0].device.type
    with torch.autocast(device_type=device_type, enabled=False):
        for gname, (pred, target, valid) in groups.items():
            p = pred if pred.dtype == torch.float64 else pred.float()
            t = target.to(p.dtype)
            for w in windows:
                out.append(fn(p, t, valid, w["n"], w["hop"], name=f"{kind}/{gname}/N{w['n']}"))
    return out
