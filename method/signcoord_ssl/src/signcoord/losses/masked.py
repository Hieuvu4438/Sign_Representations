import torch
import torch.nn.functional as F

from .distributed_reduce import weighted_mean


def masked_huber(pred, target, reliability, delta=1.0):
    if pred.shape != target.shape or reliability.shape != pred.shape[:-1]:
        raise ValueError("masked Huber shape mismatch")
    if not torch.isfinite(pred).all():
        raise ValueError("nonfinite predictions")
    valid = (reliability > 0) & torch.isfinite(target).all(-1) & torch.isfinite(reliability)
    target = torch.where(valid[..., None], target, 0)
    # Select finite pred-target residual before any nonlinear loss.
    residual = torch.where(valid[..., None], pred-target, 0)
    error = F.huber_loss(residual, torch.zeros_like(residual), delta=delta, reduction="none").mean(-1)
    weights = torch.where(valid, reliability, 0).detach()
    return weighted_mean((weights*error).sum(), weights.sum())
