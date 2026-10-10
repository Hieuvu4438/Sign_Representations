"""Losses (proposal11 §7.5, §8.5, §9.3) and the S-JEPA centred-CE baseline objective (B11)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

D = 768


def per_token_l1(pred, target, n_levels, d=D):
    """e_i^(l) = ||zhat - sg(z)||_1 / d per level -> [B, M, n_levels]."""
    B, M, _ = pred.shape
    return (pred.float() - target.detach()).abs().view(B, M, n_levels, d).mean(-1)


def region_masked_loss(err, groups, weight, balanced=True, n_groups=5, eps=1e-6):
    """§7.5 L_M. err [B,M,L], groups [B,M] (ownership id), weight [B,M] (0 for padded/invalid targets).
    balanced: mean within each present group (pooled over the batch) then equal average over present groups;
    uniform: plain weighted mean over all targets. Returns (loss, per-group coverage dict)."""
    e = err.mean(-1)  # equal layer averaging
    if not balanced:
        return (e * weight).sum() / (weight.sum() + eps), {}
    losses, cov = [], {}
    for g in range(n_groups):
        w = weight * (groups == g)
        n = w.sum()
        cov[g] = float(n.detach())
        if n > 0:
            losses.append((e * w).sum() / (n + eps))
    if not losses:  # no valid target anywhere: zero loss that keeps the graph (finite, no NaN)
        return (e * 0.0).sum(), cov
    return torch.stack(losses).mean(), cov


def context_loss(err_ctx, weight_ctx, eps=1e-6):
    e = err_ctx.mean(-1)
    return (e * weight_ctx).sum() / (weight_ctx.sum() + eps)


class CenteredCE:
    """S-JEPA objective: target centring (EMA rate beta) + sharpening; CE between softmax(pred/tau_p) and
    softmax((target - c)/tau_t) over the embedding dimension (Abdelfattah & Alahi, ECCV 2024, §3, §4.2)."""

    def __init__(self, dim, beta=0.9, tau_p=0.1, tau_t=0.06, device="cuda"):
        self.c = torch.zeros(dim, device=device); self.beta, self.tau_p, self.tau_t = beta, tau_p, tau_t

    def __call__(self, pred, target, weight, eps=1e-6):
        t = target.detach().float()
        p2 = F.softmax((t - self.c) / self.tau_t, -1)
        lp1 = F.log_softmax(pred.float() / self.tau_p, -1)
        ce = -(p2 * lp1).sum(-1)
        with torch.no_grad():
            w = weight[..., None]
            batch_c = (t * w).reshape(-1, t.shape[-1]).sum(0) / (w.sum() + eps)
            self.c.mul_(self.beta).add_(batch_c, alpha=1 - self.beta)
        return (ce * weight).sum() / (weight.sum() + eps)


def masked_smooth_l1(pred, target, mask, eps=1e-6):
    l = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
    return (l * mask).sum() / (mask.sum() + eps)
