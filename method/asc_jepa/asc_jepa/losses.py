"""Losses with explicit reductions (spec §10)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def zero_like(x):
    return x.sum() * 0.0


def generic_jepa(pred, target, visible, kappa):
    """pred/target [B,N,C] (target LN-normalised, detached), visible [B,N] bool.
    L = mean_{hidden} mean_D |.|  +  kappa * mean_{visible} mean_D |.|  (each normalised separately)."""
    e = (pred.float() - target).abs().mean(-1)
    hid, vis = (~visible).float(), visible.float()
    lm = (e * hid).sum() / hid.sum().clamp(min=1)
    lc = (e * vis).sum() / vis.sum().clamp(min=1)
    return lm + kappa * lc, {"mask": lm.item(), "ctx": lc.item()}


def balanced_part_mean(err, w):
    """err [B,P,K] per-state (already mean over D), w [B,P,K] weights. Mean over valid k within
    (b,p) -> mean over active parts -> mean over active samples (spec §10.1)."""
    den = w.sum(-1)
    per_bp = (err * w).sum(-1) / den.clamp(min=1e-8)
    act_bp = (den > 0).float()
    per_b = (per_bp * act_bp).sum(-1) / act_bp.sum(-1).clamp(min=1)
    act_b = (act_bp.sum(-1) > 0).float()
    n = act_b.sum()
    if n == 0:
        return zero_like(err), 0.0
    return (per_b * act_b).sum() / n, float(act_b.mean())


def state_loss(pred_states, tgt_states, w):
    e = F.smooth_l1_loss(pred_states, tgt_states, reduction="none", beta=1.0).mean(-1)  # B,P,T
    return balanced_part_mean(e, w)


class ChangeRMS:
    """EMA of mean-square teacher differences per (part, lag); updated after optimizer steps."""

    def __init__(self, P, lags, momentum=0.99, floor=0.05):
        self.m, self.floor = momentum, floor
        self.s2 = {(p, l): None for p in range(P) for l in lags}
        self.pending = {k: [0.0, 0] for k in self.s2}

    def scale(self, p, l, batch_ms):
        v = self.s2[(p, l)]
        v = batch_ms if v is None else v
        return max(float(v) ** 0.5, self.floor)

    def accumulate(self, p, l, sum_ms, count):
        self.pending[(p, l)][0] += sum_ms; self.pending[(p, l)][1] += count

    def step(self):
        for k, (s, c) in self.pending.items():
            if c > 0:
                v = s / c
                self.s2[k] = v if self.s2[k] is None else self.m * self.s2[k] + (1 - self.m) * v
            self.pending[k] = [0.0, 0]


def change_loss(pred_states, tgt_states, valid, gate, lags, rms: ChangeRMS):
    """pred/tgt [B,P,T,C], valid/gate [B,P,T] float. Pairs (t, t+lag); weight = valid_t*valid_j*gate_t*gate_j.
    Teacher difference stats use validity only (not the gate) — spec §10.4."""
    B, P, T, C = tgt_states.shape
    losses = []
    for lag, _h in lags:
        dt = tgt_states[:, :, lag:] - tgt_states[:, :, :-lag]
        dp = pred_states[:, :, lag:] - pred_states[:, :, :-lag]
        pv = valid[:, :, lag:] * valid[:, :, :-lag]
        w = pv * gate[:, :, lag:] * gate[:, :, :-lag]
        errs = torch.zeros(B, P, T - lag, device=tgt_states.device)
        for p in range(P):
            msq = (dt[:, p] ** 2).mean(-1)  # B,T-lag
            cnt = pv[:, p].sum().item()
            bms = float((msq * pv[:, p]).sum().item() / max(cnt, 1))
            s = rms.scale(p, lag, bms)
            rms.accumulate(p, lag, float((msq * pv[:, p]).sum().item()), int(cnt))
            errs[:, p] = F.smooth_l1_loss(dp[:, p] / s, dt[:, p] / s, reduction="none", beta=1.0).mean(-1)
        l, _ = balanced_part_mean(errs, w)
        if w.sum() > 0:
            losses.append(l)
    if not losses:
        return zero_like(pred_states), 0
    return torch.stack(losses).mean(), len(losses)


def var_cov(z, eps=1e-4):
    z = z.float()
    if z.shape[0] < 2:
        return zero_like(z)
    std = torch.sqrt(z.var(0, unbiased=False) + eps)
    lv = F.relu(1 - std).mean()
    zc = z - z.mean(0)
    C = zc.T @ zc / (z.shape[0] - 1)
    off = C - torch.diag(torch.diag(C))
    lc = (off ** 2).sum() / z.shape[1]
    return lv + 0.04 * lc


def geometry_loss(pred, target, mask):
    """pred/target [B,F,G], mask [B,F,G] -> Smooth-L1 mean over valid dims, mean over valid samples."""
    e = F.smooth_l1_loss(pred.float(), target, reduction="none", beta=1.0) * mask
    per = e.sum((1, 2)) / mask.sum((1, 2)).clamp(min=1)
    act = (mask.sum((1, 2)) > 0).float()
    return (per * act).sum() / act.sum().clamp(min=1)
