"""Losses (proposal12 v2.1 §6.7–§6.8).

ctx targets   : S-JEPA centred cross-entropy over the feature dimension (Abdelfattah & Alahi, ECCV'24: centre EMA beta 0.9,
                tau_t 0.06 (warm-up 0.04 -> 0.06), tau_s 0.1); same formulation as method/obs_jepa/oj/losses.py::CenteredCE.
proto (A6)    : iBOT/DINO-style prototype CE with centring.
l1sig (A6)    : L1 on LayerNorm'd targets + SIGReg (LeJEPA, Balestriero & LeCun 2025; Epps-Pulley statistic, adapted
                from the authors' MINIMAL example: t in [0, 3], 17 knots, doubled trapezoid weights, random slices).
int targets   : masked smooth-L1 (beta 0.5) averaged per dimension within each channel type, then over weighted tokens.
global (G1)   : centred CE between teacher / student CLS projections + KoLeo per view (DINOv2).
Loss balancing: each term divided by its running mean, frozen after warm-up (v2.1: avoids the positive feedback of
                the teacher entropy contained in the ctx running mean).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class CenteredCE:
    def __init__(self, dim, beta=0.9, tau_s=0.1, device="cuda"):
        self.c = torch.zeros(dim, device=device)
        self.beta, self.tau_s = beta, tau_s

    def target(self, t, tau_t):
        return F.softmax((t.float() - self.c) / tau_t, -1)

    def __call__(self, pred, t, w, tau_t, update=True, eps=1e-6):
        """pred / t [B,N,D]; w [B,N] (>= 0). returns weighted mean CE and the per-token CE."""
        p_t = self.target(t.detach(), tau_t)
        ce = -(p_t * F.log_softmax(pred.float() / self.tau_s, -1)).sum(-1)
        if update:
            with torch.no_grad():
                m = (w > 0).float()[..., None]
                if m.sum() > 0:
                    bc = (t.float() * m).reshape(-1, t.shape[-1]).sum(0) / m.sum()
                    self.c.mul_(self.beta).add_(bc, alpha=1 - self.beta)
        return (ce * w).sum() / (w.sum() + eps), ce

    @staticmethod
    def entropy(t_prob):
        return -(t_prob * (t_prob + 1e-12).log()).sum(-1)


def l1_layernorm(pred, t, w, eps=1e-6):
    t = F.layer_norm(t.float(), (t.shape[-1],))
    e = (pred.float() - t.detach()).abs().mean(-1)
    return (e * w).sum() / (w.sum() + eps)


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17, slices=1024):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        self.slices = slices

    def forward(self, proj):
        """proj [N, D] (rows = samples)."""
        proj = proj.float()
        A = torch.randn(proj.size(-1), self.slices, device=proj.device)
        A = A / A.norm(p=2, dim=0)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()


def intrinsic_loss(preds, targets, dim_w, tok_w, eps=1e-6):
    """lists over channels: preds/targets [B,Tb,Dt], dim_w [B,Tb,Dt] (0 = undefined dim), tok_w [B,Tb].
    returns weighted mean over tokens of (per-dimension-averaged smooth-L1 within the channel)."""
    num, den = 0.0, 0.0
    for p, y, dw, tw in zip(preds, targets, dim_w, tok_w):
        l = F.smooth_l1_loss(p.float(), y.detach().float(), reduction="none", beta=0.5)
        per_tok = (l * dw).sum(-1) / dw.sum(-1).clamp(min=1)
        tw = tw * (dw.sum(-1) > 0)
        num = num + (per_tok * tw).sum(); den = den + tw.sum()
    return num / (den + eps)


def koleo(x, eps=1e-8):
    """KoLeo (Sablayrolles et al.; DINOv2) on one view: -mean log nearest-neighbour distance of L2-normalised x [B,D]."""
    x = F.normalize(x.float(), dim=-1)
    with torch.no_grad():
        d = x @ x.T
        d.fill_diagonal_(-2)
        nn_idx = d.argmax(1)
    dist = (x - x[nn_idx]).norm(dim=-1)
    return -(dist + eps).log().mean()


class LossBalancer:
    """divide each named term by its running mean (EMA 0.99); the means freeze after `freeze_step`."""

    def __init__(self, freeze_step, decay=0.99):
        self.m, self.freeze, self.decay, self.step = {}, freeze_step, decay, 0

    def __call__(self, name, value):
        v = float(value.detach())
        if name not in self.m:
            self.m[name] = v
        elif self.step < self.freeze:
            self.m[name] = self.decay * self.m[name] + (1 - self.decay) * v
        return value / max(self.m[name], 1e-6)

    def tick(self):
        self.step += 1

    def state_dict(self):
        return {"m": self.m, "step": self.step}

    def load_state_dict(self, s):
        self.m, self.step = s["m"], s["step"]
