"""RGB token masks for proposal A (proposal11 §7.4, §13.2).

Token grid: T' = T/2 tubelet bins x g x g patches (patch 16). Every sampler returns disjoint index sets over the
valid tokens: C (context), M (targets, with ownership group and kind) and G (hidden guards without loss), with
|M| + |G| = round(rho * N_v) exactly (rho = 0.8). Regions come from TRAIN-ONLY landmarks and are never given to
the encoder; ownership is exclusive with priority hands > face > body > other.

Policies
  generic      multi-block (V-JEPA-like spatial blocks over the full temporal extent) until the budget is met
  partial      one region event (50% partial spatial block over 1-3 bins covering 25-50% of the region,
               50% whole-region temporal gap of 1-3 (hands) / 1-2 (face, body) bins with same-region anchors),
               1-patch spatial / 1-bin temporal guards when the budget allows, then generic fill;
               reliable hands keep >= 25% visible support over the window.
  whole        whole-articulator event: one region hidden over every bin (A14 control), then generic fill
  jitter       partial policy on regions translated by a random offset (A15 metadata-specificity control)
  generic_cov  generic blocks but with the same >= 25% hand-support protection (coverage-matched generic, §13.2)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from oj.data.landmarks import PARTS

GROUPS = ("body", "face", "lh", "rh", "other")
G_BODY, G_FACE, G_LH, G_RH, G_OTHER = range(5)
_PRIORITY = (("lh", G_LH), ("rh", G_RH), ("face", G_FACE), ("body", G_BODY))  # assignment order (hands first)


@dataclass
class MaskSample:
    ctx: np.ndarray
    tgt: np.ndarray
    tgt_group: np.ndarray
    tgt_structured: np.ndarray
    guard: np.ndarray
    stats: dict = field(default_factory=dict)


def token_ownership(xy, observed, Tb, g, patch=16, min_pts=3, margin_frac=0.15, offset=(0.0, 0.0)):
    """xy [2*Tb, J, 2] letterboxed pixels, observed [2*Tb, J] -> owner [Tb, g, g] (group id) and reliable [Tb, 4]."""
    owner = np.full((Tb, g, g), G_OTHER, np.int64)
    reliable = np.zeros((Tb, 4), bool)
    c = (np.arange(g) + 0.5) * patch
    for b in range(Tb):
        fr = slice(2 * b, 2 * b + 2)
        for name, gid in reversed(_PRIORITY):  # paint low priority first, hands last
            idx = PARTS[name]
            pts = xy[fr][:, idx][observed[fr][:, idx]]
            if len(pts) < min_pts:
                continue
            pts = pts + np.asarray(offset, np.float32)
            x0, y0 = pts.min(0); x1, y1 = pts.max(0)
            m = max(margin_frac * max(x1 - x0, y1 - y0), patch / 2)
            xs = (c >= x0 - m) & (c <= x1 + m); ys = (c >= y0 - m) & (c <= y1 + m)
            sel = ys[:, None] & xs[None, :]
            if sel.any():
                owner[b][sel] = gid; reliable[b, gid] = True
    # after hands-last painting, recompute reliability as "owns >= 1 token"
    for gid in range(4):
        reliable[:, gid] &= (owner == gid).reshape(Tb, -1).any(1)
    return owner, reliable


def _generic_block(rng, Tb, g, scale=0.15, aspect=(0.75, 1.5)):
    area = max(1, int(round(g * g * scale)))
    ar = rng.uniform(*aspect)
    h = int(np.clip(round(np.sqrt(area * ar)), 1, g)); w = int(np.clip(round(np.sqrt(area / ar)), 1, g))
    y = rng.integers(0, g - h + 1); x = rng.integers(0, g - w + 1)
    m = np.zeros((Tb, g, g), bool); m[:, y:y + h, x:x + w] = True
    return m


class RGBMaskSampler:
    def __init__(self, T=16, size=224, patch=16, rho=0.8, policy="partial", seed=0, hand_min_support=0.25):
        assert policy in ("generic", "generic_cov", "partial", "whole", "jitter")
        self.Tb, self.g, self.patch, self.rho, self.policy = T // 2, size // patch, patch, rho, policy
        self.rng = np.random.default_rng(seed)
        self.hand_min_support = hand_min_support

    # ---------------------------------------------------------------- events
    def _event(self, owner, reliable, hidden, target, guard, budget, stats, valid):
        rng, Tb, g = self.rng, self.Tb, self.g
        present = [gid for gid in (G_LH, G_RH, G_FACE, G_BODY) if reliable[:, gid].any()]
        if not present:
            stats["fallback"] = "no_reliable_region"; return
        gid = int(rng.choice(present))
        region = (owner == gid) & valid
        if self.policy == "whole":
            ev = region.copy()
            stats["event"] = f"whole_{GROUPS[gid]}"
        else:
            bins = np.flatnonzero(reliable[:, gid])
            npatch = region.reshape(Tb, -1).sum(1)
            if (npatch[bins] < 4).all() or rng.random() < 0.5:
                # whole-region temporal gap with anchors on both sides when possible
                max_gap = 3 if gid in (G_LH, G_RH) else 2
                L = int(rng.integers(1, max_gap + 1))
                inner = [b for b in bins if b - 1 in bins and b + L in bins] or list(bins)
                b0 = int(rng.choice(inner)); sel = np.zeros(Tb, bool); sel[b0:b0 + L] = True
                ev = region & sel[:, None, None]
                stats["event"] = f"gap_{GROUPS[gid]}_{L}"
                tg = np.zeros(Tb, bool)  # temporal guard bins at the gap boundaries (region tokens only)
                for b in (b0 - 1, b0 + L):
                    if 0 <= b < Tb:
                        tg[b] = True
                gmask = region & tg[:, None, None]
            else:
                L = int(rng.integers(1, 4)); b0 = int(rng.choice(bins)); sel = np.zeros(Tb, bool); sel[b0:b0 + L] = True
                cover = rng.uniform(0.25, 0.50)
                ev = np.zeros_like(region)
                for b in np.flatnonzero(sel):
                    ys, xs = np.nonzero(region[b])
                    if len(ys) < 4:
                        continue
                    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                    want = max(1, int(round(cover * len(ys))))
                    hh = int(np.clip(round(np.sqrt(want * (y1 - y0 + 1) / max(1, x1 - x0 + 1))), 1, y1 - y0 + 1))
                    ww = int(np.clip(int(np.ceil(want / hh)), 1, x1 - x0 + 1))
                    yy = int(rng.integers(y0, y1 - hh + 2)); xx = int(rng.integers(x0, x1 - ww + 2))
                    blk = np.zeros((g, g), bool); blk[yy:yy + hh, xx:xx + ww] = True
                    ev[b] = blk & region[b]
                stats["event"] = f"partial_{GROUPS[gid]}_{L}_{cover:.2f}"
                # 1-patch spatial ring guard in the same bins
                ring = np.zeros_like(ev)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ring |= np.roll(np.roll(ev, dy, 1), dx, 2)
                gmask = ring & ~ev
            ev &= ~hidden
            if gid in (G_LH, G_RH) and self.policy in ("partial", "jitter"):  # keep >= min support of this hand
                need = int(np.ceil(self.hand_min_support * region.sum()))
                if (region & ~hidden & ~ev & ~gmask).sum() < need:
                    gmask &= ~region
                short = need - int((region & ~hidden & ~ev).sum())
                if short > 0:
                    idx = rng.permutation(np.flatnonzero(ev.ravel()))[:short]
                    ev.ravel()[idx] = False
            n_ev = int(ev.sum())
            if n_ev == 0:
                stats["fallback"] = "empty_event"; return
            gmask &= ~hidden & ~ev
            if hidden.sum() + n_ev + gmask.sum() > budget:
                gmask[:] = False  # guard only when the budget allows
            guard |= gmask; hidden |= gmask
        if hidden.sum() + ev.sum() > budget:  # shrink the event to fit the budget
            idx = np.flatnonzero(ev.ravel()); keep = rng.permutation(idx)[: max(0, budget - int(hidden.sum()))]
            ev = np.zeros(ev.size, bool); ev[keep] = True; ev = ev.reshape(hidden.shape)
        target |= ev; hidden |= ev
        stats["structured_targets"] = int(ev.sum()); stats["guard"] = int(guard.sum())

    # ---------------------------------------------------------------- sample
    def sample(self, owner, reliable, valid, owner_weight=None):
        """owner/reliable drive the mask (jittered for A15); owner_weight (true ownership, default = owner) labels
        target groups for the loss. valid [Tb,g,g] token validity -> MaskSample."""
        owner_weight = owner if owner_weight is None else owner_weight
        rng, Tb, g = self.rng, self.Tb, self.g
        Nv = int(valid.sum()); budget = int(round(self.rho * Nv))
        hidden = ~valid  # padding never enters context, never counted in the budget
        hidden = hidden.copy(); pad = ~valid
        target = np.zeros_like(valid); guard = np.zeros_like(valid); stats = {"Nv": Nv, "budget": budget}
        budget_total = budget + int(pad.sum())
        if self.policy not in ("generic", "generic_cov"):
            self._event(owner, reliable, hidden, target, guard, budget_total, stats, valid)
        # hand support constraint: keep >= hand_min_support of each reliable hand's tokens visible
        protect = np.zeros_like(valid)
        if self.policy in ("partial", "jitter", "generic_cov"):
            for gid in (G_LH, G_RH):
                reg = (owner == gid) & valid
                if reg.sum() == 0:
                    continue
                vis_needed = int(np.ceil(self.hand_min_support * reg.sum()))
                vis_now = reg & ~hidden
                if vis_now.sum() > 0:
                    keep = rng.permutation(np.flatnonzero(vis_now.ravel()))[:vis_needed]
                    protect.ravel()[keep] = True
        n_struct = int(target.sum())
        structured = target.copy()
        tries = 0
        while hidden.sum() < budget_total and tries < 400:
            blk = _generic_block(rng, Tb, g) & ~hidden & ~protect & valid
            room = budget_total - int(hidden.sum())
            idx = np.flatnonzero(blk.ravel())
            if len(idx) > room:
                idx = rng.permutation(idx)[:room]
            target.ravel()[idx] = True; hidden.ravel()[idx] = True; tries += 1
        if hidden.sum() < budget_total:  # random fill (rare): respect protection first
            free = np.flatnonzero((~hidden & ~protect & valid).ravel())
            need = budget_total - int(hidden.sum())
            idx = rng.permutation(free)[:need]
            target.ravel()[idx] = True; hidden.ravel()[idx] = True
            stats["random_fill"] = len(idx)
        if hidden.sum() < budget_total:
            free = np.flatnonzero((~hidden & valid).ravel())
            idx = rng.permutation(free)[: budget_total - int(hidden.sum())]
            target.ravel()[idx] = True; hidden.ravel()[idx] = True; stats["protect_violated"] = len(idx)
        flat_owner = owner_weight.ravel()
        ctx = np.flatnonzero((~hidden).ravel())
        tgt = np.flatnonzero(target.ravel())
        stats["generic_targets"] = int(target.sum()) - n_struct
        for gid in (G_LH, G_RH):  # visible support left: sampler regions (constraint) and true regions (analysis)
            for tag, own in (("", owner), ("_true", owner_weight)):
                reg = (own == gid) & valid
                if reg.sum():
                    stats[f"visible_{GROUPS[gid]}{tag}"] = float((reg & ~hidden).sum() / reg.sum())
        return MaskSample(ctx=ctx, tgt=tgt, tgt_group=flat_owner[tgt], tgt_structured=structured.ravel()[tgt],
                          guard=np.flatnonzero(guard.ravel()), stats=stats)
