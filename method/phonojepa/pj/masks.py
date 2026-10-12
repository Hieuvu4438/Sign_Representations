"""Predictability-Ladder Channel Masking (PLCM) — proposal12 v2.1 §6.6.

Content-independent: the sampler only sees the number of valid bins (clip length / window placement) and an RNG.
Grid = (NC = 7 channels, Tb = 32 bins). Families:
  R0  random (channel, 2-bin) tubes                                  [control]
  R1  finger groups inside a hand CONF token (joint-level, partial)   [gate G2]
  R2  time blocks (all channels) of 4..8 bins
  R3  one HAND channel over a long span (>= 4 bins); no face channels (D8: face channels couple up to 0.9 per column)
  R4  whole articulator (dominant hand, non-dominant hand, both hands, face) over a long span   [far view, gate G2 / R4 row]
Mask families per factorial row (§9): 'PL' = PLCM-lite (near R2 / far R3), 'PLfull' = near R2(+R1) / far R4|R3,
'R4' = R4 in both views, 'R0' = R0 in both views. Closure F_HD => F_UP, F_MO is applied for every family.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pj.joints import D_CONF, D_LOC, F_HD, F_MO, F_UP, FINGER_GROUPS, N_CONF, N_LOC, NC

R3_CHANNELS = (D_CONF, D_LOC, N_CONF, N_LOC)
R3_PROBS = (0.35, 0.25, 0.25, 0.15)
R4_ARTICULATORS = ((D_CONF, D_LOC), (N_CONF, N_LOC), (D_CONF, D_LOC, N_CONF, N_LOC), (F_UP, F_MO, F_HD))
R4_PROBS = (0.40, 0.15, 0.15, 0.30)
FINGER_GROUP_NAMES = tuple(FINGER_GROUPS)


@dataclass
class ViewMask:
    valid: np.ndarray                    # [Tb] bool (bin contains at least one in-clip frame)
    hidden: np.ndarray                   # [NC, Tb] bool, fully hidden cells (removed from the student, targets)
    r1: np.ndarray                       # [Tb, 2, 21] bool, joint-level hidden finger joints (0 = D hand, 1 = N hand)
    drop: np.ndarray                     # [NC, Tb] bool, context-dropped cells (not in the student, not targets)
    stats: dict = field(default_factory=dict)

    @property
    def r1_cells(self):
        c = np.zeros_like(self.hidden)
        c[D_CONF] = self.r1[:, 0].any(-1)
        c[N_CONF] = self.r1[:, 1].any(-1)
        return c

    @property
    def target(self):
        return (self.hidden | self.r1_cells) & self.valid[None]

    @property
    def visible(self):
        return self.valid[None] & ~self.hidden & ~self.drop


class Budget:
    def __init__(self, n_valid, rate, rng):
        self.Tb_valid = n_valid
        self.cap = max(1, int(round(rate * NC * n_valid)))
        self.cells = np.zeros((NC, n_valid), bool)
        self.r1 = np.zeros((n_valid, 2, 21), bool)
        self.fails = 0
        self.rng = rng

    @property
    def used(self):
        r1c = np.stack([self.r1[:, 0].any(-1), self.r1[:, 1].any(-1)])
        extra = (r1c & ~self.cells[[D_CONF, N_CONF]]).sum()
        return int(self.cells.sum() + extra)

    def remaining(self):
        return self.cap - self.used

    def open(self):
        return self.remaining() > 0 and self.fails < 10

    def add_cells(self, chans, t0, t1):
        new = np.zeros_like(self.cells)
        new[list(chans), t0:t1] = True
        n_new = int((new & ~self.cells).sum())
        if n_new == 0 or n_new > self.remaining() + max(2, int(0.1 * self.cap)):
            self.fails += 1
            return False
        self.cells |= new; self.fails = 0
        return True

    def add_r1(self, hand, joints, t0, t1):
        cell_new = not self.r1[t0:t1, hand].any() and (t1 - t0) <= self.remaining() + 2
        if not cell_new:
            self.fails += 1
            return False
        self.r1[t0:t1, hand, joints] = True; self.fails = 0
        return True


def _span(rng, n_valid, lo_frac, hi_frac, min_len=4, cap=None):
    L = int(round(rng.uniform(lo_frac, hi_frac) * n_valid))
    L = max(min(min_len, n_valid), min(L, n_valid))
    if cap is not None:
        L = min(L, max(min(min_len, n_valid), cap))
    t0 = int(rng.integers(0, n_valid - L + 1))
    return t0, t0 + L


def _r0(b, rng):
    c = int(rng.integers(NC)); t0 = int(rng.integers(0, max(1, b.Tb_valid - 1))); b.add_cells([c], t0, min(t0 + 2, b.Tb_valid))


def _r1(b, rng):
    hand = 0 if rng.random() < 0.75 else 1
    g = FINGER_GROUPS[FINGER_GROUP_NAMES[int(rng.integers(len(FINGER_GROUP_NAMES)))]]
    L = int(rng.integers(4, 9)); L = min(L, b.Tb_valid); t0 = int(rng.integers(0, b.Tb_valid - L + 1))
    b.add_r1(hand, g, t0, t0 + L)


def _r2(b, rng):
    L = int(rng.integers(4, 9)); L = min(L, b.Tb_valid); t0 = int(rng.integers(0, b.Tb_valid - L + 1))
    b.add_cells(range(NC), t0, t0 + L)


PARTNER = {D_CONF: D_LOC, D_LOC: D_CONF, N_CONF: N_LOC, N_LOC: N_CONF}


def _r3(b, rng, max_span):
    """one hand channel over a long span; bins where the other channel of the same hand is already hidden are skipped,
    so the other channel of the same hand always stays visible (§6.6; code-review finding #2)."""
    c = R3_CHANNELS[int(rng.choice(len(R3_CHANNELS), p=R3_PROBS))]
    t0, t1 = _span(rng, b.Tb_valid, 0.3, max_span, cap=b.remaining())
    free = ~b.cells[PARTNER[c], t0:t1]
    if not free.any():
        b.fails += 1
        return
    new = np.zeros_like(b.cells)
    new[c, t0:t1] = free
    n_new = int((new & ~b.cells).sum())
    if n_new == 0 or n_new > b.remaining() + max(2, int(0.1 * b.cap)):
        b.fails += 1
        return
    b.cells |= new; b.fails = 0


def _r4(b, rng, max_span):
    art = R4_ARTICULATORS[int(rng.choice(len(R4_ARTICULATORS), p=R4_PROBS))]
    t0, t1 = _span(rng, b.Tb_valid, 0.4, max_span, cap=max(1, b.remaining() // len(art)))
    b.add_cells(art, t0, t1)


def _close_face(cells):
    cells[F_UP] |= cells[F_HD]
    cells[F_MO] |= cells[F_HD]
    return cells


def sample_view(n_valid, Tb, rng, kind, family="PL", epoch_frac=1.0, curriculum=False, use_r1=False,
                ctx_drop=0.10, min_visible=0.25, max_tries=20):
    """kind: 'near' | 'far'. Returns ViewMask on the full Tb grid (bins >= n_valid are padding)."""
    valid = np.zeros(Tb, bool); valid[:n_valid] = True
    if n_valid == 0:
        z = np.zeros((NC, Tb), bool)
        return ViewMask(valid, z, np.zeros((Tb, 2, 21), bool), z.copy(), {"empty": True})
    max_span = (0.5 + 0.5 * min(1.0, epoch_frac / 0.3)) if curriculum else 1.0
    for _ in range(max_tries):
        rate = rng.uniform(0.4, 0.7) if kind == "near" else rng.uniform(0.3, 0.6)
        b = Budget(n_valid, rate, rng)
        if kind == "far":
            # matched far-view budget for EVERY family (review round 2, N1, option A): R3 can only hide one channel per
            # hand and bin, so the far budget is rate U(0.3, 0.6) of the 2 x n_valid (hand, bin) slots (~9-17% of all
            # cells); R0 / R4 far views get the same cap so the three factorial rows differ only in mask family.
            b.cap = max(1, int(round(rate * 2 * n_valid)))
        if family == "R0":
            while b.open():
                _r0(b, rng)
        elif family == "R4":
            while b.open():
                _r4(b, rng, 1.0)
        elif kind == "near":                                   # PL / PLfull near view: R2 (+ R1 when enabled)
            while b.open():
                (_r1 if (use_r1 and rng.random() < 0.5) else _r2)(b, rng)
        else:                                                  # far view
            if family == "PLfull" and rng.random() < 0.4:
                # G2: one uncapped articulator span U(0.4, max_span) x n_valid (spec minimum 0.4 T, review N2);
                # its cells are added on top of the R3 budget so R3 still contributes hand-channel targets
                art = R4_ARTICULATORS[int(rng.choice(len(R4_ARTICULATORS), p=R4_PROBS))]
                t0, t1 = _span(rng, b.Tb_valid, 0.4, max_span)
                b.cells[list(art), t0:t1] = True
                b.cap += int(b.cells.sum())
            while b.open():
                _r3(b, rng, max_span)
        cells = _close_face(b.cells.copy())
        hidden = np.zeros((NC, Tb), bool); hidden[:, :n_valid] = cells
        r1 = np.zeros((Tb, 2, 21), bool); r1[:n_valid] = b.r1
        r1[:, 0] &= ~hidden[D_CONF][:, None]; r1[:, 1] &= ~hidden[N_CONF][:, None]
        vis = valid[None] & ~hidden
        drop = np.zeros_like(hidden)
        cand = np.argwhere(vis)
        k = int(round(ctx_drop * len(cand)))
        if k:
            sel = cand[rng.choice(len(cand), k, replace=False)]
            drop[sel[:, 0], sel[:, 1]] = True
        n_cells = NC * n_valid
        if (vis & ~drop).sum() >= min_visible * n_cells and (hidden.any() or r1.any()):
            return ViewMask(valid, hidden, r1, drop, {"rate": rate, "hidden_frac": float(hidden.sum() / n_cells),
                                                       "r1_cells": int(r1.any(-1).sum()), "family": family, "kind": kind})
    # fallback (very short clips): a single R2 block of length 1..n_valid
    hidden = np.zeros((NC, Tb), bool); L = max(1, n_valid // 2); hidden[:, :L] = True
    return ViewMask(valid, hidden, np.zeros((Tb, 2, 21), bool), np.zeros_like(hidden), {"fallback": True})


def random_view(n_valid, Tb, rng, rate, family_kind="R0"):
    """baselines B-MAE / B-MAMP: R0 tubes at a fixed rate (e.g. 0.6 / 0.9), no context dropout."""
    valid = np.zeros(Tb, bool); valid[:n_valid] = True
    b = Budget(max(n_valid, 1), rate, rng)
    while b.open():
        _r0(b, rng)
    hidden = np.zeros((NC, Tb), bool); hidden[:, :n_valid] = _close_face(b.cells.copy())[:, :n_valid]
    return ViewMask(valid, hidden, np.zeros((Tb, 2, 21), bool), np.zeros_like(hidden), {"rate": rate, "family": family_kind})
