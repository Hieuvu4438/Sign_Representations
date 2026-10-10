"""Joint-time masks for proposal B (proposal11 §8.5, §11.5).

Unit = (2-frame bin, joint): masks are expanded to both frames of a bin BEFORE the budget is fixed, so no hidden
frame is clean-averaged into a token. Budget = round(rho x observed cells), rho = 0.5 (raw hidden observations).
Events (one per window): 50% partial finger-chain/joint block of one hand (10-40% of its joints over 1-3 bins,
root and MCP anchors kept), 30% short same-part temporal gap (1-3 bins) with same-part context on both sides,
20% generic joint-time block; then random safe fill to the budget. Body wrist and hand root are tied (§8.1).
Policies: core (as above), whole (whole-part over the window, ablation), generic (fill only, B controls).
"""
from __future__ import annotations

import numpy as np

from oj.data.landmarks import DUPLICATE_WRISTS, FACE, J, LH, PARTS, RH

HAND_ANCHORS = (0, 1, 5, 9, 13, 17)  # root + MCP/CMC anchors kept in finger events


def tie_wrists(m):
    for a, b in DUPLICATE_WRISTS:
        t = m[:, a] | m[:, b]; m[:, a] = t; m[:, b] = t
    return m


class PoseMaskSampler:
    def __init__(self, Tb=32, rho=0.5, policy="core", seed=0):
        assert policy in ("core", "whole", "generic")
        self.Tb, self.rho, self.policy = Tb, rho, policy
        self.rng = np.random.default_rng(seed)

    def sample(self, obs_bins):
        """obs_bins [Tb, J] bool (cell observed in >= 1 frame) -> hidden_bins [Tb, J] bool, stats."""
        rng, Tb = self.rng, self.Tb
        hid = np.zeros((Tb, J), bool)
        budget = int(round(self.rho * obs_bins.sum())); stats = {"budget": budget, "observed": int(obs_bins.sum())}
        if self.policy == "whole":
            name = rng.choice(["lh", "rh", "face", "body"]); hid[:, PARTS[name]] = True; stats["event"] = f"whole_{name}"
        elif self.policy == "core":
            u = rng.random()
            if u < 0.5:  # partial finger-chain / joint block
                part = LH if rng.random() < 0.5 else RH
                cand = [j for j in part if (j - part[0]) not in HAND_ANCHORS]
                k = max(1, int(round(rng.uniform(0.10, 0.40) * len(part))))
                joints = rng.choice(cand, size=min(k, len(cand)), replace=False)
                L = int(rng.integers(1, 4)); b0 = int(rng.integers(0, Tb - L + 1))
                hid[b0:b0 + L, joints] = True; stats["event"] = f"finger_{L}_{len(joints)}"
            elif u < 0.8:  # short same-part temporal gap with context on both sides
                name = rng.choice(["lh", "rh", "face", "body"]); part = PARTS[name]
                L = int(rng.integers(1, 4)); b0 = int(rng.integers(1, max(2, Tb - L)))
                hid[b0:b0 + L, part] = True; stats["event"] = f"gap_{name}_{L}"
            else:  # generic joint-time block
                nj = int(rng.integers(3, 15)); joints = rng.choice(J, size=nj, replace=False)
                L = int(rng.integers(2, 9)); b0 = int(rng.integers(0, Tb - L + 1))
                hid[b0:b0 + L, joints] = True; stats["event"] = "generic_block"
        hid &= obs_bins
        hid = tie_wrists(hid)
        stats["event_cells"] = int((hid & obs_bins).sum())
        # random safe fill (cells), tied wrists stay tied
        free = np.flatnonzero((obs_bins & ~hid).ravel())
        need = budget - int((hid & obs_bins).sum())
        if need > 0:
            for idx in rng.permutation(free):
                if need <= 0:
                    break
                t, j = divmod(idx, J)
                if hid[t, j]:
                    continue
                hid[t, j] = True; need -= 1
                for a, b in DUPLICATE_WRISTS:
                    other = b if j == a else (a if j == b else None)
                    if other is not None and not hid[t, other]:
                        hid[t, other] = True
                        if obs_bins[t, other]:
                            need -= 1
        stats["hidden_cells"] = int((hid & obs_bins).sum())
        return hid, stats


def expand_to_frames(hid_bins):
    return np.repeat(hid_bins, 2, axis=0)
