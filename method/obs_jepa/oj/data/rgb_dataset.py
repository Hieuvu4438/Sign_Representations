"""Window dataset for proposals A and C (RGB student; landmarks used only for train-time masks/losses).

item = one T=16 window at effective fps min(source, 25), letterboxed to S x S (padding excluded from every loss),
masks from RGBMaskSampler (policy), true token ownership for the loss weights, and for C:
  c_reserve : up to `reserve_bins` bins per reliable hand are kept fully visible (complete visible hand bins);
              ALL controls of C must use the same reserved masks (proposal11 §13.4)
  B inputs  : local articulation features of the same physical window (frozen-B teacher targets)
  c_support : [Tb, 3] (lh, rh, face) = part fully observed in both frames AND every token of the part region in the
              bin is visible context (complete visible support, §9.3)
"""
from __future__ import annotations

import numpy as np
import torch

from oj.data.clips import decode, letterbox_frames, make_letterbox, random_start, window_indices
from oj.data.landmarks import FACE, LH, RH, load_landmarks
from oj.features.geometry import local_features
from oj.masks.rgb_masks import G_FACE, G_LH, G_RH, RGBMaskSampler, token_ownership


class RGBWindows(torch.utils.data.Dataset):
    def __init__(self, rows, meta, T=16, size=224, target_fps=25.0, policy="partial", seed=0, c_mode=False,
                 reserve_bins=2, jitter_px=48.0, pose_stats=None):
        self.rows, self.meta, self.T, self.size, self.fps = rows, meta, T, size, target_fps
        self.policy, self.seed, self.c_mode, self.reserve_bins, self.jitter_px = policy, seed, c_mode, reserve_bins, jitter_px
        self.pose_stats = pose_stats
        self.Tb, self.g = T // 2, size // 16

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed * 1_000_003 + i * 7919 + np.random.randint(1 << 30)) % (1 << 63))
        r = self.rows[i]; W, H, fps, n = self.meta[r["sample_id"]]
        lm = load_landmarks(r["sample_id"], W, H); n = min(n, len(lm.xy))
        idx, ts, eff = window_indices(n, fps, self.T, self.fps, random_start(n, fps, self.T, self.fps, rng))
        frames, fvalid = decode(r["video"], idx)
        lb = make_letterbox(W, H, self.size)
        video = letterbox_frames(frames, lb)
        cl = np.clip(idx, 0, n - 1); pad = (idx < 0) | ~fvalid
        xy_src = np.where(pad[:, None, None], np.nan, lm.xy[cl]); obs = lm.observed[cl] & ~pad[:, None]
        xy = lb.apply(xy_src)
        tok_valid = np.broadcast_to(lb.token_valid()[None], (self.Tb, self.g, self.g)).copy()
        tok_valid &= np.repeat(~pad.reshape(self.Tb, 2).all(1), 1)[:, None, None]  # a bin with no decoded frame is invalid
        own_true, rel_true = token_ownership(xy, obs, self.Tb, self.g)
        if self.policy == "jitter":
            off = rng.uniform(-self.jitter_px, self.jitter_px, size=2)
            own_mask, rel_mask = token_ownership(xy, obs, self.Tb, self.g, offset=off)
        else:
            own_mask, rel_mask = own_true, rel_true
        valid_for_mask = tok_valid.copy()
        reserved = np.zeros_like(tok_valid)
        if self.c_mode:  # complete visible hand bins: reserve them before any masking
            for gid in (G_LH, G_RH):
                bins = np.flatnonzero(rel_true[:, gid])
                for b in rng.permutation(bins)[: self.reserve_bins]:
                    reserved[b] |= (own_true[b] == gid) & tok_valid[b]
            valid_for_mask &= ~reserved
        sm = RGBMaskSampler(self.T, self.size, policy=self.policy, seed=int(rng.integers(1 << 31)))
        if self.c_mode:
            # budget computed on all valid tokens; reserved tokens are forced into context
            m = sm.sample(own_mask, rel_mask, valid_for_mask, owner_weight=own_true)
            Nv = int(tok_valid.sum()); budget = int(round(sm.rho * Nv))
            extra_ctx = np.flatnonzero(reserved.ravel())
            ctx = np.union1d(m.ctx, extra_ctx)
            hidden = len(m.tgt) + len(m.guard)
            if hidden < budget:  # move random context (non-reserved) to generic targets to restore the 80% budget
                cand = np.setdiff1d(ctx, extra_ctx); add = rng.permutation(cand)[: budget - hidden]
                ctx = np.setdiff1d(ctx, add)
                tgt = np.sort(np.concatenate([m.tgt, add])); order = np.argsort(np.concatenate([m.tgt, add]))
                grp = np.concatenate([m.tgt_group, own_true.ravel()[add]])[order]
                struct = np.concatenate([m.tgt_structured, np.zeros(len(add), bool)])[order]
            else:
                tgt, grp, struct = m.tgt, m.tgt_group, m.tgt_structured
            m.stats["reserved_tokens"] = len(extra_ctx)
            ctx_idx, tgt_idx, guard = ctx, tgt, m.guard
        else:
            m = sm.sample(own_mask, rel_mask, tok_valid, owner_weight=own_true)
            ctx_idx, tgt_idx, grp, struct, guard = m.ctx, m.tgt, m.tgt_group, m.tgt_structured, m.guard
        item = {"video": torch.from_numpy(video), "ctx": torch.from_numpy(ctx_idx.astype(np.int64)),
                "tgt": torch.from_numpy(tgt_idx.astype(np.int64)), "tgt_group": torch.from_numpy(np.asarray(grp, np.int64)),
                "tgt_struct": torch.from_numpy(np.asarray(struct, bool)), "n_guard": len(guard), "stats": m.stats,
                "owner": torch.from_numpy(own_true.reshape(-1).astype(np.int64))}
        if self.c_mode:
            ctx_mask = np.zeros(tok_valid.size, bool); ctx_mask[ctx_idx] = True; ctx_mask = ctx_mask.reshape(tok_valid.shape)
            sup = np.zeros((self.Tb, 3), bool)
            for k, (gid, part) in enumerate(((G_LH, LH), (G_RH, RH), (G_FACE, FACE))):
                for b in range(self.Tb):
                    region = own_true[b] == gid
                    full = obs[2 * b:2 * b + 2][:, part].all()
                    sup[b, k] = full and region.any() and ctx_mask[b][region].all()
            q = np.where(obs, lm.quality[cl], 0.0).astype(np.float32)
            P = np.nan_to_num(xy_src).astype(np.float32)
            loc = local_features(P, obs, q)
            S = self.pose_stats
            packed = {}
            for k, x in loc.items():
                x = x.copy()
                if S is not None and S.z is not None:
                    x[..., 0:4] = np.where(x[..., 4:6].repeat(2, -1) > 0, S.norm("local", x[..., 0:4]), 0.0)
                packed[k] = torch.from_numpy(x.reshape(self.Tb, 2, -1, x.shape[-1]).transpose(0, 2, 1, 3).reshape(self.Tb, -1, 2 * x.shape[-1]))
            item.update({"b_loc": packed, "b_vis": torch.from_numpy(obs.reshape(self.Tb, 2, -1).any(1)),
                         "c_support": torch.from_numpy(sup)})
        return item


def collate(items):
    """equal context length per batch (vendor encoder): K = min |C|; surplus context -> hidden guard (logged).
    Targets padded to max |M| with a repeated index and weight 0."""
    K = min(len(it["ctx"]) for it in items); M = max(len(it["tgt"]) for it in items)
    ctx, tgt, grp, w, moved, trimmed = [], [], [], [], 0, []
    for it in items:
        c = it["ctx"]; trimmed.append(len(c) > K)
        if len(c) > K:
            keep = torch.randperm(len(c))[:K].sort().values; moved += len(c) - K; c = c[keep]
        ctx.append(c)
        t = it["tgt"]; pad = M - len(t)
        tgt.append(torch.cat([t, t[:1].repeat(pad)])); grp.append(torch.cat([it["tgt_group"], it["tgt_group"][:1].repeat(pad)]))
        w.append(torch.cat([torch.ones(len(t)), torch.zeros(pad)]))
    out = {"video": torch.stack([it["video"] for it in items]), "ctx": torch.stack(ctx), "tgt": torch.stack(tgt),
           "tgt_group": torch.stack(grp), "tgt_w": torch.stack(w), "owner": torch.stack([it["owner"] for it in items]),
           "ctx_moved_to_guard": moved, "stats": [it["stats"] for it in items]}
    if "b_loc" in items[0]:
        out["b_loc"] = {k: torch.stack([it["b_loc"][k] for it in items]) for k in items[0]["b_loc"]}
        out["b_vis"] = torch.stack([it["b_vis"] for it in items])
        sup = torch.stack([it["c_support"] for it in items])
        sup[torch.tensor(trimmed)] = False  # trimmed context may break complete visible support -> no C loss there
        out["c_support"] = sup
    return out
