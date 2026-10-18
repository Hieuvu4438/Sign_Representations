"""Window dataset for proposal B (and the frozen-B teacher of C).

Each item = one T=64 window at effective fps min(source, 25) from an unlabelled clip, with
  student inputs  : node features / local features from VISIBLE observations (observed & ~synthetic mask)
  teacher inputs  : the same features from all observations, normalised with the context-estimated anchors
  targets         : hidden cells (bin, joint), observable anchors from the student's visible support,
                    raw targets for B10 (body-relative + hand-root-relative xy of both frames, teacher side)
Gloss/signer fields are never read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from oj.data.clips import random_start, window_indices
from oj.data.landmarks import REASON_PADDING, REASON_SYNTHETIC, load_landmarks
from oj.features.geometry import ANCHOR_DIMS, anchor_targets, body_anchors, local_features, node_features
from oj.masks.pose_masks import PoseMaskSampler, expand_to_frames


def read_corpus(path):
    rows = []
    for l in open(path):
        r = json.loads(l)
        rows.append({"sample_id": Path(r["video"]).stem, "video": r["video"]})
    return rows


class PoseStats:
    """training-set mean/std of the value slots (node 0:8, local 0:4, anchors) for normalisation (§8.2)."""

    def __init__(self, path=None):
        self.z = dict(np.load(path)) if path and Path(path).exists() else None

    def norm(self, key, x, valid=None):
        if self.z is None:
            return x
        return (x - self.z[key + "_mean"]) / self.z[key + "_std"]


class PoseWindows(torch.utils.data.Dataset):
    def __init__(self, rows, meta, T=64, target_fps=25.0, policy="core", rho=0.5, stats=None, seed=0, train=True,
                 start_s=None):
        self.rows, self.meta, self.T, self.fps, self.train = rows, meta, T, target_fps, train
        self.policy, self.rho, self.seed, self.stats = policy, rho, seed, stats or PoseStats()
        self.start_s = start_s
        self._cache = {}

    def __len__(self):
        return len(self.rows)

    def _lm(self, i):
        if i not in self._cache:
            r = self.rows[i]; W, H, fps, n = self.meta[r["sample_id"]]
            self._cache[i] = (load_landmarks(r["sample_id"], W, H), (W, H, fps, n))
        return self._cache[i]

    def build(self, i, rng, start=None, mask=True):
        lm, (W, H, fps, n) = self._lm(i)
        n = min(n, len(lm.xy))
        st = start if start is not None else (random_start(n, fps, self.T, self.fps, rng) if self.train else 0.0)
        idx, ts, eff = window_indices(n, fps, self.T, self.fps, st)
        pad = idx < 0; cl = np.clip(idx, 0, n - 1)
        P = np.where(pad[:, None, None], 0.0, np.nan_to_num(lm.xy[cl])).astype(np.float32)
        obs = lm.observed[cl] & ~pad[:, None]; q = np.where(obs, lm.quality[cl], 0.0).astype(np.float32)
        reason = np.where(pad[:, None], REASON_PADDING, lm.reason[cl])
        Tb = self.T // 2
        obs_bins = obs.reshape(Tb, 2, -1).any(1)
        if mask:
            hid_b, mstats = PoseMaskSampler(Tb, self.rho, self.policy, seed=int(rng.integers(1 << 31))).sample(obs_bins)
        else:
            hid_b, mstats = np.zeros_like(obs_bins), {}
        hid = expand_to_frames(hid_b)
        vis = obs & ~hid
        r_st = np.where(hid & obs, REASON_SYNTHETIC, reason)
        anchors = body_anchors(P, vis, (W, H))  # from VISIBLE body anchors; teacher reuses them
        st_node = node_features(P, vis, obs, q, r_st, ts, anchors)
        te_node = node_features(P, obs, obs, q, reason, ts, anchors)
        st_loc, te_loc = local_features(P, vis, q), local_features(P, obs, q)
        anc = anchor_targets(P, vis, anchors)
        S = self.stats
        def node_pack(x):
            x = x.copy(); x[..., 0:8] = np.where(x[..., 8:12].repeat(2, -1) > 0, S.norm("node", x[..., 0:8]), 0.0)
            return x.reshape(Tb, 2, -1, x.shape[-1]).transpose(0, 2, 1, 3).reshape(Tb, -1, 2 * x.shape[-1])
        def loc_pack(d):
            out = {}
            for k, x in d.items():
                x = x.copy(); x[..., 0:4] = np.where(x[..., 4:6].repeat(2, -1) > 0, S.norm("local", x[..., 0:4]), 0.0)
                out[k] = torch.from_numpy(x.reshape(Tb, 2, -1, x.shape[-1]).transpose(0, 2, 1, 3).reshape(Tb, -1, 2 * x.shape[-1]))
            return out
        raw = te_node[..., [0, 1, 2, 3]].reshape(Tb, 2, -1, 4).transpose(0, 2, 1, 3).reshape(Tb, -1, 8)
        raw_valid = te_node[..., [8, 8, 9, 9]].reshape(Tb, 2, -1, 4).transpose(0, 2, 1, 3).reshape(Tb, -1, 8) > 0
        item = {
            "st_node": torch.from_numpy(node_pack(st_node)), "te_node": torch.from_numpy(node_pack(te_node)),
            "st_loc": loc_pack(st_loc), "te_loc": loc_pack(te_loc),
            "vis": torch.from_numpy(vis.reshape(Tb, 2, -1).any(1)), "obs": torch.from_numpy(obs_bins),
            "hid": torch.from_numpy(hid_b & obs_bins),
            "t_bin": torch.from_numpy(ts.reshape(Tb, 2).mean(1).astype(np.float32) - np.float32(ts[0])),
            "raw": torch.from_numpy(raw.astype(np.float32)),
            "raw_valid": torch.from_numpy(raw_valid),
            "body_anchor_ok": torch.tensor(anchors[2]),
        }
        for k, (v, m) in anc.items():
            item["anc_" + k] = torch.from_numpy(S.norm("anc_" + k, v) if S.z is not None else v)
            item["ancm_" + k] = torch.from_numpy(m)
        item["mask_stats"] = mstats
        return item

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed * 1_000_003 + i * 7919 + np.random.randint(1 << 30)) % (1 << 63))
        return self.build(i, rng)


def collate(items):
    out = {}
    for k in items[0]:
        if k == "mask_stats":
            out[k] = [it[k] for it in items]
        elif isinstance(items[0][k], dict):
            out[k] = {kk: torch.stack([it[k][kk] for it in items]) for kk in items[0][k]}
        else:
            out[k] = torch.stack([it[k] for it in items])
    return out


def fit_stats(ds, path, n=600, seed=0):
    """value-slot statistics from unmasked training windows (no labels)."""
    rng = np.random.default_rng(seed); acc = {"node": [], "local": []}; anc = {k: [] for k in ANCHOR_DIMS}
    for i in rng.choice(len(ds), min(n, len(ds)), replace=False):
        it = ds.build(int(i), rng, mask=False)
        x = it["te_node"].numpy().reshape(-1, 2, it["te_node"].shape[-1] // 2).reshape(-1, it["te_node"].shape[-1] // 2)
        acc["node"].append(np.where(np.repeat(x[:, 8:12], 2, -1) > 0, x[:, 0:8], np.nan))
        for k, v in it["te_loc"].items():
            y = v.numpy().reshape(-1, v.shape[-1] // 2)
            acc["local"].append(np.where(np.repeat(y[:, 4:6], 2, -1) > 0, y[:, 0:4], np.nan))
        for k in ANCHOR_DIMS:
            anc[k].append(np.where(it["ancm_" + k].numpy(), it["anc_" + k].numpy(), np.nan))
    z = {}
    for k, v in acc.items():
        v = np.concatenate(v); z[k + "_mean"] = np.nan_to_num(np.nanmean(v, 0)).astype(np.float32)
        z[k + "_std"] = np.maximum(np.nan_to_num(np.nanstd(v, 0), nan=1.0), 1e-3).astype(np.float32)
    for k, v in anc.items():
        v = np.concatenate(v); z["anc_" + k + "_mean"] = np.nan_to_num(np.nanmean(v, 0)).astype(np.float32)
        z["anc_" + k + "_std"] = np.maximum(np.nan_to_num(np.nanstd(v, 0), nan=1.0), 1e-3).astype(np.float32)
    np.savez(path, **z)
