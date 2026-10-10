"""Unlabelled ASL Citizen clips -> one random 16-frame window (uint8, 224x224, squash resize as SignRep)
plus frame-level prior targets from the local COCO-WholeBody keypoints. Gloss/signer fields are never read."""
from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[0] / "p4_signssl"))
from p4.data import aslc_rows  # noqa: E402
from jt.priors import DIMS, compute  # noqa: E402

KP_DIR = Path("/home/dongvk/datasets/ASL_Citizen/keypoints")


def load_kp(sample_id):
    d = pickle.load(open(KP_DIR / f"{sample_id}.pkl", "rb"))
    T = len(d["keypoints"])
    K = np.full((T, 133, 2), np.nan, np.float32); Sc = np.zeros((T, 133), np.float32)
    for t, (k, s) in enumerate(zip(d["keypoints"], d["scores"])):
        k, s = np.asarray(k), np.asarray(s)
        if k.ndim == 3 and k.shape[0] == 1:
            K[t], Sc[t] = k[0], s[0]
    return K, Sc


def decode_window(path, t0, n=16, size=224):
    cap = cv2.VideoCapture(str(path))
    for _ in range(t0):
        if not cap.grab():
            break
    out = []
    while len(out) < n:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_LINEAR))
    cap.release()
    if not out:
        return np.zeros((n, size, size, 3), np.uint8)
    while len(out) < n:
        out.append(out[-1])
    return np.stack(out)


class PriorStats:
    """Robust per-dim standardisation (median, 1.4826*MAD) fit on unlabelled train windows; clip +-5."""

    def __init__(self, path):
        z = np.load(path)
        self.s = {g: (z[g + "_med"], z[g + "_mad"]) for g in DIMS if g + "_med" in z}

    def __call__(self, g, v):
        med, mad = self.s[g]
        return np.clip((v - med) / mad, -5, 5).astype(np.float32)


def fit_prior_stats(rows, path, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    acc = {g: [] for g in DIMS}
    for i in rng.choice(len(rows), min(n, len(rows)), replace=False):
        K, Sc = load_kp(rows[i]["sample_id"])
        for g, (v, m) in compute(K, Sc, list(DIMS)).items():
            acc[g].append(np.where(m, v, np.nan))
    out = {}
    for g, a in acc.items():
        a = np.concatenate(a)
        med = np.nanmedian(a, 0)
        mad = 1.4826 * np.nanmedian(np.abs(a - med), 0)
        out[g + "_med"] = np.nan_to_num(med).astype(np.float32)
        out[g + "_mad"] = np.maximum(np.nan_to_num(mad, nan=1.0), 1e-3).astype(np.float32)
    np.savez(path, **out)


class ClipWindows(torch.utils.data.Dataset):
    def __init__(self, split="train", groups=(), stats=None, seed=0):
        self.rows = aslc_rows(split)
        self.groups, self.stats, self.seed = list(groups), stats, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rng = random.Random(hash((self.seed, i, random.random())))
        K, Sc = load_kp(r["sample_id"])
        n = len(K)
        t0 = rng.randint(0, max(0, n - 16))
        x = decode_window(r["path"], t0)
        K, Sc = K[t0:t0 + 16], Sc[t0:t0 + 16]
        if len(K) < 16:
            pad = 16 - len(K)
            K = np.concatenate([K, np.full((pad, 133, 2), np.nan, np.float32)]); Sc = np.concatenate([Sc, np.zeros((pad, 133), np.float32)])
        item = {"video": torch.from_numpy(x)}
        for g, (v, m) in compute(K, Sc, self.groups).items():
            item[g] = torch.from_numpy(self.stats(g, v)); item[g + "_mask"] = torch.from_numpy(m)
        return item
