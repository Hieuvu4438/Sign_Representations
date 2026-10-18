"""Shared loaders and the frozen-dictionary retrieval metric for proposal17 diagnostics.

Everything here is read-only on the existing PLctx full160 feature caches. Only TRAIN (gallery) and VAL (queries) are used;
the official test split is never touched (it is quarantined, see docs/proposal16 and the continuation report).

Embedding = `_grp` clip embedding (768-d: mean over windows of the group-concatenated token readout), exactly what the
SignRep-style dictionary protocol L2-normalises and scores.
"""
from __future__ import annotations

import json
import os

import numpy as np

PJ = "/home/haipd/Sign_Representations/method/phonojepa/runtime"
JOB = f"{PJ}/plctx160_continuation/job.json"
TRAIN_DIR = f"{PJ}/eval/PJ_PLctx_FULL_160ep_dictionary/aslc/train"  # baseline (original final.pt) gallery
VAL_DIR = f"{PJ}/plctx160_continuation/baseline/feature_cache/val"  # baseline val queries
SCRATCH = "/tmp/claude-1007/-home-haipd-Sign-Representations/32c5829a-820b-4c3b-9d20-e66b08713d70/scratchpad"


def manifests():
    d = json.load(open(JOB))["manifests"]
    return {k: d[k] for k in ("train", "val")}


def _load_dir(path):
    ids, feats, nwin = [], [], []
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".npz"):
            continue
        z = np.load(os.path.join(path, fn), allow_pickle=False)
        ids.extend(z["ids"].tolist())
        feats.append(z["_grp"].astype(np.float32))
        nwin.append(z["nwin"])
    return ids, np.concatenate(feats), np.concatenate(nwin)


def load_baseline():
    """Return dict with X (n,768), gloss label int, signer str for train and val, aligned with the manifests."""
    man = manifests()
    glosses = sorted({m["gloss"] for m in man["train"]} | {m["gloss"] for m in man["val"]})
    gidx = {g: i for i, g in enumerate(glosses)}
    out = {}
    for split, path in (("train", TRAIN_DIR), ("val", VAL_DIR)):
        ids, X, nwin = _load_dir(path)
        by_id = {m["sample_id"]: m for m in man[split]}
        meta = [by_id[i] for i in ids]
        out[split] = dict(
            X=X,
            nwin=nwin,
            ids=np.array(ids),
            y=np.array([gidx[m["gloss"]] for m in meta]),
            signer=np.array([m["signer"] for m in meta]),
            gloss=np.array([m["gloss"] for m in meta]),
        )
    out["glosses"] = glosses
    return out


def l2(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def class_max_ranks(Q, yq, G, yg, chunk=512):
    """SignRep dictionary protocol: cosine, class score = max over gallery clips of that class, pessimistic ties.
    Q,G are L2-normalised inside. Returns the rank (1-based) of the correct class for every query."""
    Q, G = l2(Q.astype(np.float32)), l2(G.astype(np.float32))
    order = np.argsort(yg, kind="stable")
    Gs, ygs = G[order], yg[order]
    classes, starts = np.unique(ygs, return_index=True)
    pos = {c: i for i, c in enumerate(classes)}
    ranks = np.empty(len(Q), dtype=np.int64)
    for a in range(0, len(Q), chunk):
        s = Q[a : a + chunk] @ Gs.T
        sc = np.maximum.reduceat(s, starts, axis=1)  # (q, n_classes)
        tgt = np.array([sc[i, pos[y]] if y in pos else -np.inf for i, y in enumerate(yq[a : a + chunk])])
        ranks[a : a + chunk] = 1 + (sc >= tgt[:, None]).sum(1) - 1  # -1: the correct class itself counts in >=
    return ranks


def metrics(ranks):
    r = np.asarray(ranks)
    return dict(R1=float((r <= 1).mean() * 100), R5=float((r <= 5).mean() * 100), R10=float((r <= 10).mean() * 100),
                MRR=float((1.0 / r).mean() * 100), DCG=float((1.0 / np.log2(1 + r)).mean() * 100))


def boot_ci_by_signer_gloss(ranks, signer, gloss, k=1, n=1000, seed=0):
    """95% CI for R@k, resampling clusters = signer x gloss pairs-agnostic: signers then glosses (two-stage)."""
    rng = np.random.default_rng(seed)
    hit = (np.asarray(ranks) <= k).astype(float)
    sg = np.unique(signer)
    by_signer = {s: np.where(signer == s)[0] for s in sg}
    vals = []
    for _ in range(n):
        ss = rng.choice(sg, size=len(sg), replace=True)
        idx = np.concatenate([by_signer[s] for s in ss])
        gl = np.unique(gloss[idx])
        gs = set(rng.choice(gl, size=len(gl), replace=True).tolist())
        keep = np.array([g in gs for g in gloss[idx]])
        vals.append(hit[idx][keep].mean() * 100)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
