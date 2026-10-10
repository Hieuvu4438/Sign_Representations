"""Paired uncertainty for per-query outcomes: signer-cluster and video-level bootstrap."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


def load_per_query(path: Path, value: str) -> Dict[str, tuple]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if value == "rr":
                v = 1.0 / r["rank"]
            elif value == "r1":
                v = float(r["rank"] == 1)
            elif value == "acc":
                v = float(r["pred"] == r["label"])
            else:
                raise ValueError(value)
            out[r["sample_id"]] = (v, r["signer"])
    return out


def paired_bootstrap(a: Dict[str, tuple], b: Dict[str, tuple], n_boot: int = 2000, seed: int = 0) -> dict:
    """Delta = mean(b) - mean(a) over identical query IDs."""
    if set(a) != set(b):
        raise ValueError("paired comparison requires identical query IDs")
    ids = sorted(a)
    va = np.array([a[i][0] for i in ids])
    vb = np.array([b[i][0] for i in ids])
    signers = np.array([a[i][1] for i in ids])
    d = vb - va
    rng = np.random.default_rng(seed)
    # video-level (independent queries)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    vid = d[idx].mean(1)
    # signer-cluster bootstrap: resample signers, then videos within signer
    groups = [np.flatnonzero(signers == s) for s in np.unique(signers)]
    clus = np.empty(n_boot)
    for k in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        parts = [d[groups[p][rng.integers(0, len(groups[p]), size=len(groups[p]))]] for p in pick]
        clus[k] = np.concatenate(parts).mean()
    per_signer = {str(s): float(d[signers == s].mean()) for s in np.unique(signers)}
    return {
        "delta": float(d.mean()),
        "ci95_video": [float(np.percentile(vid, 2.5)), float(np.percentile(vid, 97.5))],
        "ci95_signer_cluster": [float(np.percentile(clus, 2.5)), float(np.percentile(clus, 97.5))],
        "num_queries": int(len(d)),
        "num_signers": int(len(groups)),
        "per_signer_delta": per_signer,
        "signers_positive": int(sum(v > 0 for v in per_signer.values())),
    }
