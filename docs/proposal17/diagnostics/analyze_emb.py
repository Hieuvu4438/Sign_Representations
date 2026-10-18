"""Analysis of the multi-variant embeddings written by emb_variants.py (proposal17 diagnostics G5-G7).

Gallery = train every-2nd clip (n~20k), queries = val every-2nd clip (n~5.2k); class-max cosine, pessimistic ties (common.py).
For every (variant, readout) we report
   R@1 raw | R@1 label-free whitened (PCA-512 fit on the gallery)          [what a readout/layer choice buys]
   LDA ceiling: supervised linear map fit on the gallery (PCA-512 -> LDA-256) [how much lexical information the readout holds]
   gloss eta^2 and signer eta^2 of the clip embedding (train half)          [share of variance]
   signer retention: top-1 NN (within the train half) same-signer rate and same-gloss rate
so that trained vs RANDOM-INIT encoder (retention ratio) and teacher vs student, per layer, are directly comparable.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from common import *  # noqa
import geo_headroom as G

EMB = f"{SCRATCH}/emb"
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/emb_analysis.json"
READ = ["grp_L1", "grp_L2", "grp_L3", "grp_L4", "grp_L5", "grp_L6", "grp_fin", "ord2", "ord4", "ord4_L4", "ch_fin", "cls_fin"]


def meta(ids, man):
    by = {m["sample_id"]: m for m in man}
    return [by[i] for i in ids]


def eta2(X, g):
    Xc = X - X.mean(0)
    tot = (Xc ** 2).sum()
    ss = 0.0
    for u in np.unique(g):
        m = g == u
        ss += m.sum() * (Xc[m].mean(0) ** 2).sum()
    return float(ss / tot)


def nn_stats(X, y, s, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    Xn = l2(X)
    S = Xn[idx] @ Xn.T
    S[np.arange(len(idx)), idx] = -2
    nn = S.argmax(1)
    return float((s[nn] == s[idx]).mean() * 100), float((y[nn] == y[idx]).mean() * 100)


def main():
    man = manifests()
    gl = sorted({m["gloss"] for sp in man.values() for m in sp})
    gi = {g: i for i, g in enumerate(gl)}
    res = {}
    for variant in ("teacher", "student", "random"):
        zt = np.load(f"{EMB}/train_{variant}.npz")
        zv = np.load(f"{EMB}/val_{variant}.npz")
        mt, mv = meta(zt["ids"], man["train"]), meta(zv["ids"], man["val"])
        yg = np.array([gi[m["gloss"]] for m in mt])
        sg = np.array([m["signer"] for m in mt])
        yq = np.array([gi[m["gloss"]] for m in mv])
        res[variant] = {}
        for r in (READ if variant == "teacher" else ["grp_L4", "grp_fin", "ord4", "cls_fin"]):
            Xg, Xq = zt[r].astype(np.float32), zv[r].astype(np.float32)
            row = {"dim": int(Xg.shape[1])}
            row["R1_raw"] = metrics(class_max_ranks(Xq, yq, Xg, yg))["R1"]
            k = min(512, Xg.shape[1] - 1)
            mu, W = G.fit_pca_whiten(Xg, k, 1.0)
            Gw, Qw = G.apply_lin(Xg, mu, W), G.apply_lin(Xq, mu, W)
            row["R1_whiten"] = metrics(class_max_ranks(Qw, yq, Gw, yg))["R1"]
            if variant != "random" or r in ("grp_fin", "ord4"):
                try:
                    mu_d, Wd = G.fit_lda(Gw, yg, min(256, k - 1), 0.05)
                    row["R1_lda_ceiling"] = metrics(class_max_ranks(G.apply_lin(Qw, mu_d, Wd), yq, G.apply_lin(Gw, mu_d, Wd), yg))["R1"]
                except Exception as e:  # pragma: no cover
                    row["R1_lda_ceiling"] = None
            row["eta2_gloss"] = eta2(Xg, yg)
            row["eta2_signer"] = eta2(Xg, sg)
            row["nn_same_signer"], row["nn_same_gloss"] = nn_stats(Xg, yg, sg)
            res[variant][r] = row
            print(variant, r, {k_: (round(v, 2) if isinstance(v, float) else v) for k_, v in row.items()}, flush=True)
            json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
