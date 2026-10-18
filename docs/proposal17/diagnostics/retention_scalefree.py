"""R1-F1/F7 fix: scale-free retention statistics (trained vs random-init encoder) and the decode comparison with style variables.

The variance-weighted eta^2 of proposal v1.0 is dominated by a few directions of the very anisotropic random encoder (participation ratio
~8 vs ~84).  Here we report, for the half-subset train embeddings `grp_fin` of random / student / teacher:
  participation ratio of the total covariance; eta^2_signer and eta^2_gloss on raw and on PCA-512-WHITENED coordinates
  (+ the null eta^2_gloss from shuffled gloss labels); 35-way linear signer decode on STANDARDISED features (70/30 clip split);
  same-signer top-1 NN.  Also: signer decode from the 30 style variables vs from the standardised embedding, same split.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from analyze_emb import eta2, meta, nn_stats
from common import *  # noqa
import geo_headroom as G

EMB = f"{SCRATCH}/emb"


def pr(X):
    w = np.linalg.eigvalsh(np.cov((X - X.mean(0)).T))
    w = np.clip(w, 0, None)
    return float(w.sum() ** 2 / (w ** 2).sum())


def main():
    man = manifests()
    gl = sorted({m["gloss"] for sp in man.values() for m in sp})
    gi = {g: i for i, g in enumerate(gl)}
    rng = np.random.default_rng(0)
    out = {}
    for variant in ("random", "student", "teacher"):
        z = np.load(f"{EMB}/train_{variant}.npz")
        mt = meta(z["ids"], man["train"])
        yg = np.array([gi[m["gloss"]] for m in mt])
        sg = np.array([m["signer"] for m in mt])
        X = z["grp_fin"].astype(np.float32)
        ok = np.linalg.norm(X, axis=1) > 0  # 17 all-zero stride-16 vectors (review F15)
        X, yg, sg = X[ok], yg[ok], sg[ok]
        mu, W = G.fit_pca_whiten(X, 512, 1.0)
        Zw = G.apply_lin(X, mu, W)
        sh = rng.permutation(yg)
        r = {"n": int(len(X)), "participation_ratio": pr(X),
             "eta2_signer_raw": eta2(X, sg), "eta2_gloss_raw": eta2(X, yg), "eta2_gloss_raw_null_shuffled": eta2(X, sh),
             "eta2_signer_whitened": eta2(Zw, sg), "eta2_gloss_whitened": eta2(Zw, yg), "eta2_gloss_whitened_null": eta2(Zw, sh)}
        Zs = StandardScaler().fit_transform(X)
        perm = rng.permutation(len(Zs))
        cut = int(0.7 * len(Zs))
        a, b = perm[:cut], perm[cut:]
        clf = LogisticRegression(max_iter=400, C=1.0).fit(Zs[a], sg[a])
        r["signer_decode_linear_standardised"] = float((clf.predict(Zs[b]) == sg[b]).mean() * 100)
        r["nn_same_signer"], r["nn_same_gloss"] = nn_stats(X, yg, sg)
        out[variant] = r
        print(variant, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    json.dump(out, open("results/retention_scalefree.json", "w"), indent=1)


if __name__ == "__main__":
    main()
