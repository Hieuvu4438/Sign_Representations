"""R1-F5/F13 fixes: composition control, per-signer whitening rung, held-out-gloss LDA ceiling with an extended k grid.

(a) Which operation is the effective one?  per-signer WHITENING without re-colouring (X_s -> (X_s - mu_s) Sigma_s^{-1/2}, ridge-shrunk)
    vs CORAL vs CMN, on the full gallery AND on a gallery restricted to the heavy signers (gloss-coverage control).
(b) LDA ceiling that does not memorise the evaluated glosses: partition glosses 70/30 (seed 0 = the probe partition, 819 held-out glosses);
    LDA is fit on train clips of the 70 % glosses only and used on held-out glosses (gallery = train clips of held-out glosses,
    queries = val clips of held-out glosses); also in-sample LDA on the held-out glosses and a k grid {32, 48, 64, 96, 128}.
"""
from __future__ import annotations

import json

import numpy as np

from common import *  # noqa
import geo_headroom as G
import signer_norm as SN


def per_signer_whiten(X, s, lam=0.2, min_n=200):
    d = X.shape[1]
    out = X.copy()
    for u in np.unique(s):
        m = s == u
        if m.sum() < min_n:
            out[m] = X[m] - X[m].mean(0)
            continue
        Z = X[m] - X[m].mean(0)
        C = np.cov(Z.T)
        C = (1 - lam) * C + lam * np.trace(C) / d * np.eye(d)
        w, V = np.linalg.eigh(C)
        out[m] = Z @ (V / np.sqrt(w)) @ V.T
    return out


def main():
    d = load_baseline()
    tr, va = d["train"], d["val"]
    Xg, yg, sg = tr["X"], tr["y"], tr["signer"]
    Xq, yq, sq = va["X"], va["y"], va["signer"]
    out = {}
    r1 = lambda Q, Gl, ygal=yg, yqry=yq: float(metrics(class_max_ranks(Q, yqry, Gl, ygal))["R1"])

    def rungs(Xg_, yg_, sg_, tag):
        res = {}
        res["raw"] = float(metrics(class_max_ranks(Xq, yq, Xg_, yg_))["R1"])
        mu, W = G.fit_pca_whiten(Xg_, 512, 1.0)
        res["whiten"] = float(metrics(class_max_ranks(G.apply_lin(Xq, mu, W), yq, G.apply_lin(Xg_, mu, W), yg_))["R1"])
        for name, f in (("CMN", SN.cmn), ("CORAL", SN.coral), ("per-signer-whiten", per_signer_whiten)):
            Gn, Qn = f(Xg_, sg_), f(Xq, sq)
            m2, W2 = G.fit_pca_whiten(Gn, 512, 1.0)
            res[name + "+whiten"] = float(metrics(class_max_ranks(G.apply_lin(Qn, m2, W2), yq, G.apply_lin(Gn, m2, W2), yg_))["R1"])
            res[name] = float(metrics(class_max_ranks(Qn, yq, Gn, yg_))["R1"])
        print(tag, {k: round(v, 2) for k, v in res.items()}, flush=True)
        return res

    # (a) composition
    out["full_gallery"] = rungs(Xg, yg, sg, "full")
    nglo = {u: len(np.unique(yg[sg == u])) for u in np.unique(sg)}
    ncl = {u: int((sg == u).sum()) for u in np.unique(sg)}
    heavy = [u for u in nglo if nglo[u] >= 1500]
    m = np.isin(sg, heavy)
    out["heavy_signers"] = {"signers": heavy, "clips": int(m.sum()), "share_of_train_clips": float(m.mean()),
                            "distinct_glosses": {u: nglo[u] for u in heavy}}
    out["heavy_gallery"] = rungs(Xg[m], yg[m], sg[m], "heavy")
    out["train_signer_clip_counts"] = ncl

    # (b) held-out-gloss LDA ceilings
    glosses = np.unique(np.concatenate([tr["gloss"], va["gloss"]]))
    rng = np.random.default_rng(0)
    test_g = set(rng.choice(glosses, size=int(0.3 * len(glosses)), replace=False).tolist())
    gt = np.array([g in test_g for g in tr["gloss"]])
    gv = np.array([g in test_g for g in va["gloss"]])
    out["heldout_gloss"] = {"n_test_glosses": len(test_g), "gallery_clips": int(gt.sum()), "query_clips": int(gv.sum())}
    Gt, yGt = Xg[gt], yg[gt]
    Qt, yQt = Xq[gv], yq[gv]
    Fit, yFit = Xg[~gt], yg[~gt]
    h = out["heldout_gloss"]
    h["raw"] = float(metrics(class_max_ranks(Qt, yQt, Gt, yGt))["R1"])
    mu, W = G.fit_pca_whiten(Fit, 512, 1.0)  # label-free whitening fit on the OTHER glosses' clips
    h["whiten_fit_on_other_glosses"] = float(metrics(class_max_ranks(G.apply_lin(Qt, mu, W), yQt, G.apply_lin(Gt, mu, W), yGt))["R1"])
    for k in (32, 48, 64, 96, 128):
        m_, Wl = G.fit_lda(Fit, yFit, k, 0.05)
        h[f"lda_fit_on_disjoint_glosses_k{k}"] = float(metrics(class_max_ranks(G.apply_lin(Qt, m_, Wl), yQt, G.apply_lin(Gt, m_, Wl), yGt))["R1"])
        m_, Wl = G.fit_lda(Gt, yGt, k, 0.05)
        h[f"lda_insample_k{k}"] = float(metrics(class_max_ranks(G.apply_lin(Qt, m_, Wl), yQt, G.apply_lin(Gt, m_, Wl), yGt))["R1"])
    print("heldout", {k: round(v, 2) for k, v in h.items() if isinstance(v, float)}, flush=True)
    # full-val LDA ceiling with an extended k grid (k < 96 was not scanned in geo_ladder.py)
    out["full_val_lda_k_grid"] = {}
    for k in (32, 48, 64, 96):
        m_, Wl = G.fit_lda(Xg, yg, k, 0.05)
        out["full_val_lda_k_grid"][k] = float(metrics(class_max_ranks(G.apply_lin(Xq, m_, Wl), yq, G.apply_lin(Xg, m_, Wl), yg))["R1"])
    print("full-val LDA k grid", out["full_val_lda_k_grid"], flush=True)
    json.dump(out, open("results/ladder_extra.json", "w"), indent=1)


if __name__ == "__main__":
    main()
