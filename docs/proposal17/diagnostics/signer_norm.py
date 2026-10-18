"""D-SNORM: ceilings of signer-level normalisation and the feasibility of cross-signer nearest-neighbour positives
(proposal17 diagnostics G6, G8).  Embeddings = PLctx full160 `_grp` (train gallery, val queries), baseline checkpoint.

(a) Transductive oracle normalisations (they use the signer id of every clip in BOTH gallery and queries - upper bounds, not methods):
      per-signer mean (CMN), per-signer mean+variance (CMVN), per-signer covariance alignment (CORAL, ridge-shrunk), each followed
      optionally by the label-free PCA whitening.  Question: is the signer effect an additive offset, a per-dimension scale, or a
      signer-specific linear map?
(b) Cross-signer NN precision: for train clips, the nearest neighbour among clips of a DIFFERENT signer - how often is it the same
      gloss (raw / whitened / signer-erased / LDA ceiling)?  This is the quality of pseudo-positives that an NNCLR-style stage would
      mine; the literature gate (Q2 sec. 5) is >= 40 % for mutual pairs.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from common import *  # noqa
import geo_headroom as G

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/signer_norm.json"


def cmn(X, s):
    out = X.copy()
    for u in np.unique(s):
        m = s == u
        out[m] = X[m] - X[m].mean(0)
    return out


def cmvn(X, s, eps=1e-6):
    out = X.copy()
    for u in np.unique(s):
        m = s == u
        if m.sum() < 50:
            continue
        mu, sd = X[m].mean(0), X[m].std(0) + eps
        out[m] = (X[m] - mu) / sd
    return out


def coral(X, s, lam=0.2):
    d = X.shape[1]
    Xc = X - X.mean(0)
    Sg = np.cov(Xc.T)
    w, V = np.linalg.eigh(Sg + 1e-6 * np.eye(d))
    Sg_half = (V * np.sqrt(w)) @ V.T
    out = X.copy()
    for u in np.unique(s):
        m = s == u
        if m.sum() < 200:  # too few clips to estimate a covariance
            out[m] = X[m] - X[m].mean(0)
            continue
        Z = X[m] - X[m].mean(0)
        Ss = np.cov(Z.T)
        Ss = (1 - lam) * Ss + lam * np.trace(Ss) / d * np.eye(d)
        w2, V2 = np.linalg.eigh(Ss)
        Ss_ihalf = (V2 / np.sqrt(w2)) @ V2.T
        out[m] = Z @ Ss_ihalf @ Sg_half
    return out


def r1(Q, yq, Gl, yg):
    return metrics(class_max_ranks(Q, yq, Gl, yg))["R1"]


def cross_nn_precision(X, y, s, n=12000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    Xn = l2(X[idx])
    yy, ss = y[idx], s[idx]
    S = Xn @ Xn.T
    S[ss[:, None] == ss[None, :]] = -2
    nn = S.argmax(1)
    same = yy[nn] == yy
    mutual = nn[nn] == np.arange(len(idx))
    return dict(top1_precision=float(same.mean() * 100), mutual_fraction=float(mutual.mean() * 100),
                mutual_precision=float(same[mutual].mean() * 100) if mutual.any() else float("nan"))


def main():
    d = load_baseline()
    tr, va = d["train"], d["val"]
    out = {}
    Xg, Xq = tr["X"], va["X"]
    out["raw"] = r1(Xq, va["y"], Xg, tr["y"])
    # whiten fit (label-free) as in geo_headroom
    mu_w, W = G.fit_pca_whiten(Xg, 512, 1.0)
    out["whiten"] = r1(G.apply_lin(Xq, mu_w, W), va["y"], G.apply_lin(Xg, mu_w, W), tr["y"])
    for name, f in (("CMN", cmn), ("CMVN", cmvn), ("CORAL", coral)):
        Gn, Qn = f(Xg, tr["signer"]), f(Xq, va["signer"])
        out[name] = r1(Qn, va["y"], Gn, tr["y"])
        mu2, W2 = G.fit_pca_whiten(Gn, 512, 1.0)
        out[name + "+whiten"] = r1(G.apply_lin(Qn, mu2, W2), va["y"], G.apply_lin(Gn, mu2, W2), tr["y"])
        print(name, out[name], out[name + "+whiten"], flush=True)
    print("raw", out["raw"], "whiten", out["whiten"], flush=True)

    # cross-signer NN precision under several geometries (train clips only)
    prec = {}
    prec["raw"] = cross_nn_precision(Xg, tr["y"], tr["signer"])
    prec["whitened"] = cross_nn_precision(G.apply_lin(Xg, mu_w, W), tr["y"], tr["signer"])
    sg = np.unique(tr["signer"])
    sidx = {u: i for i, u in enumerate(sg)}
    z = np.array([sidx[u] for u in tr["signer"]])
    mu_l, M = G.fit_leace(Xg, z)
    Xl = mu_l + (Xg - mu_l) @ M
    mu3, W3 = G.fit_pca_whiten(Xl, 512, 1.0)
    prec["leace+whiten"] = cross_nn_precision(G.apply_lin(Xl, mu3, W3), tr["y"], tr["signer"])
    prec["CORAL+whiten"] = cross_nn_precision(G.apply_lin(coral(Xg, tr["signer"]), *G.fit_pca_whiten(coral(Xg, tr["signer"]), 512, 1.0)),
                                              tr["y"], tr["signer"])
    mu_d, Wd = G.fit_lda(Xg, tr["y"], 256, 0.05)
    prec["lda_ceiling_(train-fit, optimistic)"] = cross_nn_precision(G.apply_lin(Xg, mu_d, Wd), tr["y"], tr["signer"])
    out["cross_signer_nn_precision"] = prec
    for k, v in prec.items():
        print("cross-signer NN", k, v, flush=True)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
