"""D-PSIGN: can a LABEL-FREE pseudo-signer (clusters of morphology/posture style variables) stand in for the signer id in the
normalisations that helped?  (proposal17 diagnostic G9)

Style variables (style_vars.py; MORPH + POSTURE only: proportions, head height/offset, rest height; no resolution, fps, duration or
observation fractions, which are recording-protocol leaks) are standardised, clustered by k-means on TRAIN clips, and every clip is
assigned to a pseudo-signer = nearest centroid.  We then compare, on val R@1 against the train gallery (class-max cosine):
   true-signer CORAL (oracle, 38.5 with whitening)  vs  pseudo-signer CORAL, in two modes
   transductive : cluster statistics estimated from the clips of that cluster in the same split (as the oracle does)
   inductive    : each val clip uses the transform of the TRAIN cluster it is assigned to (no val statistics at all)
and report how well the pseudo clusters recover true signers (adjusted mutual information, purity).
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from common import *  # noqa
import geo_headroom as G
import signer_norm as SN
import style_vars as SV

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/pseudo_signer.json"
USE = SV.GROUPS["MORPH"] + SV.GROUPS["POSTURE"]


def table(split, d):
    ids, rows = SV.build(split, 1)
    names = [n for g in SV.GROUPS.values() for n in g]
    F = np.array([[r.get(n, np.nan) for n in names] for r in rows], dtype=np.float64)
    pos = {i: k for k, i in enumerate(d[split]["ids"])}
    keep = [k for k, i in enumerate(ids) if i in pos and np.isfinite(F[k, [names.index(u) for u in USE]]).all()]
    cols = [names.index(u) for u in USE]
    return np.array([pos[ids[k]] for k in keep]), F[keep][:, cols]


def coral_with(Xtarget_groups_stats, X, labels, stats, lam=0.2):
    """apply per-cluster CORAL where stats[c] = (mu, Sigma_ihalf) come from `stats`; clusters absent in stats -> mean-centre only."""
    out = X.copy()
    for c in np.unique(labels):
        m = labels == c
        if c in stats:
            mu, T = stats[c]
            out[m] = (X[m] - mu) @ T
        else:
            out[m] = X[m] - X[m].mean(0)
    return out


def fit_stats(X, labels, lam=0.2, min_n=200):
    d = X.shape[1]
    Xc = X - X.mean(0)
    w, V = np.linalg.eigh(np.cov(Xc.T) + 1e-6 * np.eye(d))
    Sg_half = (V * np.sqrt(w)) @ V.T
    st = {}
    for c in np.unique(labels):
        m = labels == c
        if m.sum() < min_n:
            continue
        mu = X[m].mean(0)
        Ss = np.cov((X[m] - mu).T)
        Ss = (1 - lam) * Ss + lam * np.trace(Ss) / d * np.eye(d)
        w2, V2 = np.linalg.eigh(Ss)
        st[c] = (mu, ((V2 / np.sqrt(w2)) @ V2.T) @ Sg_half)
    return st


def main():
    d = load_baseline()
    tr, va = d["train"], d["val"]
    ti, Ft = table("train", d)
    vi, Fv = table("val", d)
    mu, sd = Ft.mean(0), Ft.std(0) + 1e-9
    Zt, Zv = (Ft - mu) / sd, (Fv - mu) / sd
    out = {"n_train_with_style": int(len(ti)), "n_val_with_style": int(len(vi))}
    # restrict gallery/queries to clips with style variables (a subset); compute reference numbers on the SAME subsets
    Xg, yg, sg = tr["X"][ti], tr["y"][ti], tr["signer"][ti]
    Xq, yq, sq = va["X"][vi], va["y"][vi], va["signer"][vi]
    mu_w, W = G.fit_pca_whiten(Xg, 512, 1.0)
    wh = lambda A, B: (G.apply_lin(A, mu_w, W), G.apply_lin(B, mu_w, W))
    r1 = lambda Q, Gl: float(metrics(class_max_ranks(Q, yq, Gl, yg))["R1"])
    out["subset_raw"] = r1(Xq, Xg)
    a, b = wh(Xq, Xg)
    out["subset_whiten"] = r1(a, b)
    Gc, Qc = SN.coral(Xg, sg), SN.coral(Xq, sq)
    m2, W2 = G.fit_pca_whiten(Gc, 512, 1.0)
    out["subset_true_signer_CORAL+whiten"] = r1(G.apply_lin(Qc, m2, W2), G.apply_lin(Gc, m2, W2))
    print(out, flush=True)
    for K in (35, 70, 140):
        km = KMeans(K, n_init=4, random_state=0).fit(Zt)
        lt, lv = km.labels_, km.predict(Zv)
        # recovery of true signers
        out[f"K{K}_AMI_train"] = float(adjusted_mutual_info_score(sg, lt))
        out[f"K{K}_AMI_val"] = float(adjusted_mutual_info_score(sq, lv))
        # transductive (cluster stats from same split)
        Gp, Qp = SN.coral(Xg, lt), SN.coral(Xq, lv)
        m3, W3 = G.fit_pca_whiten(Gp, 512, 1.0)
        out[f"K{K}_pseudoCORAL_transductive+whiten"] = r1(G.apply_lin(Qp, m3, W3), G.apply_lin(Gp, m3, W3))
        # inductive: val clips use train-cluster transform
        st = fit_stats(Xg, lt)
        Gi = coral_with(None, Xg, lt, st)
        Qi = coral_with(None, Xq, lv, st)
        m4, W4 = G.fit_pca_whiten(Gi, 512, 1.0)
        out[f"K{K}_pseudoCORAL_inductive+whiten"] = r1(G.apply_lin(Qi, m4, W4), G.apply_lin(Gi, m4, W4))
        print({k: round(v, 3) for k, v in out.items() if k.startswith(f"K{K}")}, flush=True)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
