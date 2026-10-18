"""R1-F6 fix: cluster-CORAL with a LABEL-FREE choice of the number of clusters, and mean over K instead of best-of-grid.

The v1.0 numbers used Kq = 6 = the true number of val signers.  Here Kq is chosen by the silhouette of k-means on the query set
(candidates 2..24) and the gallery K_g by silhouette (candidates 30, 40, 50, 70); we report the chosen K and the R@1, and also the mean/min/max
over a fixed candidate set so that no oracle K enters.  Three settings:
  (A) val queries vs full train gallery, clusters in the whitened `grp` embedding;
  (B) the 5 held-out-signer CV splits of ladder_cv.py (true #signers = 6 per split), clusters in the whitened `grp` embedding;
  (C) half-subset with CLS-cluster pseudo-signers (queries: val-half; gallery: train-half).
Clustering features: first 64 PCA-whitened dims (A, B) or the L2-normalised CLS (C).
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from analyze_emb import meta
from common import *  # noqa
import geo_headroom as G
import signer_norm as SN

QC = [2, 3, 4, 6, 8, 12, 16, 24]
GC = [30, 40, 50, 70]


def best_k(F, cand, rng, n=4000):
    idx = rng.choice(len(F), min(n, len(F)), replace=False)
    sc = {}
    for k in cand:
        lab = KMeans(k, n_init=2, random_state=0).fit(F[idx]).labels_
        sc[k] = float(silhouette_score(F[idx], lab))
    return max(sc, key=sc.get), sc


def run_setting(Xg, yg, Xq, yq, Fg, Fq, tag, rng):
    r1 = lambda Q, Gl: float(metrics(class_max_ranks(Q, yq, Gl, yg))["R1"])
    mu, W = G.fit_pca_whiten(Xg, 512, 1.0)
    res = {"raw": r1(Xq, Xg), "whiten": r1(G.apply_lin(Xq, mu, W), G.apply_lin(Xg, mu, W))}
    kq, sq_ = best_k(Fq, QC, rng)
    kg, sg_ = best_k(Fg, GC, rng)
    res["silhouette_query"] = sq_
    res["silhouette_gallery"] = sg_
    res["chosen_Kq"], res["chosen_Kg"] = kq, kg
    lab_g = {k: KMeans(k, n_init=2, random_state=0).fit(Fg).labels_ for k in GC}
    lab_q = {k: KMeans(k, n_init=3, random_state=0).fit(Fq).labels_ for k in QC}
    grid = {}
    for kg_ in GC:
        Gn = SN.coral(Xg, lab_g[kg_])
        m3, W3 = G.fit_pca_whiten(Gn, 512, 1.0)
        for kq_ in QC:
            Qn = SN.coral(Xq, lab_q[kq_])
            grid[f"Kg{kg_}_Kq{kq_}"] = r1(G.apply_lin(Qn, m3, W3), G.apply_lin(Gn, m3, W3))
    res["grid"] = grid
    res["R1_at_chosen_K"] = grid[f"Kg{kg}_Kq{kq}"]
    sel = [v for k, v in grid.items() if int(k.split("Kq")[1]) in (4, 6, 8, 12, 16)]
    res["mean_over_Kq_4..16_and_all_Kg"] = float(np.mean(sel))
    res["min_max_over_same"] = [float(np.min(sel)), float(np.max(sel))]
    print(tag, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items() if k not in ("grid", "silhouette_query", "silhouette_gallery")}, flush=True)
    return res


def main():
    rng = np.random.default_rng(0)
    d = load_baseline()
    tr, va = d["train"], d["val"]
    out = {}
    # (A) val vs full gallery
    mu, W = G.fit_pca_whiten(tr["X"], 512, 1.0)
    Zg, Zq = G.apply_lin(tr["X"], mu, W), G.apply_lin(va["X"], mu, W)
    out["A_val_grp_clusters"] = run_setting(tr["X"], tr["y"], va["X"], va["y"], Zg[:, :64], Zq[:, :64], "A", rng)
    # (C) half-subset with CLS clusters
    EMB = f"{SCRATCH}/emb"
    man = manifests()
    gl = sorted({m["gloss"] for sp in man.values() for m in sp})
    gi = {g: i for i, g in enumerate(gl)}
    zt, zv = np.load(f"{EMB}/train_teacher.npz"), np.load(f"{EMB}/val_teacher.npz")
    mt, mv = meta(zt["ids"], man["train"]), meta(zv["ids"], man["val"])
    yg = np.array([gi[m["gloss"]] for m in mt])
    yq = np.array([gi[m["gloss"]] for m in mv])
    out["C_half_CLS_clusters"] = run_setting(zt["grp_fin"].astype(np.float32), yg, zv["grp_fin"].astype(np.float32), yq,
                                             l2(zt["cls_fin"].astype(np.float32)), l2(zv["cls_fin"].astype(np.float32)), "C", rng)
    # (B) CV splits with held-out train signers
    X, y, s = tr["X"], tr["y"], tr["signer"]
    sg = np.unique(s)
    cv = []
    rng2 = np.random.default_rng(0)
    for split in range(5):
        hold = rng2.choice(sg, size=6, replace=False)
        mq = np.isin(s, hold)
        Xg_, yg_, Xq_, yq_ = X[~mq], y[~mq], X[mq], y[mq]
        ok = np.isin(yq_, np.unique(yg_))
        Xq_, yq_ = Xq_[ok], yq_[ok]
        if len(Xq_) > 6000:
            sub = rng2.choice(len(Xq_), 6000, replace=False)
            Xq_, yq_ = Xq_[sub], yq_[sub]
        mu_, W_ = G.fit_pca_whiten(Xg_, 512, 1.0)
        Zg_, Zq_ = G.apply_lin(Xg_, mu_, W_), G.apply_lin(Xq_, mu_, W_)
        cv.append(run_setting(Xg_, yg_, Xq_, yq_, Zg_[:, :64], Zq_[:, :64], f"B{split}", rng))
    out["B_cv"] = cv
    out["B_cv_summary"] = {k: float(np.mean([c[k] for c in cv])) for k in ("raw", "whiten", "R1_at_chosen_K", "mean_over_Kq_4..16_and_all_Kg")}
    print("CV summary", out["B_cv_summary"], flush=True)
    json.dump(out, open("results/kfree_cluster_corral.json", "w"), indent=1)


if __name__ == "__main__":
    main()
