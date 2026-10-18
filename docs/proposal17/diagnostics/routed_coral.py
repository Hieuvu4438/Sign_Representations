"""D-ROUTED: INDUCTIVE, label-free cluster normalisation routed by the CLS code (proposal17 diagnostic G21).

Train-half (gallery) clips are clustered by their CLS (K clusters); for each cluster the CORAL-style statistics (mean, shrunk covariance ->
colour-to-global transform) are learned on the gallery.  A val QUERY is routed to the nearest cluster by its OWN CLS only (no query-batch
statistics, no label) and receives that cluster's transform; gallery clips use their own cluster's transform; then label-free PCA-whitening
(fit on the transformed gallery).  Controls: random routing of queries; transductive cluster-CORAL (query clusters from the query batch) for scale.
Half-subset: whiten 24.94; transductive CLS-CORAL ~31.6-32.3; true-signer CORAL 31.7.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.cluster import KMeans

from analyze_emb import meta
from common import *  # noqa
import geo_headroom as G
import signer_norm as SN


def fit_cluster_transforms(X, labels, lam=0.2, min_n=200):
    d = X.shape[1]
    Xc = X - X.mean(0)
    w, V = np.linalg.eigh(np.cov(Xc.T) + 1e-6 * np.eye(d))
    Sg_half = (V * np.sqrt(w)) @ V.T
    tr = {}
    for c in np.unique(labels):
        m = labels == c
        mu = X[m].mean(0)
        if m.sum() < min_n:
            tr[c] = (mu, None)
            continue
        Ss = np.cov((X[m] - mu).T)
        Ss = (1 - lam) * Ss + lam * np.trace(Ss) / d * np.eye(d)
        w2, V2 = np.linalg.eigh(Ss)
        tr[c] = (mu, ((V2 / np.sqrt(w2)) @ V2.T) @ Sg_half)
    return tr


def apply(X, labels, tr):
    out = X.copy()
    for c in np.unique(labels):
        m = labels == c
        mu, T = tr[c]
        out[m] = (X[m] - mu) if T is None else (X[m] - mu) @ T
    return out


def main():
    EMB = f"{SCRATCH}/emb"
    man = manifests()
    gl = sorted({m["gloss"] for sp in man.values() for m in sp})
    gi = {g: i for i, g in enumerate(gl)}
    zt, zv = np.load(f"{EMB}/train_teacher.npz"), np.load(f"{EMB}/val_teacher.npz")
    mt, mv = meta(zt["ids"], man["train"]), meta(zv["ids"], man["val"])
    yg = np.array([gi[m["gloss"]] for m in mt])
    yq = np.array([gi[m["gloss"]] for m in mv])
    Xg, Xq = zt["grp_fin"].astype(np.float32), zv["grp_fin"].astype(np.float32)
    Cg, Cq = l2(zt["cls_fin"].astype(np.float32)), l2(zv["cls_fin"].astype(np.float32))
    r1 = lambda Q, Gl: float(metrics(class_max_ranks(Q, yq, Gl, yg))["R1"])
    mu, W = G.fit_pca_whiten(Xg, 512, 1.0)
    out = {"raw": r1(Xq, Xg), "whiten": r1(G.apply_lin(Xq, mu, W), G.apply_lin(Xg, mu, W))}
    print(out, flush=True)
    rng = np.random.default_rng(0)
    for K in (8, 16, 35, 48, 70):
        km = KMeans(K, n_init=3, random_state=0).fit(Cg)
        lg = km.labels_
        tr = fit_cluster_transforms(Xg, lg)
        Gn = apply(Xg, lg, tr)
        m3, W3 = G.fit_pca_whiten(Gn, 512, 1.0)
        route = np.argmax(Cq @ km.cluster_centers_.T / np.linalg.norm(km.cluster_centers_, axis=1), 1)
        Qn = apply(Xq, route, tr)
        out[f"K{K}_routed_inductive"] = r1(G.apply_lin(Qn, m3, W3), G.apply_lin(Gn, m3, W3))
        Qr = apply(Xq, rng.integers(0, K, len(Xq)), tr)
        out[f"K{K}_random_routing_control"] = r1(G.apply_lin(Qr, m3, W3), G.apply_lin(Gn, m3, W3))
        lq = KMeans(min(K, 12), n_init=3, random_state=0).fit(Cq).labels_
        Qt = SN.coral(Xq, lq)
        out[f"K{K}_transductive_Kq{min(K, 12)}"] = r1(G.apply_lin(Qt, m3, W3), G.apply_lin(Gn, m3, W3))
        print(K, {k: round(v, 2) for k, v in out.items() if k.startswith(f"K{K}_")}, flush=True)
        json.dump(out, open("results/routed_coral.json", "w"), indent=1)


if __name__ == "__main__":
    main()
