"""D-GEO: information-vs-geometry decomposition of the PLctx full160 frozen embedding (proposal17 diagnostic G1).

Question: the frozen dictionary retrieval is R@1 22.7 (val) while supervised finetuning of a skeleton encoder reaches ~55 top-1.
Is the missing performance (a) information that is not in the embedding, or (b) geometry: signer offset / anisotropy that a
better objective or a linear map could remove?  We apply, to the SAME 768-d embedding, ladders of linear maps that are fit on the
TRAIN split only (never val) and score VAL queries against the TRAIN gallery with the official class-max cosine protocol:

  raw -> centred -> PCA-whitened (label-free) -> signer-offset oracle (transductive, uses signer ids)
      -> LEACE signer erasure (train signer ids, weak supervision) -> LDA on gloss (supervised linear upper bound)

Hyper-parameters (whitening dims, shrinkage) are chosen on a signer-disjoint hold-out INSIDE train, so val is only used to report.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from common import *  # noqa

rng = np.random.default_rng(0)
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/geo_headroom.json"


def fit_center(X):
    return X.mean(0)


def fit_pca_whiten(X, k, power=1.0, ridge=1e-3):
    mu = X.mean(0)
    C = np.cov((X - mu).T)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1][:k]
    w, V = w[idx], V[:, idx]
    scale = (w + ridge * w.max()) ** (-0.5 * power)
    return mu, V * scale


def apply_lin(X, mu, W):
    return (X - mu) @ W


def fit_leace(X, z, ridge=1e-4):
    """LEACE (Belrose et al. 2023) linear concept erasure for class labels z. Returns (mu, P) with x' = mu + (x-mu) P."""
    mu = X.mean(0)
    Xc = X - mu
    C = np.cov(Xc.T) + ridge * np.eye(X.shape[1]) * np.trace(np.cov(Xc.T)) / X.shape[1]
    w, V = np.linalg.eigh(C)
    Wm = V @ np.diag(w ** -0.5) @ V.T
    Wp = V @ np.diag(w ** 0.5) @ V.T
    Z = np.eye(z.max() + 1)[z]
    Z = Z - Z.mean(0)
    cross = (Xc.T @ Z) / len(X)  # (d, K)
    A = Wm @ cross
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    keep = s > 1e-8 * s.max()
    U = U[:, keep]
    Pproj = U @ U.T  # projection onto whitened cross-cov column space
    # x' = x - Wp Pproj Wm (x - mu)  -> as right-multiplication on row vectors
    M = np.eye(X.shape[1]) - (Wm.T @ Pproj.T @ Wp.T)
    return mu, M


def fit_lda(X, y, k, shrink):
    mu = X.mean(0)
    classes = np.unique(y)
    cm = np.stack([X[y == c].mean(0) for c in classes])
    Xc = X - cm[np.searchsorted(classes, y)]
    Sw = np.cov(Xc.T)
    Sw = Sw + shrink * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
    Sb = np.cov(cm.T)
    w, V = np.linalg.eigh(Sw)
    Wm = V @ np.diag(w ** -0.5) @ V.T
    B = Wm @ Sb @ Wm
    wb, Vb = np.linalg.eigh(B)
    idx = np.argsort(wb)[::-1][:k]
    return mu, Wm @ Vb[:, idx]


def per_signer_center(X, signer):
    out = X.copy()
    for s in np.unique(signer):
        m = signer == s
        out[m] = X[m] - X[m].mean(0)
    return out


def r1(Q, yq, G, yg):
    return metrics(class_max_ranks(Q, yq, G, yg))["R1"]


def main():
    d = load_baseline()
    tr, va = d["train"], d["val"]
    res = {"n_train": int(len(tr["y"])), "n_val": int(len(va["y"])), "train_signers": int(len(np.unique(tr["signer"])))}
    t0 = time.time()
    base = metrics(class_max_ranks(va["X"], va["y"], tr["X"], tr["y"]))
    res["raw"] = base
    print("raw", base, flush=True)

    # signer-disjoint hold-out inside train for hyper-parameter selection
    sg = np.unique(tr["signer"])
    hold = set(rng.choice(sg, size=6, replace=False).tolist())
    mh = np.array([s in hold for s in tr["signer"]])
    Xa, ya, sa = tr["X"][~mh], tr["y"][~mh], tr["signer"][~mh]
    Xh, yh = tr["X"][mh], tr["y"][mh]
    # (hold-out queries only count glosses present in the inner gallery)
    keepq = np.isin(yh, np.unique(ya))
    Xh, yh = Xh[keepq], yh[keepq]
    sub = rng.choice(len(Xh), size=min(3000, len(Xh)), replace=False)
    Xh, yh = Xh[sub], yh[sub]

    # --- centred
    mu = fit_center(tr["X"])
    res["centered"] = metrics(class_max_ranks(va["X"] - mu, va["y"], tr["X"] - mu, tr["y"]))
    print("centered", res["centered"], flush=True)

    # --- PCA whitening: choose k, power on inner hold-out
    best = None
    grid = []
    for k in (64, 128, 256, 512):
        for power in (0.5, 1.0):
            mu_a, W = fit_pca_whiten(Xa, k, power)
            sc = r1(apply_lin(Xh, mu_a, W), yh, apply_lin(Xa, mu_a, W), ya)
            grid.append(dict(k=k, power=power, inner_R1=sc))
            if best is None or sc > best[0]:
                best = (sc, k, power)
    print("whiten grid", grid, "best", best, flush=True)
    res["whiten_grid"] = grid
    mu_w, W = fit_pca_whiten(tr["X"], best[1], best[2])
    res["whitened"] = dict(**metrics(class_max_ranks(apply_lin(va["X"], mu_w, W), va["y"], apply_lin(tr["X"], mu_w, W), tr["y"])),
                           k=best[1], power=best[2])
    print("whitened", res["whitened"], flush=True)

    # --- signer-offset oracle (transductive, uses signer ids of both sides)
    Xg_c = per_signer_center(tr["X"], tr["signer"])
    Xq_c = per_signer_center(va["X"], va["signer"])
    res["signer_offset_oracle"] = metrics(class_max_ranks(Xq_c, va["y"], Xg_c, tr["y"]))
    print("signer offset oracle", res["signer_offset_oracle"], flush=True)
    # same oracle followed by whitening fit on the signer-centred train
    mu2, W2 = fit_pca_whiten(Xg_c, best[1], best[2])
    res["signer_offset_oracle+whiten"] = metrics(class_max_ranks(apply_lin(Xq_c, mu2, W2), va["y"], apply_lin(Xg_c, mu2, W2), tr["y"]))
    print("signer offset oracle + whiten", res["signer_offset_oracle+whiten"], flush=True)

    # --- LEACE signer erasure (fit on train signer ids)
    sidx = {s: i for i, s in enumerate(sg)}
    z = np.array([sidx[s] for s in tr["signer"]])
    mu_l, M = fit_leace(tr["X"], z)
    Xg_l = mu_l + (tr["X"] - mu_l) @ M
    Xq_l = mu_l + (va["X"] - mu_l) @ M
    res["leace_signer"] = metrics(class_max_ranks(Xq_l, va["y"], Xg_l, tr["y"]))
    print("leace signer", res["leace_signer"], flush=True)
    mu3, W3 = fit_pca_whiten(Xg_l, best[1], best[2])
    res["leace_signer+whiten"] = metrics(class_max_ranks(apply_lin(Xq_l, mu3, W3), va["y"], apply_lin(Xg_l, mu3, W3), tr["y"]))
    print("leace + whiten", res["leace_signer+whiten"], flush=True)

    # --- LDA on gloss (supervised linear upper bound); choose k and shrinkage on the inner hold-out
    bestl, gridl = None, []
    for k in (256, 512, 767):
        for shrink in (0.05, 0.3, 1.0):
            mu_a, Wl = fit_lda(Xa, ya, k, shrink)
            sc = r1(apply_lin(Xh, mu_a, Wl), yh, apply_lin(Xa, mu_a, Wl), ya)
            gridl.append(dict(k=k, shrink=shrink, inner_R1=sc))
            if bestl is None or sc > bestl[0]:
                bestl = (sc, k, shrink)
    print("lda grid", gridl, "best", bestl, flush=True)
    res["lda_grid"] = gridl
    mu_d, Wd = fit_lda(tr["X"], tr["y"], bestl[1], bestl[2])
    res["lda_gloss_supervised"] = dict(**metrics(class_max_ranks(apply_lin(va["X"], mu_d, Wd), va["y"], apply_lin(tr["X"], mu_d, Wd), tr["y"])),
                                       k=bestl[1], shrink=bestl[2])
    print("lda", res["lda_gloss_supervised"], flush=True)

    # --- gallery-composition scaling (raw embedding): number of gallery signers, clips per gloss
    sc_sign = {}
    for ns in (4, 8, 16, 24, 35):
        vals = []
        for rep in range(3 if ns < 35 else 1):
            chosen = set(rng.choice(sg, size=ns, replace=False).tolist())
            m = np.array([s in chosen for s in tr["signer"]])
            vals.append(r1(va["X"], va["y"], tr["X"][m], tr["y"][m]))
        sc_sign[ns] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals)))
        print("gallery signers", ns, sc_sign[ns], flush=True)
    res["gallery_signers_curve"] = sc_sign
    sc_clips = {}
    for kpg in (1, 2, 4, 8, 16):
        vals = []
        for rep in range(3):
            keep = []
            for c in np.unique(tr["y"]):
                ii = np.where(tr["y"] == c)[0]
                keep.extend(rng.choice(ii, size=min(kpg, len(ii)), replace=False).tolist())
            keep = np.array(keep)
            vals.append(r1(va["X"], va["y"], tr["X"][keep], tr["y"][keep]))
        sc_clips[kpg] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals)))
        print("clips/gloss", kpg, sc_clips[kpg], flush=True)
    res["gallery_clips_per_gloss_curve"] = sc_clips

    res["elapsed_s"] = time.time() - t0
    json.dump(res, open(OUT, "w"), indent=1)
    print("done", res["elapsed_s"])


if __name__ == "__main__":
    main()
