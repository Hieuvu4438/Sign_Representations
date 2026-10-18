"""R1-F2 fix: cross-signer / cross-cluster nearest-neighbour precision on the FULL 40,154 pool, and an NN-positive head with positives mined in
the aligned (CORAL) geometry.  Replaces the 12k-pool numbers of signer_norm.py (which understated precision; the real gallery has ~14.7
clips per gloss, the 12k pool ~4.4).

Precision definition (H10 in the proposal v1.1): for a train clip, the nearest neighbour among clips of a DIFFERENT group (group = true
signer, or label-free CLS cluster), computed in the geometry stated; a pair is 'mutual' if the neighbour's own cross-group NN is the clip.
Reported: top-1 precision (same gloss), mutual fraction, mutual precision.
Part 1 (full 40k pool, true-signer exclusion): raw, whiten, CMN+whiten, CORAL+whiten, LEACE+whiten, LDA (train-fit, optimistic).
Part 2 (half-subset pool 20k, because CLS is only stored there): exclusion by CLS cluster (label-free) vs true signer, in raw / whiten / CLS-CORAL geometry.
Part 3: NN-contrastive head on inductive whitened inputs (as nn_head.py) but with positives mined in the CORAL geometry.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.cluster import KMeans

from analyze_emb import meta
from common import *  # noqa
import geo_headroom as G
import nn_head as NH
import signer_norm as SN

torch.set_num_threads(4)


def cross_group_nn(Z, groups, bs=2048):
    Zt = torch.as_tensor(l2(Z).astype(np.float32))
    gt = torch.as_tensor(groups)
    nn_ = np.zeros(len(Z), np.int64)
    for a in range(0, len(Z), bs):
        s = Zt[a : a + bs] @ Zt.T
        s[gt[a : a + bs, None] == gt[None, :]] = -2
        nn_[a : a + bs] = s.argmax(1).numpy()
    return nn_


def prec(nn_, y):
    same = y[nn_] == y
    mut = nn_[nn_] == np.arange(len(y))
    return {"top1_precision": float(same.mean() * 100), "mutual_fraction": float(mut.mean() * 100),
            "mutual_precision": float(same[mut].mean() * 100) if mut.any() else None}


def main():
    d = load_baseline()
    tr, va = d["train"], d["val"]
    Xg, yg, sg = tr["X"], tr["y"], tr["signer"]
    sidx = {u: i for i, u in enumerate(np.unique(sg))}
    strue = np.array([sidx[u] for u in sg])
    out = {"pool": int(len(Xg))}
    mu, W = G.fit_pca_whiten(Xg, 512, 1.0)
    geos = {"raw": Xg, "whiten": G.apply_lin(Xg, mu, W)}
    Gm = SN.cmn(Xg, sg)
    m1, W1 = G.fit_pca_whiten(Gm, 512, 1.0)
    geos["CMN+whiten"] = G.apply_lin(Gm, m1, W1)
    Gc = SN.coral(Xg, sg)
    m2, W2 = G.fit_pca_whiten(Gc, 512, 1.0)
    geos["CORAL+whiten"] = G.apply_lin(Gc, m2, W2)
    ml, M = G.fit_leace(Xg, strue)
    Gl = ml + (Xg - ml) @ M
    m4, W4 = G.fit_pca_whiten(Gl, 512, 1.0)
    geos["LEACE+whiten"] = G.apply_lin(Gl, m4, W4)
    mu_d, Wd = G.fit_lda(Xg, yg, 96, 0.05)
    geos["LDA96 (train-fit, optimistic)"] = G.apply_lin(Xg, mu_d, Wd)
    out["full_pool_true_signer_exclusion"] = {}
    nns = {}
    for name, Z in geos.items():
        nns[name] = cross_group_nn(Z, strue)
        out["full_pool_true_signer_exclusion"][name] = prec(nns[name], yg)
        print("full", name, out["full_pool_true_signer_exclusion"][name], flush=True)

    # Part 2: half pool with CLS clusters
    EMB = f"{SCRATCH}/emb"
    man = manifests()
    gl = sorted({m["gloss"] for sp in man.values() for m in sp})
    gi = {g: i for i, g in enumerate(gl)}
    zt = np.load(f"{EMB}/train_teacher.npz")
    mt = meta(zt["ids"], man["train"])
    yh = np.array([gi[m["gloss"]] for m in mt])
    sh = np.array([m["signer"] for m in mt])
    Xh = zt["grp_fin"].astype(np.float32)
    Ch = l2(zt["cls_fin"].astype(np.float32))
    ok = np.linalg.norm(Xh, axis=1) > 0
    Xh, yh, sh, Ch = Xh[ok], yh[ok], sh[ok], Ch[ok]
    sidx2 = {u: i for i, u in enumerate(np.unique(sh))}
    sh_i = np.array([sidx2[u] for u in sh])
    cl48 = KMeans(48, n_init=3, random_state=0).fit(Ch).labels_
    muh, Wh = G.fit_pca_whiten(Xh, 512, 1.0)
    Gcls = SN.coral(Xh, cl48)
    mc, Wc = G.fit_pca_whiten(Gcls, 512, 1.0)
    Gtrue = SN.coral(Xh, sh)
    mt_, Wt_ = G.fit_pca_whiten(Gtrue, 512, 1.0)
    half_geos = {"raw": Xh, "whiten": G.apply_lin(Xh, muh, Wh), "CORAL(CLS48)+whiten": G.apply_lin(Gcls, mc, Wc),
                 "CORAL(true signer)+whiten": G.apply_lin(Gtrue, mt_, Wt_)}
    out["half_pool"] = {"pool": int(len(Xh)), "exclusion_by_CLS48_cluster": {}, "exclusion_by_true_signer": {}}
    for name, Z in half_geos.items():
        out["half_pool"]["exclusion_by_CLS48_cluster"][name] = prec(cross_group_nn(Z, cl48), yh)
        out["half_pool"]["exclusion_by_true_signer"][name] = prec(cross_group_nn(Z, sh_i), yh)
        print("half", name, out["half_pool"]["exclusion_by_CLS48_cluster"][name], out["half_pool"]["exclusion_by_true_signer"][name], flush=True)

    # Part 3: NN-head with positives mined in the CORAL geometry (inductive whitened inputs)
    Zg = geos["whiten"].astype(np.float32)
    Zq = G.apply_lin(va["X"], mu, W).astype(np.float32)
    r1 = lambda A, B: float(metrics(class_max_ranks(A, va["y"], B, yg))["R1"])
    out["head"] = {"whiten_reference": r1(Zq, Zg)}
    for name in ("CORAL+whiten", "whiten"):
        nn_ = nns[name]
        mut = nn_[nn_] == np.arange(len(nn_))
        for mutual_only in (False, True):
            keep = mut if mutual_only else np.ones(len(Zg), bool)
            head = NH.train_head(Zg, nn_, keep)
            key = f"positives_from_{name}{'_mutual' if mutual_only else ''}"
            out["head"][key] = {"mined_precision": float((yg[nn_][keep] == yg[keep]).mean() * 100), "kept": float(keep.mean() * 100),
                                "val_R1": r1(NH.embed(head, Zq), NH.embed(head, Zg))}
            print(key, out["head"][key], flush=True)
    json.dump(out, open("results/nn_precision_fullpool.json", "w"), indent=1)


if __name__ == "__main__":
    main()
