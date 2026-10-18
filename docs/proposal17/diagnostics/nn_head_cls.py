"""R1-F2 follow-up: LABEL-FREE NN-positive head (half-subset): positives mined in the CLS-cluster-aligned geometry with CLS-cluster exclusion.

Half-subset (train-half gallery 20k, val-half queries 5.2k, teacher grp_fin, CLS K = 48 clusters).  Head = nn_head.train_head (MLP, InfoNCE,
in-batch negatives) trained on INDUCTIVE whitened inputs; only the positive-mining geometry changes:
   whiten geometry, exclusion by CLS cluster          (label-free)
   CLS-CORAL+whiten geometry, exclusion by CLS cluster (label-free)      <- the proposal's G-NN test
   CLS-CORAL+whiten geometry, exclusion by true signer (diagnostic)
   gloss oracle                                         (upper bound)
Reports mined precision, kept fraction and val R@1 (whiten reference 24.94 on this subset).
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
from nn_precision_fullpool import cross_group_nn

torch.set_num_threads(4)


def main():
    EMB = f"{SCRATCH}/emb"
    man = manifests()
    gl = sorted({m["gloss"] for sp in man.values() for m in sp})
    gi = {g: i for i, g in enumerate(gl)}
    zt, zv = np.load(f"{EMB}/train_teacher.npz"), np.load(f"{EMB}/val_teacher.npz")
    mt, mv = meta(zt["ids"], man["train"]), meta(zv["ids"], man["val"])
    yg = np.array([gi[m["gloss"]] for m in mt])
    sg = np.array([m["signer"] for m in mt])
    yq = np.array([gi[m["gloss"]] for m in mv])
    Xg, Xq = zt["grp_fin"].astype(np.float32), zv["grp_fin"].astype(np.float32)
    Cg = l2(zt["cls_fin"].astype(np.float32))
    ok = np.linalg.norm(Xg, axis=1) > 0
    Xg, yg, sg, Cg = Xg[ok], yg[ok], sg[ok], Cg[ok]
    sidx = {u: i for i, u in enumerate(np.unique(sg))}
    s_i = np.array([sidx[u] for u in sg])
    cl = KMeans(48, n_init=3, random_state=0).fit(Cg).labels_
    mu, W = G.fit_pca_whiten(Xg, 512, 1.0)
    Zg, Zq = G.apply_lin(Xg, mu, W).astype(np.float32), G.apply_lin(Xq, mu, W).astype(np.float32)
    Gc = SN.coral(Xg, cl)
    m2, W2 = G.fit_pca_whiten(Gc, 512, 1.0)
    Zc = G.apply_lin(Gc, m2, W2)
    r1 = lambda A, B: float(metrics(class_max_ranks(A, yq, B, yg))["R1"])
    out = {"whiten_reference": r1(Zq, Zg)}
    cfg = {"whiten_geometry_excl_CLS48": (Zg, cl), "CLSCORAL_geometry_excl_CLS48": (Zc, cl), "CLSCORAL_geometry_excl_true_signer": (Zc, s_i)}
    for name, (geo, groups) in cfg.items():
        nn_ = cross_group_nn(geo, groups)
        mut = nn_[nn_] == np.arange(len(nn_))
        for mutual_only in (False, True):
            keep = mut if mutual_only else np.ones(len(Zg), bool)
            head = NH.train_head(Zg, nn_, keep)
            key = f"{name}{'_mutual' if mutual_only else ''}"
            out[key] = {"mined_precision": float((yg[nn_][keep] == yg[keep]).mean() * 100), "kept": float(keep.mean() * 100),
                        "val_R1": r1(NH.embed(head, Zq), NH.embed(head, Zg))}
            print(key, out[key], flush=True)
            json.dump(out, open("results/nn_head_cls.json", "w"), indent=1)
    rng = np.random.default_rng(0)
    by = {c: np.where(yg == c)[0] for c in np.unique(yg)}
    gpos = np.array([rng.choice(by[c][by[c] != i]) if len(by[c]) > 1 else i for i, c in enumerate(yg)])
    head = NH.train_head(Zg, gpos, np.ones(len(Zg), bool))
    out["gloss_oracle"] = {"val_R1": r1(NH.embed(head, Zq), NH.embed(head, Zg))}
    print("gloss oracle", out["gloss_oracle"], flush=True)
    json.dump(out, open("results/nn_head_cls.json", "w"), indent=1)


if __name__ == "__main__":
    main()
