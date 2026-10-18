"""D-CORALHEAD2: inductive signer-aligned linear map on the half-subset with LABEL-FREE CLS-cluster pseudo-signers (proposal17 G17b).
Same objective as coral_head.py; gallery = train-half (20,077 clips, teacher stride-16 grp_fin), queries = val-half (5,152).
Groups: true signers (diagnostic), CLS k-means K=35/70 (label-free), grp k-means K=70 (label-free), and ablations mean-only / cov-only."""
import json

import numpy as np
import torch
from sklearn.cluster import KMeans

from analyze_emb import meta
from common import *  # noqa
import geo_headroom as G
from coral_head import train_W

torch.set_num_threads(4)
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
mu, W0 = G.fit_pca_whiten(Xg, 512, 1.0)
r1 = lambda W: float(metrics(class_max_ranks((Xq - mu) @ W, yq, (Xg - mu) @ W, yg))["R1"])
out = {"raw": float(metrics(class_max_ranks(Xq, yq, Xg, yg))["R1"]), "whiten_init": r1(W0)}
print(out, flush=True)
sidx = {u: i for i, u in enumerate(np.unique(sg))}
strue = np.array([sidx[u] for u in sg])
Zg = (Xg - mu) @ W0
groups = {
    "true_signers": strue,
    "CLS_kmeans35": KMeans(35, n_init=3, random_state=0).fit(Cg).labels_,
    "CLS_kmeans70": KMeans(70, n_init=3, random_state=0).fit(Cg).labels_,
    "grp_kmeans70": KMeans(70, n_init=2, random_state=0).fit(Zg[:, :64]).labels_,
}
cfgs = [("true_signers", 1, 1), ("CLS_kmeans35", 1, 1), ("CLS_kmeans70", 1, 1), ("grp_kmeans70", 1, 1), ("true_signers", 1, 0),
        ("true_signers", 0, 1)]
for name, a_mu, a_cov in cfgs:
    W = train_W(Xg, groups[name], W0, mu, a_mu=a_mu, a_cov=a_cov, steps=250)
    k = f"{name}_mu{a_mu}_cov{a_cov}"
    out[k] = r1(W)
    print(k, out[k], flush=True)
    json.dump(out, open("results/coral_head2.json", "w"), indent=1)
