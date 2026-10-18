"""D-NNHEAD: how much can cross-instance nearest-neighbour positives repair the frozen embedding, as a function of their precision?
(proposal17 diagnostic G13; frozen-feature analogue of the MAE-CT / STARS NN-contrastive stage; CPU, minutes)

A small MLP head is trained on the label-free PCA-whitened PLctx train embeddings (40k clips) with InfoNCE (in-batch negatives);
positives are mined from the embedding itself:
   none      : top-1 neighbour of any other clip                       (control: mostly same-signer)
   cluster70 : top-1 neighbour outside the clip's k-means cluster      (label-free pseudo-signer exclusion)
   signer    : top-1 neighbour of a different TRUE signer              (diagnostic; signer id = weak supervision)
   gloss     : random clip of the same gloss                            (supervised upper bound, diagnostic)
Optionally only mutual pairs.  Rounds 2-3 re-mine with the head's own embedding (self-training).  Evaluation: val R@1 of head(Z_val)
vs head(Z_train), class-max cosine.  No val statistic and no val label is used in training.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from common import *  # noqa
import geo_headroom as G

torch.set_num_threads(4)
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/nn_head.json"
dev = "cpu"


def mine(Z, groups, mutual_only, bs=2048):
    Zt = torch.as_tensor(l2(Z))
    nn_ = np.zeros(len(Z), np.int64)
    gt = torch.as_tensor(groups) if groups is not None else None
    for a in range(0, len(Z), bs):
        s = Zt[a : a + bs] @ Zt.T
        s[torch.arange(s.shape[0]), torch.arange(a, a + s.shape[0])] = -2
        if gt is not None:
            s[gt[a : a + bs, None] == gt[None, :]] = -2
        nn_[a : a + bs] = s.argmax(1).numpy()
    mut = nn_[nn_] == np.arange(len(Z))
    return nn_, mut


def train_head(Z, pos, keep, epochs=40, bs=1024, tau=0.1, seed=0, out=256, noise=0.1):
    torch.manual_seed(seed)
    Zt = torch.as_tensor(Z, dtype=torch.float32)
    head = nn.Sequential(nn.Linear(Z.shape[1], 1024), nn.GELU(), nn.Linear(1024, out))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    idx_all = np.where(keep)[0]
    steps = epochs * max(1, len(idx_all) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        b = rng.choice(idx_all, size=min(bs, len(idx_all)), replace=False)
        a, p = Zt[b], Zt[pos[b]]
        a = a + noise * torch.randn_like(a) * a.std()
        p = p + noise * torch.randn_like(p) * p.std()
        za, zp = F.normalize(head(a), dim=1), F.normalize(head(p), dim=1)
        logits = za @ zp.T / tau
        loss = 0.5 * (F.cross_entropy(logits, torch.arange(len(b))) + F.cross_entropy(logits.T, torch.arange(len(b))))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return head


def embed(head, Z):
    with torch.no_grad():
        return F.normalize(head(torch.as_tensor(Z, dtype=torch.float32)), dim=1).numpy()


def main():
    d = load_baseline()
    tr, va = d["train"], d["val"]
    mu, W = G.fit_pca_whiten(tr["X"], 512, 1.0)
    Zg, Zq = G.apply_lin(tr["X"], mu, W).astype(np.float32), G.apply_lin(va["X"], mu, W).astype(np.float32)
    yg, sg = tr["y"], tr["signer"]
    sidx = {u: i for i, u in enumerate(np.unique(sg))}
    strue = np.array([sidx[u] for u in sg])
    cl70 = KMeans(70, n_init=2, random_state=0).fit(Zg[:, :64]).labels_
    r1 = lambda A, B: float(metrics(class_max_ranks(A, va["y"], B, yg))["R1"])
    out = {"whiten_reference": r1(Zq, Zg)}
    rng = np.random.default_rng(0)
    # same-gloss positives (supervised upper bound)
    by = {c: np.where(yg == c)[0] for c in np.unique(yg)}
    gpos = np.array([rng.choice(by[c][by[c] != i]) if len(by[c]) > 1 else i for i, c in enumerate(yg)])
    configs = [("none", None, False), ("cluster70", cl70, False), ("cluster70_mutual", cl70, True), ("signer", strue, False),
               ("signer_mutual", strue, True)]
    for name, groups, mutual in configs:
        Zcur_g, Zcur_q = Zg, Zq
        for rnd in (1, 2, 3):
            nn_, mut = mine(Zcur_g, groups, mutual)
            keep = mut if mutual else np.ones(len(Zg), bool)
            prec = float((yg[nn_][keep] == yg[keep]).mean() * 100)
            head = train_head(Zg, nn_, keep)
            Zcur_g, Zcur_q = embed(head, Zg), embed(head, Zq)
            out[f"{name}_round{rnd}"] = {"mined_precision": prec, "kept_fraction": float(keep.mean() * 100), "val_R1": r1(Zcur_q, Zcur_g)}
            print(name, rnd, out[f"{name}_round{rnd}"], flush=True)
            json.dump(out, open(OUT, "w"), indent=1)
    head = train_head(Zg, gpos, np.ones(len(Zg), bool))
    out["gloss_oracle"] = {"mined_precision": 100.0, "val_R1": r1(embed(head, Zq), embed(head, Zg))}
    print("gloss oracle", out["gloss_oracle"], flush=True)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
