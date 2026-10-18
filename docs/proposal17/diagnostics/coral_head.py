"""D-CORALHEAD: can an INDUCTIVE linear map trained with a label-free cluster-alignment objective reproduce the transductive CORAL gain?
(proposal17 diagnostic G17; frozen-feature analogue of a Deep-CORAL / domain-invariant-component regulariser inside JEPA)

Map z = W (x - mu), W: 768 -> 512 initialised at the label-free PCA-whitening.  Loss on TRAIN embeddings only:
    L = a_tot * ||Cov(z) - I||_F^2 / d                      (keep isotropy; avoids the trivial solution)
      + a_mu  * sum_c w_c ||mean_c(z)||^2                    (cluster means aligned)
      + a_cov * sum_c w_c ||Cov_c(z) - Cov(z)||_F^2 / d      (cluster covariances aligned)
Clusters c are (a) TRUE signers (diagnostic; signer id = weak supervision) or (b) k-means clusters of the whitened embedding (label-free).
A single W is applied to every clip at test time (no val statistics): val R@1 of W-embedded queries vs W-embedded train gallery.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans

from common import *  # noqa
import geo_headroom as G

torch.set_num_threads(4)
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/coral_head.json"


def cov(z):
    zc = z - z.mean(0, keepdim=True)
    return zc.T @ zc / (len(z) - 1)


def train_W(Xg, labels, W0, mu0, a_tot=1.0, a_mu=1.0, a_cov=1.0, steps=300, lr=2e-3, seed=0, min_n=100):
    torch.manual_seed(seed)
    X = torch.as_tensor(Xg - mu0, dtype=torch.float32)
    W = torch.nn.Parameter(torch.as_tensor(W0, dtype=torch.float32).clone())
    opt = torch.optim.Adam([W], lr=lr)
    cl = [torch.as_tensor(np.where(labels == c)[0]) for c in np.unique(labels) if (labels == c).sum() >= min_n]
    wts = torch.as_tensor([len(i) for i in cl], dtype=torch.float32)
    wts = wts / wts.sum()
    d = W.shape[1]
    I = torch.eye(d)
    for it in range(steps):
        Z = X @ W
        Ct = cov(Z)
        loss = a_tot * ((Ct - I) ** 2).sum() / d
        lm = lc = 0.0
        for w, idx in zip(wts, cl):
            zc = Z[idx]
            lm = lm + w * (zc.mean(0) ** 2).sum()
            lc = lc + w * ((cov(zc) - Ct) ** 2).sum() / d
        loss = loss + a_mu * lm + a_cov * lc
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 100 == 0:
            print(f"  it {it} loss {loss.item():.4f} tot {((Ct - I) ** 2).sum().item() / d:.4f} mu {float(lm):.4f} cov {float(lc):.4f}", flush=True)
    return W.detach().numpy()


def main():
    d = load_baseline()
    tr, va = d["train"], d["val"]
    Xg, yg, sg = tr["X"], tr["y"], tr["signer"]
    mu, W0 = G.fit_pca_whiten(Xg, 512, 1.0)
    r1 = lambda W: float(metrics(class_max_ranks((va["X"] - mu) @ W, va["y"], (Xg - mu) @ W, yg))["R1"])
    out = {"whiten_init": r1(W0)}
    print(out, flush=True)
    sidx = {u: i for i, u in enumerate(np.unique(sg))}
    strue = np.array([sidx[u] for u in sg])
    Zg = (Xg - mu) @ W0
    clusters = {"true_signers": strue}
    for K in (35, 70):
        clusters[f"kmeans{K}"] = KMeans(K, n_init=2, random_state=0).fit(Zg[:, :64]).labels_
    for name, lab in clusters.items():
        for a_mu, a_cov in ((1.0, 1.0), (1.0, 0.0), (0.0, 1.0)):
            W = train_W(Xg, lab, W0, mu, a_mu=a_mu, a_cov=a_cov)
            key = f"{name}_mu{a_mu}_cov{a_cov}"
            out[key] = r1(W)
            print(key, out[key], flush=True)
            json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
