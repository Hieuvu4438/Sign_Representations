"""R1-F9 fix: emulate module R's mini-batch mechanics on the frozen-feature head (proposal17 diagnostic G17c).

coral_head*.py used FULL-batch statistics.  In SA-JEPA training a batch has 32 clips; the per-cluster mean and covariance must come from an
EMA statistics bank mixed with the batch estimate (alpha = 0.25, gradient only through the batch part) and clusters with too few samples are
skipped.  This script reproduces that estimator on the same head (768 -> 512, init at the label-free whitening) with CLS-cluster
pseudo-signers (K = 48, half-subset gallery) and reports the final val R@1 and the magnitude of the alignment loss over training.
Compare with the full-batch heads (CLS K=35: 28.5; true signers 28.1; whiten init 24.9).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans

from analyze_emb import meta
from common import *  # noqa
import geo_headroom as G

torch.set_num_threads(4)
OUT = "results/coral_head_mb.json"


def run(X, labels, W0, mu0, Xq, yq, yg, B, a_mu, a_cov, a_iso, alpha=0.25, beta=0.99, min_count=20, steps=3000, lr=1e-3, seed=0, tag="", cb=None):
    """cb = (n_clusters, clips_per_cluster) -> cluster-balanced batches (label-free: clusters are the pseudo-signers); B is then n*m."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xt = torch.as_tensor(X - mu0, dtype=torch.float32)
    W = torch.nn.Parameter(torch.as_tensor(W0, dtype=torch.float32).clone())
    opt = torch.optim.Adam([W], lr=lr)
    d = W.shape[1]
    ks = np.unique(labels)
    K = len(ks)
    kidx = {k: i for i, k in enumerate(ks)}
    lab = torch.as_tensor(np.array([kidx[k] for k in labels]))
    mu_e = torch.zeros(K, d)
    cov_e = torch.zeros(K, d, d)
    cnt_e = torch.zeros(K)
    mu_g = torch.zeros(d)
    cov_g = torch.eye(d)
    g_init = False
    I = torch.eye(d)
    log = []
    members = {k: np.where(labels == k)[0] for k in ks}
    big = [k for k in ks if len(members[k]) >= (cb[1] if cb else 1)]
    for step in range(steps):
        if cb is None:
            b = torch.as_tensor(rng.choice(len(X), size=B, replace=False))
        else:
            cs = rng.choice(big, size=cb[0], replace=False)
            b = torch.as_tensor(np.concatenate([rng.choice(members[k], size=cb[1], replace=False) for k in cs]))
            B = len(b)
        Z = Xt[b] @ W
        lb = lab[b]
        Zd = Z.detach()
        if not g_init:
            mu_g, cov_g, g_init = Zd.mean(0), (Zd - Zd.mean(0)).T @ (Zd - Zd.mean(0)) / B + 1e-3 * I, True
        mu_hat = (1 - alpha) * mu_g + alpha * Z.mean(0)
        Zc = Z - mu_g
        cov_hat = (1 - alpha) * cov_g + alpha * (Zc.T @ Zc) / B
        loss_R = a_iso * ((cov_hat - I) ** 2).sum() / d
        tot_w = 0.0
        terms = 0.0
        present = torch.unique(lb)
        used = []
        for k in present.tolist():
            m = lb == k
            n = int(m.sum())
            if cnt_e[k] < min_count:  # cluster not yet reliable: only update its statistics
                continue
            Zk = Z[m]
            mu_k = (1 - alpha) * mu_e[k] + alpha * Zk.mean(0)
            Zkc = Zk - mu_e[k]
            cov_k = (1 - alpha) * cov_e[k] + alpha * (Zkc.T @ Zkc) / n
            terms = terms + n * (a_mu * ((mu_k - mu_hat) ** 2).sum() + a_cov * ((cov_k - cov_hat) ** 2).sum() / d)
            tot_w += n
            used.append(k)
        L_align = terms / tot_w if tot_w > 0 else torch.zeros(())
        loss = loss_R + L_align
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():  # update EMA statistics with the (detached) batch
            for k in present.tolist():
                m = lb == k
                n = int(m.sum())
                Zk = Zd[m]
                if cnt_e[k] == 0:
                    mu_e[k] = Zk.mean(0)
                    cov_e[k] = (Zk - mu_e[k]).T @ (Zk - mu_e[k]) / max(n, 1) + 1e-3 * I
                else:
                    cov_e[k] = beta * cov_e[k] + (1 - beta) * (Zk - mu_e[k]).T @ (Zk - mu_e[k]) / n
                    mu_e[k] = beta * mu_e[k] + (1 - beta) * Zk.mean(0)
                cnt_e[k] += n
            mu_g = beta * mu_g + (1 - beta) * Zd.mean(0)
            cov_g = beta * cov_g + (1 - beta) * (Zd - mu_g).T @ (Zd - mu_g) / B
        if step % 250 == 0 or step == steps - 1:
            log.append({"step": step, "L_align": float(L_align), "L_iso": float(loss_R), "clusters_used": len(used),
                        "clusters_present": int(len(present)), "grad_norm_W": float(W.grad.norm())})
    Wn = W.detach().numpy()
    r1 = float(metrics(class_max_ranks((Xq - mu0) @ Wn, yq, (X - mu0) @ Wn, yg))["R1"])
    print(tag, "val R@1", round(r1, 2), "log", [(l["step"], round(l["L_align"], 4), l["clusters_used"]) for l in log[:4]], "...", log[-1], flush=True)
    return {"R1": r1, "log": log}


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
    Cg = l2(zt["cls_fin"].astype(np.float32))
    mu, W0 = G.fit_pca_whiten(Xg, 512, 1.0)
    r0 = float(metrics(class_max_ranks((Xq - mu) @ W0, yq, (Xg - mu) @ W0, yg))["R1"])
    out = {"whiten_init": r0}
    print(out, flush=True)
    lab = KMeans(48, n_init=3, random_state=0).fit(Cg).labels_
    cfgs = [("CLS48_B32_random_default(0.25,1,0.25)", 32, None, (0.25, 1.0, 0.25)),
            ("CLS48_B32_clusterbalanced4x8_default", 32, (4, 8), (0.25, 1.0, 0.25)),
            ("CLS48_B256_random_default", 256, None, (0.25, 1.0, 0.25)),
            ("CLS48_B32_random_headstyle(1,1,1)", 32, None, (1.0, 1.0, 1.0)),
            ("CLS48_B32_clusterbalanced4x8_headstyle", 32, (4, 8), (1.0, 1.0, 1.0))]
    for key, B, cb, (a_mu, a_cov, a_iso) in cfgs:
        out[key] = run(Xg, lab, W0, mu, Xq, yq, yg, B, a_mu, a_cov, a_iso, tag=key, cb=cb)
        json.dump(out, open(OUT, "w"), indent=1)
    rng = np.random.default_rng(0)  # shuffled-cluster control with the cluster-balanced sampler and default weights
    out["CLS48_B32_clusterbalanced4x8_default_SHUFFLED"] = run(Xg, rng.permutation(lab), W0, mu, Xq, yq, yg, 32, 0.25, 1.0, 0.25, tag="shuffled", cb=(4, 8))
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
