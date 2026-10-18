"""Robustness of the retrieval ladder to the choice of held-out signers (proposal17 diagnostic G12b).

The 5-effective-signer val set is a fragile basis for ladder claims.  Here, 5 random splits of the 35 TRAIN signers: 6 signers become queries,
the other 29 form the gallery; every normalisation / ceiling is refit on the gallery signers only.  CAVEAT: the encoder saw these clips
during self-supervised pretraining (no labels), so absolute R@1 is optimistic; only the RELATIVE gains between ladder rungs are of interest.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.cluster import KMeans

from common import *  # noqa
import geo_headroom as G
import signer_norm as SN


def main():
    d = load_baseline()
    tr = d["train"]
    X, y, s = tr["X"], tr["y"], tr["signer"]
    sg = np.unique(s)
    rng = np.random.default_rng(0)
    rungs = ["raw", "whiten", "CMN+whiten", "CORAL+whiten", "LEACE+whiten", "cluster-CORAL(label-free)+whiten", "LDA96"]
    acc = {r: [] for r in rungs}
    for split in range(5):
        hold = rng.choice(sg, size=6, replace=False)
        mq = np.isin(s, hold)
        Xg, yg, sgl = X[~mq], y[~mq], s[~mq]
        Xq, yq, sq = X[mq], y[mq], s[mq]
        ok = np.isin(yq, np.unique(yg))
        Xq, yq, sq = Xq[ok], yq[ok], sq[ok]
        # cap queries for speed
        if len(Xq) > 6000:
            sub = rng.choice(len(Xq), 6000, replace=False)
            Xq, yq, sq = Xq[sub], yq[sub], sq[sub]
        r1 = lambda Q, Gl: float(metrics(class_max_ranks(Q, yq, Gl, yg))["R1"])
        acc["raw"].append(r1(Xq, Xg))
        mu, W = G.fit_pca_whiten(Xg, 512, 1.0)
        acc["whiten"].append(r1(G.apply_lin(Xq, mu, W), G.apply_lin(Xg, mu, W)))
        Gm, Qm = SN.cmn(Xg, sgl), SN.cmn(Xq, sq)
        m1, W1 = G.fit_pca_whiten(Gm, 512, 1.0)
        acc["CMN+whiten"].append(r1(G.apply_lin(Qm, m1, W1), G.apply_lin(Gm, m1, W1)))
        Gc, Qc = SN.coral(Xg, sgl), SN.coral(Xq, sq)
        m2, W2 = G.fit_pca_whiten(Gc, 512, 1.0)
        acc["CORAL+whiten"].append(r1(G.apply_lin(Qc, m2, W2), G.apply_lin(Gc, m2, W2)))
        su = np.unique(sgl)
        z = np.array([{u: i for i, u in enumerate(su)}[u] for u in sgl])
        ml, M = G.fit_leace(Xg, z)
        Gl, Ql = ml + (Xg - ml) @ M, ml + (Xq - ml) @ M
        m4, W4 = G.fit_pca_whiten(Gl, 512, 1.0)
        acc["LEACE+whiten"].append(r1(G.apply_lin(Ql, m4, W4), G.apply_lin(Gl, m4, W4)))
        Zg, Zq = G.apply_lin(Xg, mu, W), G.apply_lin(Xq, mu, W)
        lq = KMeans(6, n_init=3, random_state=0).fit(Zq[:, :64]).labels_
        lg = KMeans(60, n_init=2, random_state=0).fit(Zg[:, :64]).labels_
        Gu, Qu = SN.coral(Xg, lg), SN.coral(Xq, lq)
        m3, W3 = G.fit_pca_whiten(Gu, 512, 1.0)
        acc["cluster-CORAL(label-free)+whiten"].append(r1(G.apply_lin(Qu, m3, W3), G.apply_lin(Gu, m3, W3)))
        ml_, Wl = G.fit_lda(Xg, yg, 96, 0.05)
        acc["LDA96"].append(r1(G.apply_lin(Xq, ml_, Wl), G.apply_lin(Xg, ml_, Wl)))
        print(split, {k: round(v[-1], 1) for k, v in acc.items()}, flush=True)
    out = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "per_split": [float(x) for x in v]} for k, v in acc.items()}
    for k in rungs[1:]:
        dlt = np.array(acc[k]) - np.array(acc["raw"])
        out[k]["delta_vs_raw_mean"] = float(dlt.mean())
        out[k]["delta_vs_raw_min"] = float(dlt.min())
    json.dump(out, open("results/ladder_cv.json", "w"), indent=1)
    print(json.dumps({k: (round(v["mean"], 1), round(v["sd"], 1)) for k, v in out.items()}))


if __name__ == "__main__":
    main()
