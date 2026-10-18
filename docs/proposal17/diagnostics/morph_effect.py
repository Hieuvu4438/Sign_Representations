"""D-MORPH: how sensitive is the PLctx teacher embedding to clip-constant analytic morphology perturbations, and do the
perturbation-induced shifts live in the signer subspace?  (proposal17 diagnostic G7)

Inputs (emb_variants.py): val clean teacher embeddings, val morph-{face,arm,hand,all} teacher embeddings (first 1500 val clips of
the every-2nd subset), train-half teacher embeddings.  Measures
  1. cos(emb(x), emb(morph(x))) per mode   vs   cos of same-gloss-different-signer pairs and same-signer-different-gloss pairs
  2. R@1 of perturbed queries against the CLEAN train-half gallery vs the same queries clean (invariance deficit)
  3. fraction of the shift energy inside the 34-d signer subspace (gloss-residualised signer means of the train half) vs the
     fraction expected for a random 34-d subspace (4.4 %) and vs the top-34 within-gloss (Sw) eigenspace
"""
from __future__ import annotations

import json
import sys

import numpy as np

from common import *  # noqa

EMB = f"{SCRATCH}/emb"
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/morph_effect.json"
N = 1500


def main():
    man = manifests()
    gl = sorted({m["gloss"] for sp in man.values() for m in sp})
    gi = {g: i for i, g in enumerate(gl)}
    by = {sp: {m["sample_id"]: m for m in man[sp]} for sp in man}
    zt = np.load(f"{EMB}/train_teacher.npz")
    zv = np.load(f"{EMB}/val_teacher.npz")
    Xg = zt["grp_fin"].astype(np.float32)
    yg = np.array([gi[by["train"][i]["gloss"]] for i in zt["ids"]])
    sg = np.array([by["train"][i]["signer"] for i in zt["ids"]])
    Xv = zv["grp_fin"].astype(np.float32)
    yv = np.array([gi[by["val"][i]["gloss"]] for i in zv["ids"]])
    sv = np.array([by["val"][i]["signer"] for i in zv["ids"]])
    clean = Xv[:N]
    res = {}
    rng = np.random.default_rng(0)
    # reference cosines among clean val clips
    a, b = rng.integers(0, len(Xv), 200000), rng.integers(0, len(Xv), 200000)
    cs = (l2(Xv[a]) * l2(Xv[b])).sum(1)
    same_g = (yv[a] == yv[b]) & (sv[a] != sv[b]) & (a != b)
    same_s = (sv[a] == sv[b]) & (yv[a] != yv[b])
    res["ref_cos"] = {"same_gloss_diff_signer": float(cs[same_g].mean()) if same_g.any() else None,
                      "same_signer_diff_gloss": float(cs[same_s].mean()), "random_pair": float(cs.mean())}
    # more same-gloss pairs: enumerate within gloss
    pairs = []
    for g in np.unique(yv):
        idx = np.where(yv == g)[0]
        for i in idx:
            for j in idx:
                if i < j and sv[i] != sv[j]:
                    pairs.append((i, j))
    pairs = np.array(pairs)
    res["ref_cos"]["same_gloss_diff_signer"] = float((l2(Xv[pairs[:, 0]]) * l2(Xv[pairs[:, 1]])).sum(1).mean())
    base_ranks = class_max_ranks(clean, yv[:N], Xg, yg)
    res["R1_clean_subset"] = metrics(base_ranks)["R1"]
    # signer subspace (gloss-residualised) from the train half, and within-gloss top eigenspace
    En = l2(Xg)
    gm = np.zeros((yg.max() + 1, En.shape[1]))
    cnt = np.bincount(yg, minlength=yg.max() + 1)
    np.add.at(gm, yg, En)
    gm = gm / np.maximum(cnt, 1)[:, None]
    R = En - gm[yg]
    sm = np.stack([R[sg == s].mean(0) for s in np.unique(sg)])
    Vs = np.linalg.svd((sm - sm.mean(0)).T, full_matrices=False)[0][:, :34]
    ww, Vw = np.linalg.eigh(np.cov(R.T))
    Vw = Vw[:, ::-1][:, :34]
    Q = np.linalg.qr(rng.standard_normal((En.shape[1], 34)))[0]
    for mode in ("face", "arm", "hand", "all"):
        try:
            zm = np.load(f"{EMB}/val_morph-{mode}-s1.15_teacher.npz")
        except FileNotFoundError:
            continue
        assert list(zm["ids"][:N]) == list(zv["ids"][:N])
        Xm = zm["grp_fin"].astype(np.float32)[:N]
        cosv = (l2(clean) * l2(Xm)).sum(1)
        ranks = class_max_ranks(Xm, yv[:N], Xg, yg)
        delta = l2(Xm) - l2(clean)
        e = (delta ** 2).sum(1)
        okm = e > 1e-12  # face mode: the encoder is exactly invariant, the shift is numerically zero (review F15: no NaN in JSON)
        row = {
            "cos_clean_vs_morph_mean": float(cosv.mean()),
            "R1_morph_queries": metrics(ranks)["R1"],
            "R1_drop": res["R1_clean_subset"] - metrics(ranks)["R1"],
            "shift_energy_in_signer_subspace34": float((((delta[okm] @ Vs) ** 2).sum(1) / e[okm]).mean()) if okm.any() else None,
            "shift_energy_in_within_gloss_top34": float((((delta[okm] @ Vw) ** 2).sum(1) / e[okm]).mean()) if okm.any() else None,
            "shift_energy_random_subspace34": float((((delta[okm] @ Q) ** 2).sum(1) / e[okm]).mean()) if okm.any() else None,
            "mean_shift_cos_with_signer_subspace": float(np.linalg.norm(Vs.T @ delta.mean(0)) / max(np.linalg.norm(delta.mean(0)), 1e-12)),
        }
        # fraction of the true inter-signer displacement (gloss-matched) that the shift resembles: skip; report shift norm
        row["shift_norm_mean"] = float(np.sqrt(e).mean())
        res[mode] = row
        print(mode, {k: (round(v, 4) if v is not None else None) for k, v in row.items()}, flush=True)
    # typical gloss-matched cross-signer displacement norm for scale
    d2 = np.linalg.norm(l2(Xv[pairs[:, 0]]) - l2(Xv[pairs[:, 1]]), axis=1)
    res["ref_norm_same_gloss_diff_signer"] = float(d2.mean())
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res["ref_cos"]), res["ref_norm_same_gloss_diff_signer"])


if __name__ == "__main__":
    main()
