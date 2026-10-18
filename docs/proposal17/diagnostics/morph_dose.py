"""R1-F7 fix: morphology perturbation DOSE-RESPONSE on a SIGNER-BALANCED subset (proposal17 diagnostic G7b).

emb_variants.py --balanced 300 [--morph MODE --morph_scale S] gives teacher embeddings of 300 val clips per signer (P12, P21, P26, P39, P5;
P13 excluded) clean and under clip-constant perturbations of arm length / hand scale / all / non-uniform (independent arm, hand, per-finger
scales), scales +-15 %, 30 %, 50 %.  For each condition: mean cosine to the clean embedding, paired R@1 (same queries clean vs perturbed
against the CLEAN train-half gallery), per-signer R@1 change, and the share of the shift energy in the 34-d signer subspace.
References from the clean val half: cosine of same-gloss/different-signer pairs.
"""
from __future__ import annotations

import json
import os

import numpy as np

from analyze_emb import meta
from common import *  # noqa

EMB = f"{SCRATCH}/emb"


def main():
    man = manifests()
    gl = sorted({m["gloss"] for sp in man.values() for m in sp})
    gi = {g: i for i, g in enumerate(gl)}
    zt = np.load(f"{EMB}/train_teacher.npz")
    mt = meta(zt["ids"], man["train"])
    yg = np.array([gi[m["gloss"]] for m in mt])
    sg = np.array([m["signer"] for m in mt])
    Xg = zt["grp_fin"].astype(np.float32)
    clean = np.load(f"{EMB}/val_bal300_teacher.npz")
    mv = meta(clean["ids"], man["val"])
    yv = np.array([gi[m["gloss"]] for m in mv])
    sv = np.array([m["signer"] for m in mv])
    Xc = clean["grp_fin"].astype(np.float32)
    base = class_max_ranks(Xc, yv, Xg, yg)
    res = {"n_queries": int(len(yv)), "clips_per_signer": {s: int((sv == s).sum()) for s in np.unique(sv)},
           "R1_clean": metrics(base)["R1"],
           "R1_clean_per_signer": {s: float((base[sv == s] <= 1).mean() * 100) for s in np.unique(sv)}}
    # references on the balanced clean subset
    pairs = [(i, j) for g in np.unique(yv) for i in np.where(yv == g)[0] for j in np.where(yv == g)[0] if i < j and sv[i] != sv[j]]
    pairs = np.array(pairs) if pairs else np.zeros((0, 2), int)
    if len(pairs):
        res["ref_cos_same_gloss_diff_signer"] = float((l2(Xc[pairs[:, 0]]) * l2(Xc[pairs[:, 1]])).sum(1).mean())
        res["ref_norm_same_gloss_diff_signer"] = float(np.linalg.norm(l2(Xc[pairs[:, 0]]) - l2(Xc[pairs[:, 1]]), axis=1).mean())
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, len(Xc), 100000), rng.integers(0, len(Xc), 100000)
    res["ref_cos_random_pair"] = float((l2(Xc[a]) * l2(Xc[b])).sum(1).mean())
    # signer subspace from the train half (gloss-residualised signer means)
    En = l2(Xg)
    gm = np.zeros((yg.max() + 1, En.shape[1]))
    cnt = np.bincount(yg, minlength=yg.max() + 1)
    np.add.at(gm, yg, En)
    gm = gm / np.maximum(cnt, 1)[:, None]
    R = En - gm[yg]
    sm = np.stack([R[sg == s].mean(0) for s in np.unique(sg)])
    Vs = np.linalg.svd((sm - sm.mean(0)).T, full_matrices=False)[0][:, :34]
    Q = np.linalg.qr(rng.standard_normal((En.shape[1], 34)))[0]
    res["conditions"] = {}
    for fn in sorted(os.listdir(EMB)):
        if not fn.startswith("val_bal300_morph-"):
            continue
        tag = fn[len("val_bal300_morph-"): -len("_teacher.npz")]
        z = np.load(f"{EMB}/{fn}")
        assert list(z["ids"]) == list(clean["ids"])
        Xm = z["grp_fin"].astype(np.float32)
        ranks = class_max_ranks(Xm, yv, Xg, yg)
        d = l2(Xm) - l2(Xc)
        e = (d ** 2).sum(1)
        okm = e > 0
        res["conditions"][tag] = {
            "cos_to_clean": float((l2(Xc) * l2(Xm)).sum(1).mean()),
            "shift_norm": float(np.sqrt(e).mean()),
            "R1": metrics(ranks)["R1"],
            "R1_delta_vs_clean": metrics(ranks)["R1"] - res["R1_clean"],
            "R1_delta_per_signer": {s: float(((ranks[sv == s] <= 1).mean() - (base[sv == s] <= 1).mean()) * 100) for s in np.unique(sv)},
            "shift_energy_in_signer_subspace34": float((((d[okm] @ Vs) ** 2).sum(1) / e[okm]).mean()) if okm.any() else None,
            "shift_energy_random_subspace34": float((((d[okm] @ Q) ** 2).sum(1) / e[okm]).mean()) if okm.any() else None,
        }
        print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["conditions"][tag].items() if k != "R1_delta_per_signer"}, flush=True)
    json.dump(res, open("results/morph_dose.json", "w"), indent=1, allow_nan=False)


if __name__ == "__main__":
    main()
