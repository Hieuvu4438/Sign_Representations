"""D-NUIS: what is the within-gloss variance of the PLctx full160 embedding made of? (proposal17 diagnostic G2)

LDA ceiling (47.6) >> whitening (29.4) >> raw (22.7) says the gloss information is present but buried under variance that is
NOT gloss. Which nuisance? We decompose the within-gloss residual R = X - gloss_mean(X) (TRAIN, 40k clips, 35 signers) into
variance explained by categorical metadata factors (marginal eta^2 and joint R^2):
  signer | clip duration bin | fps | resolution | signer+duration+fps+resolution
and measure the dimensionality / alignment of the within-gloss (Sw) vs between-gloss (Sb) covariance.
Metadata only: no pose, no labels beyond gloss/signer ids. Val is not used here.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from common import *  # noqa

META = "/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc/metadata_{}.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/nuisance_decomp.json"


def eta2(R, g):
    """Share of the total sum of squares of R explained by the categorical factor g (marginal)."""
    tot = (R ** 2).sum()
    ss = 0.0
    for u in np.unique(g):
        m = g == u
        ss += m.sum() * (R[m].mean(0) ** 2).sum()
    return ss / tot


def r2_joint(R, factors):
    """Joint R^2 of an additive one-hot regression on several categorical factors (ridge-free lstsq)."""
    cols = []
    for g in factors:
        u = np.unique(g)
        oh = (g[:, None] == u[None, :]).astype(np.float32)
        cols.append(oh[:, 1:])
    A = np.concatenate(cols + [np.ones((len(R), 1), np.float32)], 1)
    coef, *_ = np.linalg.lstsq(A, R, rcond=None)
    pred = A @ coef
    return float(((pred - R.mean(0)) ** 2).sum() / ((R - R.mean(0)) ** 2).sum())


def pr(w):
    w = np.clip(w, 0, None)
    return float(w.sum() ** 2 / (w ** 2).sum())


def main():
    d = load_baseline()
    tr = d["train"]
    meta = {}
    for l in open(META.format("train")):
        r = json.loads(l)
        meta[r["source_stem"]] = r
    m = [meta[i] for i in tr["ids"]]
    dur = np.array([x["duration_sec"] for x in m])
    fps = np.array([round(x["fps"]) for x in m])
    res_ = np.array([f'{x["width"]}x{x["height"]}' for x in m])
    nf = np.array([x["num_frames"] for x in m])
    qd = np.quantile(dur, np.linspace(0, 1, 11)[1:-1])
    dbin = np.digitize(dur, qd)
    X, y, sg = tr["X"], tr["y"], tr["signer"]
    X = l2(X)  # the protocol L2-normalises; analysis is on the normalised clip embedding
    gm = np.zeros((y.max() + 1, X.shape[1]))
    cnt = np.bincount(y, minlength=y.max() + 1)
    np.add.at(gm, y, X)
    gm = gm / np.maximum(cnt, 1)[:, None]
    R = X - gm[y]
    tot = ((X - X.mean(0)) ** 2).sum()
    out = {
        "n": int(len(X)),
        "within_gloss_share_of_total_variance": float((R ** 2).sum() / tot),
        "eta2_signer_of_total": float(eta2(X - X.mean(0), sg)),
        "eta2_gloss_of_total": 1 - float((R ** 2).sum() / tot),
        "within_gloss_eta2": {
            "signer": float(eta2(R, sg)),
            "duration_decile": float(eta2(R, dbin)),
            "fps": float(eta2(R, fps)),
            "resolution": float(eta2(R, res_)),
        },
        "within_gloss_joint_R2_signer+duration+fps+resolution": r2_joint(R, [sg, dbin, fps, res_]),
        "within_gloss_joint_R2_duration+fps+resolution": r2_joint(R, [dbin, fps, res_]),
        "fps_counts": {str(k): int(v) for k, v in zip(*np.unique(fps, return_counts=True))},
        "res_counts": {k: int(v) for k, v in zip(*np.unique(res_, return_counts=True))},
    }
    # signer share inside the within-gloss residual after removing duration/fps/res first (incremental)
    out["within_gloss_signer_incremental_over_dur_fps_res"] = out["within_gloss_joint_R2_signer+duration+fps+resolution"] - out[
        "within_gloss_joint_R2_duration+fps+resolution"]

    # dimensionality & alignment of within- vs between-gloss covariance
    Sw = np.cov(R.T)
    gmc = gm[cnt > 0]
    Sb = np.cov(gmc.T)
    ww, Vw = np.linalg.eigh(Sw)
    wb, Vb = np.linalg.eigh(Sb)
    ww, Vw, wb, Vb = ww[::-1], Vw[:, ::-1], wb[::-1], Vb[:, ::-1]
    out["participation_ratio"] = {"within_gloss_Sw": pr(ww), "between_gloss_Sb": pr(wb)}
    out["trace"] = {"Sw": float(ww.sum()), "Sb": float(wb.sum())}
    ov = {}
    for k in (10, 30, 100):
        ov[str(k)] = float((Vw[:, :k].T @ Vb[:, :k]) .__pow__(2).sum() / k)  # mean cos^2 between the two k-dim top subspaces
    out["top_subspace_overlap_Sw_vs_Sb"] = ov
    # what fraction of Sb variance lies in the top-k Sw directions (is gloss signal hidden in the nuisance subspace?)
    out["Sb_variance_in_top_Sw_dirs"] = {str(k): float(np.trace(Vw[:, :k].T @ Sb @ Vw[:, :k]) / np.trace(Sb)) for k in (10, 30, 100)}
    out["Sw_variance_in_top_Sw_dirs"] = {str(k): float(ww[:k].sum() / ww.sum()) for k in (10, 30, 100)}
    # between-signer (gloss-residualised) subspace: share of Sw it spans, and Sb it hides
    smeans = np.stack([R[sg == s].mean(0) for s in np.unique(sg)])
    U, s_, _ = np.linalg.svd(smeans - smeans.mean(0), full_matrices=False)
    P = U.T  # placeholder to keep names short
    Vs = np.linalg.svd((smeans - smeans.mean(0)).T, full_matrices=False)[0][:, :34]  # (d, <=34)
    out["signer_subspace_34d"] = {
        "share_of_Sw_trace": float(np.trace(Vs.T @ Sw @ Vs) / np.trace(Sw)),
        "share_of_Sb_trace": float(np.trace(Vs.T @ Sb @ Vs) / np.trace(Sb)),
    }
    # neighbour statistics on a random subset of train (top-1 non-self neighbour)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=8000, replace=False)
    Xn = l2(X)
    S = Xn[idx] @ Xn.T
    S[np.arange(len(idx)), idx] = -2
    nn = S.argmax(1)
    out["nn_top1_same_signer"] = float((sg[nn] == sg[idx]).mean())
    out["nn_top1_same_gloss"] = float((y[nn] == y[idx]).mean())
    p = np.bincount(np.unique(sg, return_inverse=True)[1]) / len(sg)
    out["chance_same_signer"] = float((p ** 2).sum())
    # same, but restricting to neighbours of a different signer (cross-signer retrieval inside train)
    S2 = Xn[idx] @ Xn.T
    S2[sg[idx][:, None] == sg[None, :]] = -2
    nn2 = S2.argmax(1)
    out["cross_signer_nn_top1_same_gloss"] = float((y[nn2] == y[idx]).mean())
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
