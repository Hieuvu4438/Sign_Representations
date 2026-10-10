"""EX15 evaluation on ASL Citizen VAL (no test access).

1. Retrieval: leave-one-signer-out inside val (queries = one signer, gallery = the other val signers,
   class-max cosine), clip embedding = mean of window features (R0-like) and two halves (bins2).
2. Pose-part probes ("what did it learn about body parts and their relations"): window-level ridge
   regression from frozen window features to window-averaged prior targets, signer-disjoint 2-fold over
   val signers; R^2 per sub-group: handshape (hand joints rel. wrist + finger angles + fingertip
   distances), body (upper-body keypoints + elbows), location/relations (REL), non-manual (NM), movement (MOT).
3. Signer probe (logistic regression on clip embeddings, gloss-disjoint split) and geometry (mean cosine,
   effective rank) of clip embeddings.
Feature sets: runtime/feats/<name> (jt/extract.py) or 'signrep' (released SignRep windows from the p4
cache, stride-2 windows subsampled to stride 8).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[0] / "p4_signssl"))
from jt.data import PriorStats, load_kp  # noqa: E402
from jt.priors import DIMS, compute  # noqa: E402
from p4.retrieval import class_max_scores, l2n, metrics_from_ranks, paired_cluster_bootstrap, per_query, ranks_from_scores  # noqa: E402

RT = ROOT / "runtime"
P4 = ROOT.parents[0] / "p4_signssl" / "runtime" / "feats" / "aslc_val_s2"
SUB = {  # column ranges inside each group (see jt/priors.py layout)
    "handshape": ("POSE", np.r_[26:26 + 82, 26 + 82:26 + 164]),
    "body": ("POSE", np.r_[0:26, 190:194]),
    "location_relations": ("REL", np.arange(17)),
    "non_manual": ("NM", np.arange(65)),
    "movement": ("MOT", np.arange(14)),
}


def load_feats(name):
    if name == "signrep":
        off = np.load(P4 / "offsets.npy"); W = np.load(P4 / "windows_features.npy")
        clips = [json.loads(l) for l in open(P4 / "clips.jsonl")]
        win, offs, starts = [], [0], []
        for i in range(len(off) - 1):
            st = 2 * np.arange(off[i + 1] - off[i])
            keep = np.where(st % 8 == 0)[0]
            win.append(W[off[i] + keep]); starts += list(st[keep]); offs.append(offs[-1] + len(keep))
        return clips, np.concatenate(win).astype(np.float32), np.array(offs), np.array(starts)
    d = RT / "feats" / name
    clips = [json.loads(l) for l in open(d / "clips.jsonl")]
    return clips, np.load(d / "windows.npy").astype(np.float32), np.load(d / "offsets.npy"), np.load(d / "starts.npy")


def window_priors(clips, offs, starts, cache):
    if cache.exists():
        z = np.load(cache)
        if len(z["starts"]) == len(starts) and (z["starts"] == starts).all():
            return {g: (z[g], z[g + "_m"]) for g in DIMS}
    stats = PriorStats(RT / "prior_stats_aslc_train.npz")
    acc = {g: ([], []) for g in DIMS}
    for i, c in enumerate(clips):
        K, Sc = load_kp(c["sample_id"])
        pr = compute(K, Sc, list(DIMS))
        for s in starts[offs[i]:offs[i + 1]]:
            for g, (v, m) in pr.items():
                vv, mm = v[s:s + 16], m[s:s + 16]
                cnt = mm.sum(0)
                acc[g][0].append(stats(g, (vv * mm).sum(0) / np.maximum(cnt, 1)) if len(vv) else np.zeros(DIMS[g], np.float32))
                acc[g][1].append(cnt > 0 if len(vv) else np.zeros(DIMS[g], bool))
    out = {g: (np.stack(v), np.stack(m)) for g, (v, m) in acc.items()}
    np.savez(cache, starts=starts, **{g: out[g][0] for g in DIMS}, **{g + "_m": out[g][1] for g in DIMS})
    return out


def pool(win, offs, mode):
    out = []
    for i in range(len(offs) - 1):
        w = l2n(win[offs[i]:offs[i + 1]])
        if mode == "mean":
            out.append(w.mean(0))
        else:
            h = max(1, len(w) // 2)
            out.append(np.concatenate([l2n(w[:h].mean(0, keepdims=True))[0], l2n(w[len(w) - h:].mean(0, keepdims=True))[0]]))
    return l2n(np.stack(out))


def whiten(win, k=512, seed=0):
    """EX15-v1.2: label-free PCA whitening fit on the evaluated split's windows (every 7th window)."""
    S = win[::7]; mu = S.mean(0)
    _, s, Vt = np.linalg.svd(S - mu, full_matrices=False); s = s / np.sqrt(len(S))
    return ((win - mu) @ Vt[:k].T) / (s[:k] + 1e-3 * s[0])


def loso(F, y, sg, ncls):
    r = np.zeros(len(y), int)
    for s in np.unique(sg):
        q, g = sg == s, sg != s
        r[q] = ranks_from_scores(class_max_scores(F[q], F[g], y[g], ncls), y[q])
    return r


def probes(win, wsg, pri, max_n=30000, seed=0, alphas=(1e2, 1e3, 1e4, 1e5)):
    """EX15-v1.1: alpha per model and sub-group chosen by inner leave-one-signer-out CV inside the
    training fold (signer-disjoint outer 2-fold)."""
    rng = np.random.default_rng(seed)
    sig = np.unique(wsg); rng.shuffle(sig)
    folds = [sig[: len(sig) // 2], sig[len(sig) // 2:]]
    X = (win - win.mean(0)) / (win.std(0) + 1e-6)

    def r2(Y, P):
        return 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()

    res = {}
    for name, (g, cols) in SUB.items():
        Y, M = pri[g][0][:, cols], pri[g][1][:, cols]
        ok = M.all(1)
        num, den, chosen = 0.0, 0.0, []
        for k in range(2):
            tr = np.where(np.isin(wsg, folds[k]) & ok)[0]; te = np.where(~np.isin(wsg, folds[k]) & ok)[0]
            if len(tr) > max_n:
                tr = rng.choice(tr, max_n, replace=False)
            if len(te) > max_n:
                te = rng.choice(te, max_n, replace=False)
            score = {al: 0.0 for al in alphas}
            for s in folds[k]:  # inner leave-one-signer-out
                itr, ite = tr[wsg[tr] != s], tr[wsg[tr] == s]
                if len(ite) == 0 or len(itr) == 0:
                    continue
                for al in alphas:
                    score[al] += r2(Y[ite], Ridge(alpha=al).fit(X[itr], Y[itr]).predict(X[ite]))
            best = max(score, key=score.get); chosen.append(best)
            P = Ridge(alpha=best).fit(X[tr], Y[tr]).predict(X[te])
            num += ((Y[te] - P) ** 2).sum(); den += ((Y[te] - Y[te].mean(0)) ** 2).sum()
        res[name] = float(1 - num / den); res[name + "_alpha"] = chosen
    return res


def signer_probe(F, sg, gl, seed=0):
    rng = np.random.default_rng(seed)
    g = np.unique(gl); rng.shuffle(g)
    tr = np.isin(gl, g[: len(g) // 2])
    mdl = LogisticRegression(max_iter=2000, C=1.0).fit(F[tr], sg[tr])
    return float((mdl.predict(F[~tr]) == sg[~tr]).mean())


def geometry(F):
    Fc = F - F.mean(0)
    s = np.linalg.svd(Fc, compute_uv=False); p = s / s.sum()
    idx = np.random.default_rng(0).choice(len(F), min(3000, len(F)), replace=False)
    C = F[idx] @ F[idx].T
    return {"mean_cos": float(C[~np.eye(len(idx), dtype=bool)].mean()), "eff_rank": float(np.exp(-(p * np.log(p + 1e-12)).sum()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True, help="comma list")
    ap.add_argument("--ref", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res, ranks, sgs = {}, {}, None
    for name in a.feats.split(","):
        clips, win, offs, starts = load_feats(name)
        assert all(c["split"] == "val" for c in clips)
        gl_ = sorted({c["gloss"] for c in clips}); lab = {g: i for i, g in enumerate(gl_)}
        y = np.array([lab[c["gloss"]] for c in clips]); sg = np.array([c["signer"] for c in clips]); sgs = sg
        r = {}
        Ww = whiten(win)
        for mode, src, pm in (("mean", win, "mean"), ("bins2", win, "bins2"), ("white512", Ww, "mean"), ("white512_bins2", Ww, "bins2")):
            F = pool(src, offs, pm)
            rk = loso(F, y, sg, len(gl_)); r[mode] = metrics_from_ranks(rk, y)
            if mode == "mean":
                Fm = F
            if mode == "white512":
                ranks[name] = rk
        cache = RT / "feats" / ("_priors_signrep.npz" if name == "signrep" else "_priors_val_s8.npz")
        pri = window_priors(clips, offs, starts, cache)
        wsg = np.repeat(sg, np.diff(offs))
        r["probe_R2"] = probes(win, wsg, pri)
        r["signer_probe_acc"] = signer_probe(Fm, sg, np.array([c["gloss"] for c in clips]))
        r["geometry"] = geometry(Fm)
        res[name] = r
        print(name, json.dumps({"R@1_white512": r["white512"]["R@1"], "R@1_white512_bins2": r["white512_bins2"]["R@1"], "R@1_mean": r["mean"]["R@1"], "R@1_bins2": r["bins2"]["R@1"], **{k: v for k, v in r["probe_R2"].items() if not k.endswith("alpha")}, "signer": r["signer_probe_acc"], **r["geometry"]}), flush=True)
    if a.ref and a.ref in ranks:
        for name, rk in ranks.items():
            if name != a.ref:
                bs = paired_cluster_bootstrap(per_query(ranks[a.ref])["R@1"], per_query(rk)["R@1"], sgs, n_boot=10000)
                res[f"{name}-{a.ref}"] = {"diff": bs["diff"], "ci95": bs["ci95"], "pos": f"{bs['n_clusters_positive']}/{bs['n_clusters']}"}
    json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
