"""Control for H010/H011 (exploratory, val only): SemanticMoments-style training-free readouts
(temporal moments of frozen SignRep active-span windows; Huberman et al. 2026, arXiv 2602.09146)
vs the temporal-difference chamfer readouts T0/T1/T4. Same 3,000-query val subset and same
wmean top-20 class rerank as explore_teachers.py (identical rng call sequence).
Readouts (each block L2-normalised, then concatenated and L2-normalised):
  M1 mean | M2 mean,std | M3 mean,std,3rd moment | D2 mean,std of temporal differences (lag 4)
  M2D2 = M2 + D2 blocks.  Each scored globally (class-max over the full train gallery) and as a
  top-20 rerank of the wmean shortlist (comparable to the chamfer readouts).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import class_max_scores, l2n, metrics_from_ranks, ranks_from_scores  # noqa: E402
from diag_order import load, pad, seqs  # noqa: E402
from explore_teachers import K_DIFF, q2g  # noqa: E402

RT = Path(__file__).resolve().parents[1] / "runtime"


def tdiff(w):
    return l2n(w[K_DIFF:] - w[:-K_DIFF]) if len(w) > K_DIFF else l2n(w[-1:] - w[:1] + 1e-6)


def mom(w):
    mu = w.mean(0); c = w - mu
    sd = np.sqrt((c ** 2).mean(0) + 1e-8)
    m3 = np.cbrt((c ** 3).mean(0))
    return mu, sd, m3


def readouts(sq):
    out = {k: [] for k in ("M1", "M2", "M3", "D2", "M2D2")}
    for w, _ in sq:
        mu, sd, m3 = mom(w)
        dmu, dsd, _ = mom(tdiff(w))
        b = lambda *v: l2n(np.concatenate([l2n(x[None])[0] for x in v])[None])[0]
        out["M1"].append(l2n(mu[None])[0]); out["M2"].append(b(mu, sd)); out["M3"].append(b(mu, sd, m3))
        out["D2"].append(b(dmu, dsd)); out["M2D2"].append(b(mu, sd, dmu, dsd))
    return {k: np.stack(v).astype(np.float32) for k, v in out.items()}


def main():
    gc, goff, gw, ga = load(RT / "feats" / "aslc_train_s2"); qc, qoff, qw, qa = load(RT / "feats" / "aslc_val_s2")
    gs, qs = seqs(goff, gw, ga), seqs(qoff, qw, qa)
    rng = np.random.default_rng(0)
    allw_n = sum(len(w) for w, _ in gs)
    rng.choice(allw_n, 300000, replace=False); rng.choice(len(gs), 3000, replace=False)  # replay explore_teachers rng
    sub = rng.choice(len(qs), 3000, replace=False)
    wm = lambda sq: l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in sq]))
    GW, QW = wm(gs), wm(qs)
    GR, QR = readouts(gs), readouts(qs)
    gl = np.array([c["gloss"] for c in gc]); labs = sorted(set(gl)); lab = {g: i for i, g in enumerate(labs)}
    gy = np.array([lab[x] for x in gl]); qy = np.array([lab[c["gloss"]] for c in qc])
    res = {"global": {}, "rerank20": {}}
    for k in GR:
        sc = class_max_scores(QR[k][sub], GR[k], gy, len(labs), device="cuda")
        res["global"][k] = metrics_from_ranks(ranks_from_scores(sc, qy[sub]))["R@1"]
    base = class_max_scores(QW[sub], GW, gy, len(labs), device="cuda")
    PADs = {}
    for k, f in (("T0", lambda w: w), ("T1", tdiff)):
        X, M = pad([(f(w), None) for w, _ in gs], 48); PADs[k] = (torch.from_numpy(X.astype(np.float16)), torch.from_numpy(M))
        X, M = pad([(f(w), None) for w, _ in qs], 48); PADs["q" + k] = (torch.from_numpy(X.astype(np.float16)), torch.from_numpy(M))
    rr = {k: [] for k in list(GR) + ["T0", "T1", "T4", "T4+M2D2"]}
    Gt = {k: torch.from_numpy(v).cuda() for k, v in GR.items()}
    for j, i in enumerate(sub):
        top = np.argsort(-base[j])[:20]; items = np.where(np.isin(gy, top))[0]
        sims = {k: (Gt[k][items] @ torch.from_numpy(QR[k][i]).cuda()).float().cpu().numpy() for k in GR}
        for k in ("T0", "T1"):
            gx, gm = PADs[k]; qx, qm = PADs["q" + k]
            sims[k] = q2g(qx[i].cuda(), qm[i].cuda(), gx[items].cuda(), gm[items].cuda()).float().cpu().numpy()
        sims["T4"] = sims["T0"] + sims["T1"]; sims["T4+M2D2"] = sims["T4"] + sims["M2D2"]
        for k, v in sims.items():
            new = np.full(len(labs), -1e9, np.float32); np.maximum.at(new, gy[items], v)
            rr[k].append(ranks_from_scores(np.where(new > -1e8, 10 + new, base[j])[None], [qy[i]])[0])
    for k, v in rr.items():
        res["rerank20"][k] = metrics_from_ranks(np.array(v))["R@1"]
    for s in res:
        for k, v in res[s].items():
            print(f"{s:9s} {k:8s} val R@1 {v:.4f}", flush=True)
    json.dump(res, open(RT / "explore" / "moments_control.json", "w"), indent=1)


if __name__ == "__main__":
    main()
