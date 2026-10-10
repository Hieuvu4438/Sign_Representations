"""EX07-v1 (LOCKED): MS-ASL test — R0, R1, R1W, R2, R2W and confirmatory statistics."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import (activity_weights, class_max_scores, l2n, metrics_from_ranks, paired_cluster_bootstrap,  # noqa
                          per_query, pool_clips, ranks_from_scores)
from diag_order import binned, load, pad, seq_scores, seqs  # noqa: E402
from ex05_analysis import boot_p, holm  # noqa: E402

RT = Path(__file__).resolve().parents[1] / "runtime"
K, EPS = 512, 0.01


def fitw(X):
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    s = S / np.sqrt(len(X))
    return lambda Z: ((Z - mu) @ Vt[:K].T) / (s[:K] + EPS * s[0])


def wmean(sq):
    return l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in sq]))


def chamfer_ranks(gs, qs, gy, qy, ncls, GWm, QWm):
    gX, gM = pad(gs, 48); qX, qM = pad(qs, 48)
    gXt, gMt = torch.from_numpy(gX.astype(np.float16)), torch.from_numpy(gM)
    out = []
    for i in range(len(qy)):
        base = class_max_scores(QWm[i:i + 1], GWm, gy, ncls, device="cuda")[0]
        top = np.argsort(-base)[:20]; items = np.where(np.isin(gy, top))[0]
        sc = seq_scores(torch.from_numpy(qX[i]).cuda().half(), torch.from_numpy(qM[i]).cuda(),
                        gXt[items].cuda(), gMt[items].cuda(), "chamfer").float().cpu().numpy()
        new = np.full(ncls, -1e9, np.float32); np.maximum.at(new, gy[items], sc)
        out.append(ranks_from_scores(np.where(new > -1e8, 10 + new, base)[None], [qy[i]])[0])
    return np.array(out)


def main():
    gc, goff, gw, ga = load(RT / "feats" / "msasl_train_s2")
    qc, qoff, qw, qa = load(RT / "feats" / "msasl_test_s2")
    assert all(c["split"] == "test" for c in qc) and all(c["split"] == "train" for c in gc)
    gl_ = sorted({c["gloss"] for c in gc}); lab = {g: i for i, g in enumerate(gl_)}
    gy = np.array([lab[c["gloss"]] for c in gc]); qy = np.array([lab[c["gloss"]] for c in qc])
    ncls = len(gl_)
    r = {}
    G0 = l2n(pool_clips(gw, goff, activity_weights(ga))[0]); Q0 = l2n(pool_clips(qw, qoff, activity_weights(qa))[0])
    r["R0"] = ranks_from_scores(class_max_scores(Q0, G0, gy, ncls), qy)
    gs, qs = seqs(goff, gw, ga), seqs(qoff, qw, qa)
    GB, QB = binned(gs, 2), binned(qs, 2)
    r["R1"] = ranks_from_scores(class_max_scores(QB, GB, gy, ncls), qy)
    d = GB.shape[1] // 2
    f1, f2 = fitw(GB[:, :d]), fitw(GB[:, d:])
    wb = lambda B: l2n(np.concatenate([l2n(f1(B[:, :d])), l2n(f2(B[:, d:]))], 1)).astype(np.float32)
    r["R1W"] = ranks_from_scores(class_max_scores(wb(QB), wb(GB), gy, ncls), qy)
    print("single-vector done", {k: metrics_from_ranks(v)["R@1"] for k, v in r.items()}, flush=True)
    r["R2"] = chamfer_ranks(gs, qs, gy, qy, ncls, wmean(gs), wmean(qs))
    rng = np.random.default_rng(0)
    allw = np.concatenate([w for w, _ in gs])
    fw = fitw(allw[rng.choice(len(allw), min(300000, len(allw)), replace=False)])
    gs2 = [(l2n(fw(w)).astype(np.float32), g) for w, g in gs]; qs2 = [(l2n(fw(w)).astype(np.float32), g) for w, g in qs]
    r["R2W"] = chamfer_ranks(gs2, qs2, gy, qy, ncls, wmean(gs2), wmean(qs2))
    out = {"metrics": {k: metrics_from_ranks(v, qy) for k, v in r.items()}, "n_gallery": len(gy), "n_query": len(qy)}
    comps = {}
    for a, b in (("R0", "R1"), ("R1", "R1W"), ("R0", "R2"), ("R2", "R2W"), ("R0", "R1W")):
        pa, pb = per_query(r[a])["R@1"], per_query(r[b])["R@1"]
        bs = paired_cluster_bootstrap(pa, pb, qy, n_boot=10000)
        comps[f"{b}-{a}"] = {"diff_R@1": bs["diff"], "ci95": bs["ci95"], "p": boot_p(pa, pb, qy)[0]}
    adj = holm([comps["R1-R0"]["p"], comps["R1W-R1"]["p"]])
    comps["R1-R0"]["p_holm"], comps["R1W-R1"]["p_holm"] = float(adj[0]), float(adj[1])
    out["comparisons"] = comps
    hr = comps["R1-R0"]; hw = comps["R1W-R1"]
    out["verdict_H-R"] = "SUPPORTED_WITHIN_SCOPE" if hr["diff_R@1"] > 0 and hr["ci95"][0] > 0 and hr["p_holm"] < 0.05 else ("REFUTED" if hr["diff_R@1"] <= 0 else "INCONCLUSIVE")
    out["verdict_H-W"] = ("SUPPORTED_WITHIN_SCOPE" if hw["diff_R@1"] >= 0.01 and hw["ci95"][0] > 0 and hw["p_holm"] < 0.05
                          else ("REFUTED" if hw["diff_R@1"] <= 0 else "INCONCLUSIVE"))
    for k, v in r.items():
        np.save(RT / "eval" / f"EX07_ranks_{k}.npy", v)
    json.dump(out, open(RT / "eval" / "EX07_result.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
