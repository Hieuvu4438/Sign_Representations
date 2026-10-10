"""EX10-v1 (LOCKED): within-test leave-one-signer-out on the fixed ASL Citizen test half.
SHuBERT residual layer-12 readouts S1/S2/S3 and SignRep R0/R1/R2 on the same clips."""
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


def loso_ranks(F, y, sg, ncls):
    r = np.zeros(len(y), int)
    for s in np.unique(sg):
        q, g = sg == s, sg != s
        r[q] = ranks_from_scores(class_max_scores(F[q], F[g], y[g], ncls), y[q])
    return r


def loso_chamfer(sq, base_vec, y, sg, ncls):
    X, M = pad(sq, 48)
    Xt, Mt = torch.from_numpy(X.astype(np.float16)), torch.from_numpy(M)
    out = np.zeros(len(y), int)
    for s in np.unique(sg):
        q_idx, g_idx = np.where(sg == s)[0], np.where(sg != s)[0]
        base = class_max_scores(base_vec[q_idx], base_vec[g_idx], y[g_idx], ncls)
        for j, i in enumerate(q_idx):
            top = np.argsort(-base[j])[:20]; items = g_idx[np.isin(y[g_idx], top)]
            sc = seq_scores(Xt[i].cuda(), Mt[i].cuda(), Xt[items].cuda(), Mt[items].cuda(), "chamfer").float().cpu().numpy()
            new = np.full(ncls, -1e9, np.float32); np.maximum.at(new, y[items], sc)
            out[i] = ranks_from_scores(np.where(new > -1e8, 10 + new, base[j])[None], [y[i]])[0]
    return out


def main():
    keep = [l.strip() for l in open(RT.parent / "configs" / "test_half.txt") if l.strip()]
    # SHuBERT
    D = RT / "feats" / "shubert_res_test_half"
    sc = [json.loads(l) for l in open(D / "clips.jsonl")]
    assert all(c["split"] == "test" for c in sc) and len(sc) == len(keep)
    ok = np.array([c["status"] == "ok" for c in sc])
    off = np.load(D / "offsets.npy"); Fr = np.load(D / "frames_l12.npy"); V = np.load(D / "frame_valid.npy")
    gl_ = sorted({c["gloss"] for c in sc}); lab = {g: i for i, g in enumerate(gl_)}; ncls = len(gl_)
    S1, S2, sq = [], [], []
    for i in range(len(sc)):
        f = Fr[off[i]:off[i + 1]].astype(np.float32); v = V[off[i]:off[i + 1]]
        if len(f) == 0:
            f = np.zeros((1, Fr.shape[1]), np.float32); v = np.ones(1, bool)
        idx = np.where(v)[0]; a, b = (idx[0], idx[-1] + 1) if len(idx) else (0, len(f))
        g = f[a:b]; S1.append(g.mean(0)); h = max(1, len(g) // 2)
        S2.append(np.concatenate([l2n(g[:h].mean(0, keepdims=True))[0], l2n(g[h:].mean(0, keepdims=True) if len(g) > 1 else g.mean(0, keepdims=True))[0]]) / np.sqrt(2))
        w = g[::2]
        if len(w) > 48: w = w[np.round(np.linspace(0, len(w) - 1, 48)).astype(int)]
        sq.append((l2n(w), None))
    S1, S2 = l2n(np.stack(S1)), np.stack(S2).astype(np.float32)
    y = np.array([lab[c["gloss"]] for c in sc]); sg = np.array([c["signer"] for c in sc])
    r = {"S1": loso_ranks(S1, y, sg, ncls), "S2": loso_ranks(S2, y, sg, ncls)}
    r["S3"] = loso_chamfer(sq, S1, y, sg, ncls)
    # SignRep on the same clips
    gc, goff, gw, ga = load(RT / "feats" / "aslc_test_s2")
    pos = {c["sample_id"]: i for i, c in enumerate(gc)}
    sel = np.array([pos[c["sample_id"]] for c in sc])
    R0all = l2n(pool_clips(gw, goff, activity_weights(ga))[0])
    sqall = seqs(goff, gw, ga)
    sqs = [sqall[i] for i in sel]
    r["R0"] = loso_ranks(R0all[sel], y, sg, ncls)
    r["R1"] = loso_ranks(binned(sqs, 2), y, sg, ncls)
    wm = l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in sqs]))
    r["R2"] = loso_chamfer(sqs, wm, y, sg, ncls)
    out = {"n": len(y), "shubert_failed_clips": int((~ok).sum()), "metrics": {k: metrics_from_ranks(v, y) for k, v in r.items()}}
    comps = {}
    for a, b in (("S1", "S2"), ("S1", "S3"), ("R0", "R1"), ("R0", "R2")):
        pa, pb = per_query(r[a])["R@1"], per_query(r[b])["R@1"]
        bs = paired_cluster_bootstrap(pa, pb, sg, n_boot=10000)
        comps[f"{b}-{a}"] = {"diff_R@1": bs["diff"], "ci95": bs["ci95"], "p": boot_p(pa, pb, sg)[0],
                             "pos": f"{bs['n_clusters_positive']}/{bs['n_clusters']}"}
    adj = holm([comps["S2-S1"]["p"], comps["S3-S1"]["p"]])
    comps["S2-S1"]["p_holm"], comps["S3-S1"]["p_holm"] = float(adj[0]), float(adj[1])
    c = comps["S2-S1"]
    out["verdict_shubert_S2"] = ("SUPPORTED_WITHIN_SCOPE" if c["diff_R@1"] > 0 and c["ci95"][0] > 0 and c["p_holm"] < 0.05
                                 else ("REFUTED" if c["diff_R@1"] <= 0 else "INCONCLUSIVE"))
    out["comparisons"] = comps
    for k, v in r.items():
        np.save(RT / "eval" / f"EX10_ranks_{k}.npy", v)
    json.dump(out, open(RT / "eval" / "EX10_result.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
