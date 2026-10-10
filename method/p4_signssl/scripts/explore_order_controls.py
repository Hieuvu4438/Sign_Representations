"""R002/E1 (val, official gallery; exploratory): is the R1 gain about ORDER or about having two
vectors / 2x dimensions? All on the same frozen SignRep windows (active span, γ>=0.5).
  R1        ordered halves, cosine of concatenation
  R1-swapinv order-invariant 2-segment: max over the two half-alignments
  K2means   order-free 2-means clusters of the span windows, matched order-invariantly
  mean+max  1536-d order-free concatenation [wmean ; max]
  mean+std  1536-d order-free concatenation [wmean ; std]
  R1-reverseq ordered halves with the QUERY's halves swapped (order violated)
Also: reversed-query R0/R1/R2 on the 2000-query subset (aslc_val2000_reverse)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import activity_weights, class_max_scores, l2n, metrics_from_ranks, paired_cluster_bootstrap, per_query, pool_clips, ranks_from_scores  # noqa
from diag_order import binned, load, pad, seq_scores, seqs  # noqa: E402

RT = Path(__file__).resolve().parents[1] / "runtime"


def halves(sq):
    B = binned(sq, 2); d = B.shape[1] // 2
    return l2n(B[:, :d]), l2n(B[:, d:])


def kmeans2(sq, iters=10):
    A, B = [], []
    for w, g in sq:
        if len(w) < 2:
            A.append(w[0]); B.append(w[0]); continue
        c = w[[0, len(w) - 1]].copy()
        for _ in range(iters):
            lab = (w @ c.T).argmax(1)
            for k in (0, 1):
                if (lab == k).any():
                    c[k] = w[lab == k].mean(0)
        A.append(c[0]); B.append(c[1])
    return l2n(np.stack(A)), l2n(np.stack(B))


@torch.no_grad()
def swapinv_scores(qa, qb, ga, gb, gy, ncls, chunk=1024):
    Ga, Gb = torch.from_numpy(ga).cuda(), torch.from_numpy(gb).cuda(); gl = torch.from_numpy(gy).cuda()
    out = []
    for i in range(0, len(qa), chunk):
        Qa, Qb = torch.from_numpy(qa[i:i + chunk]).cuda(), torch.from_numpy(qb[i:i + chunk]).cuda()
        s = torch.maximum(Qa @ Ga.T + Qb @ Gb.T, Qa @ Gb.T + Qb @ Ga.T) / 2
        cs = torch.full((len(Qa), ncls), -1e9, device="cuda")
        cs.scatter_reduce_(1, gl.expand(len(Qa), -1), s, reduce="amax", include_self=True)
        out.append(cs.cpu())
    return torch.cat(out).numpy()


def main():
    gc, goff, gw, ga = load(RT / "feats" / "aslc_train_s2"); qc, qoff, qw, qa = load(RT / "feats" / "aslc_val_s2")
    gl_ = sorted({c["gloss"] for c in gc}); lab = {g: i for i, g in enumerate(gl_)}; ncls = len(gl_)
    gy = np.array([lab[c["gloss"]] for c in gc]); qy = np.array([lab[c["gloss"]] for c in qc]); sg = np.array([c["signer"] for c in qc])
    gs, qs = seqs(goff, gw, ga), seqs(qoff, qw, qa)
    r = {}
    r["R0"] = ranks_from_scores(class_max_scores(l2n(pool_clips(qw, qoff, activity_weights(qa))[0]), l2n(pool_clips(gw, goff, activity_weights(ga))[0]), gy, ncls), qy)
    GB, QB = binned(gs, 2), binned(qs, 2)
    r["R1"] = ranks_from_scores(class_max_scores(QB, GB, gy, ncls), qy)
    d = GB.shape[1] // 2
    QBs = np.concatenate([QB[:, d:], QB[:, :d]], 1)
    r["R1-reverseq"] = ranks_from_scores(class_max_scores(QBs, GB, gy, ncls), qy)
    ga1, ga2 = halves(gs); qa1, qa2 = halves(qs)
    r["R1-swapinv"] = ranks_from_scores(swapinv_scores(qa1, qa2, ga1, ga2, gy, ncls), qy)
    gk1, gk2 = kmeans2(gs); qk1, qk2 = kmeans2(qs)
    r["K2means-swapinv"] = ranks_from_scores(swapinv_scores(qk1, qk2, gk1, gk2, gy, ncls), qy)
    wm = lambda sq: l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in sq]))
    mx = lambda sq: l2n(np.stack([w.max(0) for w, _ in sq]))
    sd = lambda sq: l2n(np.stack([w.std(0) if len(w) > 1 else np.zeros(w.shape[1], np.float32) + 1e-6 for w, _ in sq]))
    GW, QW = wm(gs), wm(qs)
    r["wmean_span"] = ranks_from_scores(class_max_scores(QW, GW, gy, ncls), qy)
    r["mean+max"] = ranks_from_scores(class_max_scores(l2n(np.concatenate([QW, mx(qs)], 1)), l2n(np.concatenate([GW, mx(gs)], 1)), gy, ncls), qy)
    r["mean+std"] = ranks_from_scores(class_max_scores(l2n(np.concatenate([QW, sd(qs)], 1)), l2n(np.concatenate([GW, sd(gs)], 1)), gy, ncls), qy)
    res = {"official_gallery": {}}
    for k, v in r.items():
        m = metrics_from_ranks(v, qy)
        b = paired_cluster_bootstrap(per_query(r["R0"])["R@1"], per_query(v)["R@1"], sg) if k != "R0" else None
        res["official_gallery"][k] = {"R@1": m["R@1"], "MRR": m["MRR"], "R@1_classmacro": m["R@1_classmacro"],
                                      "dR1_vs_R0": None if b is None else b["diff"], "ci": None if b is None else b["ci95"]}
        print(f"{k:16s} R@1 {m['R@1']:.4f}" + ("" if b is None else f"  vs R0 {b['diff']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}]"), flush=True)
    # reversed queries (2000 subset) under R0 / R1 / R2
    sub = [json.loads(l)["sample_id"] for l in open(RT / "feats" / "aslc_val2000_reverse" / "clips.jsonl")]
    idx = {c["sample_id"]: i for i, c in enumerate(qc)}; qi = np.array([idx[s] for s in sub])
    rc, roff, rw, ra = load(RT / "feats" / "aslc_val2000_reverse"); rs = seqs(roff, rw, ra)
    G0 = l2n(pool_clips(gw, goff, activity_weights(ga))[0])
    out = {}
    for tag, Qsq, Q0 in (("normal", [qs[i] for i in qi], l2n(pool_clips(qw, qoff, activity_weights(qa))[0])[qi]),
                         ("reversed", rs, l2n(pool_clips(rw, roff, activity_weights(ra))[0]))):
        yq = qy[qi]
        r0 = ranks_from_scores(class_max_scores(Q0, G0, gy, ncls), yq)
        r1 = ranks_from_scores(class_max_scores(binned(Qsq, 2), GB, gy, ncls), yq)
        QWs = wm(Qsq); gX, gM = pad(gs, 48); qX, qM = pad(Qsq, 48)
        gXt, gMt = torch.from_numpy(gX.astype(np.float16)), torch.from_numpy(gM)
        r2 = []
        for j in range(len(Qsq)):
            base = class_max_scores(QWs[j:j + 1], GW, gy, ncls, device="cuda")[0]; top = np.argsort(-base)[:20]; items = np.where(np.isin(gy, top))[0]
            scs = seq_scores(torch.from_numpy(qX[j]).cuda().half(), torch.from_numpy(qM[j]).cuda(), gXt[items].cuda(), gMt[items].cuda(), "chamfer").float().cpu().numpy()
            new = np.full(ncls, -1e9, np.float32); np.maximum.at(new, gy[items], scs)
            r2.append(ranks_from_scores(np.where(new > -1e8, 10 + new, base)[None], [yq[j]])[0])
        out[tag] = {k: metrics_from_ranks(np.asarray(v))["R@1"] for k, v in (("R0", r0), ("R1", r1), ("R2", r2))}
        print(tag, out[tag], flush=True)
    res["reversed_queries_2000"] = out
    json.dump(res, open(RT / "explore" / "order_controls_val.json", "w"), indent=1)


if __name__ == "__main__":
    main()
