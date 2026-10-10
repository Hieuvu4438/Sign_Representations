"""EXPLORATORY (ASL Citizen val, official gallery, 3000-query subset): which part of chamfer
matters? sym (default), q->g only, g->q only, partial (mean of the best 50% matched windows)."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import class_max_scores, l2n, metrics_from_ranks, ranks_from_scores
from diag_order import load, seqs, pad
R = Path(__file__).resolve().parents[1] / "runtime" / "feats"
gc, goff, gw, ga = load(R / "aslc_train_s2"); qc, qoff, qw, qa = load(R / "aslc_val_s2")
gl_ = sorted({c["gloss"] for c in gc}); lab = {g: i for i, g in enumerate(gl_)}
gy = np.array([lab[c["gloss"]] for c in gc]); qy = np.array([lab[c["gloss"]] for c in qc])
gs, qs = seqs(goff, gw, ga), seqs(qoff, qw, qa)
GW = l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in gs])); QW = l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in qs]))
gX, gM = pad(gs, 48); gXt, gMt = torch.from_numpy(gX.astype(np.float16)), torch.from_numpy(gM)
qX, qM = pad(qs, 48)
sub = np.random.default_rng(0).choice(len(qy), 3000, replace=False)
def variants(q, qm, G, Gm):
    S = torch.einsum("qd,bld->bql", q.float(), G.float())
    s1 = S.masked_fill(~Gm[:, None, :], -9).amax(2)  # per query window best match  [B,Lq]
    s2 = S.masked_fill(~qm[None, :, None], -9).amax(1)  # per gallery window best match [B,Lg]
    a = (s1 * qm[None]).sum(1) / qm.sum(); b = (s2 * Gm).sum(1) / Gm.sum(1).clamp(min=1)
    k = max(1, int(qm.sum()) // 2)
    part = s1.masked_fill(~qm[None], -9).topk(k, dim=1).values.mean(1)
    return {"sym": 0.5 * (a + b), "q2g": a, "g2q": b, "partial50_q2g": part}
res = {}
for i in sub:
    base = class_max_scores(QW[i:i+1], GW, gy, len(gl_), device="cuda")[0]
    top = np.argsort(-base)[:20]; items = np.where(np.isin(gy, top))[0]
    v = variants(torch.from_numpy(qX[i]).cuda(), torch.from_numpy(qM[i]).cuda(), gXt[items].cuda(), gMt[items].cuda())
    for k, sc in v.items():
        new = np.full(len(gl_), -1e9, np.float32); np.maximum.at(new, gy[items], sc.cpu().numpy())
        res.setdefault(k, []).append(ranks_from_scores(np.where(new > -1e8, 10 + new, base)[None], [qy[i]])[0])
    res.setdefault("wmean", []).append(ranks_from_scores(base[None], [qy[i]])[0])
for k, r in res.items():
    print(f"{k:14s} R@1 {metrics_from_ranks(np.array(r))['R@1']:.4f}")
