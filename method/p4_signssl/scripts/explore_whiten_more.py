"""EXPLORATORY C004: whitening dims k in {512,640,768} for bins2; window-level whitening
(fit on train windows, active spans) followed by chamfer@20 (3000-query subset)."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import class_max_scores, l2n, metrics_from_ranks, ranks_from_scores
from diag_order import load, seqs, binned, pad, seq_scores
R = Path(__file__).resolve().parents[1] / "runtime" / "feats"
def get(name):
    c, off, W, act = load(R / name); sq = seqs(off, W, act); return c, sq
gc, gs = get("aslc_train_s2"); qc, qs = get("aslc_val_s2")
gl_ = sorted({c["gloss"] for c in gc}); lab = {g: i for i, g in enumerate(gl_)}
gy = np.array([lab[c["gloss"]] for c in gc]); qy = np.array([lab[c["gloss"]] for c in qc])
def fitw(X, k):
    mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False); s = S / np.sqrt(len(X))
    return lambda Z: ((Z - mu) @ Vt[:k].T) / (s[:k] + 1e-2 * s[0])
GB, QB = binned(gs, 2), binned(qs, 2); d = GB.shape[1] // 2
for k in (512, 640, 768):
    fs = [fitw(GB[:, :d], k), fitw(GB[:, d:], k)]
    G = l2n(np.concatenate([l2n(fs[0](GB[:, :d])), l2n(fs[1](GB[:, d:]))], 1)); Q = l2n(np.concatenate([l2n(fs[0](QB[:, :d])), l2n(fs[1](QB[:, d:]))], 1))
    print(f"bins2+whiten k={k} R@1 {metrics_from_ranks(ranks_from_scores(class_max_scores(Q, G, gy, len(gl_)), qy))['R@1']:.4f}", flush=True)
# window-level whitening (fit on a 300k-window sample of train active-span windows)
rng = np.random.default_rng(0)
allw = np.concatenate([w for w, _ in gs]); samp = allw[rng.choice(len(allw), min(300000, len(allw)), replace=False)]
fw = fitw(samp, 512)
gs2 = [(l2n(fw(w)).astype(np.float32), g) for w, g in gs]; qs2 = [(l2n(fw(w)).astype(np.float32), g) for w, g in qs]
GW2 = l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in gs2])); QW2 = l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in qs2]))
print(f"window-whiten(512) wmean R@1 {metrics_from_ranks(ranks_from_scores(class_max_scores(QW2, GW2, gy, len(gl_)), qy))['R@1']:.4f}", flush=True)
B2g, B2q = binned(gs2, 2), binned(qs2, 2)
print(f"window-whiten(512) bins2 R@1 {metrics_from_ranks(ranks_from_scores(class_max_scores(B2q, B2g, gy, len(gl_)), qy))['R@1']:.4f}", flush=True)
sub = rng.choice(len(qy), 3000, replace=False)
for tag, G_s, Q_s, GWm, QWm in (("raw", gs, qs, l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in gs])), l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in qs]))), ("window-whiten", gs2, qs2, GW2, QW2)):
    gX, gM = pad(G_s, 48); qX, qM = pad(Q_s, 48); gXt, gMt = torch.from_numpy(gX.astype(np.float16)), torch.from_numpy(gM)
    rr = []
    for i in sub:
        base = class_max_scores(QWm[i:i+1], GWm, gy, len(gl_), device="cuda")[0]; top = np.argsort(-base)[:20]; items = np.where(np.isin(gy, top))[0]
        sc = seq_scores(torch.from_numpy(qX[i]).cuda().half(), torch.from_numpy(qM[i]).cuda(), gXt[items].cuda(), gMt[items].cuda(), "chamfer").float().cpu().numpy()
        new = np.full(len(gl_), -1e9, np.float32); np.maximum.at(new, gy[items], sc); rr.append(ranks_from_scores(np.where(new > -1e8, 10 + new, base)[None], [qy[i]])[0])
    print(f"chamfer@20 ({tag}) on 3000-subset R@1 {metrics_from_ranks(np.array(rr))['R@1']:.4f}", flush=True)
