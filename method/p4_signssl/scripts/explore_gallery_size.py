"""EXPLORATORY (val, official gallery subsampled to k exemplars/class): does the readout gain
depend on gallery size per class? R0 paper readout, R1 bins2, R2 chamfer@20."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import activity_weights, class_max_scores, l2n, metrics_from_ranks, pool_clips, ranks_from_scores
from diag_order import load, seqs, binned, pad, seq_scores
R = Path(__file__).resolve().parents[1] / "runtime" / "feats"
gc, goff, gw, ga = load(R / "aslc_train_s2"); qc, qoff, qw, qa = load(R / "aslc_val_s2")
gl_ = sorted({c["gloss"] for c in gc}); lab = {g: i for i, g in enumerate(gl_)}
gy = np.array([lab[c["gloss"]] for c in gc]); qy = np.array([lab[c["gloss"]] for c in qc])
G0 = l2n(pool_clips(gw, goff, activity_weights(ga))[0]); Q0 = l2n(pool_clips(qw, qoff, activity_weights(qa))[0])
gs, qs = seqs(goff, gw, ga), seqs(qoff, qw, qa)
GW = l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in gs])); QW = l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in qs]))
GB, QB = binned(gs, 2), binned(qs, 2)
gX, gM = pad(gs, 48); gXt, gMt = torch.from_numpy(gX.astype(np.float16)).pin_memory(), torch.from_numpy(gM)
qX, qM = pad(qs, 48)
rng = np.random.default_rng(0)
qsub = rng.choice(len(qy), 3000, replace=False)
for k in (3, 7, 15):
    keep = np.concatenate([rng.permutation(np.where(gy == c)[0])[:k] for c in range(len(gl_))])
    r0 = ranks_from_scores(class_max_scores(Q0[qsub], G0[keep], gy[keep], len(gl_)), qy[qsub])
    r1 = ranks_from_scores(class_max_scores(QB[qsub], GB[keep], gy[keep], len(gl_)), qy[qsub])
    base = class_max_scores(QW[qsub], GW[keep], gy[keep], len(gl_))
    r2 = []
    for j, i in enumerate(qsub):
        top = np.argsort(-base[j])[:20]; items = keep[np.isin(gy[keep], top)]
        sc = seq_scores(torch.from_numpy(qX[i]).cuda().half(), torch.from_numpy(qM[i]).cuda(), gXt[items].cuda(), gMt[items].cuda(), "chamfer").float().cpu().numpy()
        new = np.full(len(gl_), -1e9, np.float32); np.maximum.at(new, gy[items], sc)
        r2.append(ranks_from_scores(np.where(new > -1e8, 10 + new, base[j])[None], [qy[i]])[0])
    m = [metrics_from_ranks(r)["R@1"] for r in (r0, r1, np.array(r2))]
    print(f"k={k:2d}/class  R0 {m[0]:.4f}  bins2 {m[1]:.4f} ({m[1]-m[0]:+.4f})  chamfer {m[2]:.4f} ({m[2]-m[0]:+.4f})", flush=True)
