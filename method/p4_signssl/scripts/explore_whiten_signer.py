"""EXPLORATORY H005 diagnostic (WLASL val): same-signer distractor penalty with and without
gallery whitening (fit on unlabelled WLASL train gallery). Penalty = R@1(signer-disjoint gallery)
- R@1(official gallery). If whitening removes signer nuisance, its penalty should shrink."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import class_max_scores, l2n, metrics_from_ranks, ranks_from_scores
from diag_order import load, seqs, binned
RT = Path(__file__).resolve().parents[1] / "runtime"
sig = {i["video_id"]: i["signer_id"] for e in json.load(open(RT / "external/wlasl/WLASL_v0.3.json")) for i in e["instances"]}
def get(n):
    c, off, W, act = load(RT / "feats" / n); sq = seqs(off, W, act)
    return c, l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in sq])), binned(sq, 2)
gc, GW, GB = get("wlasl_train_s2"); qc, QW, QB = get("wlasl_val_s2")
gl_ = sorted({c["gloss"] for c in gc}); lab = {g: i for i, g in enumerate(gl_)}
gy = np.array([lab[c["gloss"]] for c in gc]); qy = np.array([lab[c["gloss"]] for c in qc])
gsg = np.array([sig.get(c["sample_id"], -1) for c in gc]); qsg = np.array([sig.get(c["sample_id"], -2) for c in qc])
def fitw(X, k=512):
    mu = X.mean(0); _, S, Vt = np.linalg.svd(X - mu, full_matrices=False); s = S / np.sqrt(len(X)); k = min(k, len(s))
    return lambda Z: l2n(((Z - mu) @ Vt[:k].T) / (s[:k] + 0.01 * s[0]))
def r1(G, Q, disjoint):
    if not disjoint:
        return metrics_from_ranks(ranks_from_scores(class_max_scores(Q, G, gy, len(gl_)), qy))["R@1"]
    S = class_max_scores(Q, G, gy, len(gl_)); r = []
    for s in np.unique(qsg):
        q = qsg == s; keep = gsg != s
        r.append(ranks_from_scores(class_max_scores(Q[q], G[keep], gy[keep], len(gl_)), qy[q]))
    order = np.concatenate([np.where(qsg == s)[0] for s in np.unique(qsg)])
    rr = np.empty(len(qy), int); rr[order] = np.concatenate(r)
    return metrics_from_ranks(rr)["R@1"]
d = GB.shape[1] // 2
fw = fitw(GW); f1, f2 = fitw(GB[:, :d]), fitw(GB[:, d:])
wb = lambda B: l2n(np.concatenate([f1(B[:, :d]), f2(B[:, d:])], 1)).astype(np.float32)
for name, G, Q in (("wmean", GW, QW), ("wmean+whiten", fw(GW).astype(np.float32), fw(QW).astype(np.float32)),
                   ("bins2", GB, QB), ("bins2+whiten", wb(GB), wb(QB))):
    a, b = r1(G, Q, False), r1(G, Q, True)
    print(f"{name:14s} official {a:.4f}  signer-disjoint {b:.4f}  same-signer penalty {b-a:+.4f}", flush=True)
