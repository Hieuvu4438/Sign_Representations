"""EXPLORATORY (ASL Citizen val, official gallery): per-clip rest-state subtraction.
rest r = mean of the clip's low-activity windows (γ < 0.2; else the 2 lowest-γ windows);
readout = L2N( L2N(active readout) − α·L2N(r) ), applied to gallery and queries alike."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import activity_weights, class_max_scores, l2n, metrics_from_ranks, ranks_from_scores, per_query, paired_cluster_bootstrap
from diag_order import load, seqs, binned
R = Path(__file__).resolve().parents[1] / "runtime" / "feats"
def rest(off, W, A):
    out, nrest = [], []
    for i in range(len(off) - 1):
        w = W[off[i]:off[i+1]].astype(np.float32); g = A[off[i]:off[i+1]]
        idx = np.where(g < 0.2)[0]
        if len(idx) == 0: idx = np.argsort(g)[:2]
        out.append(l2n(w[idx]).mean(0)); nrest.append(len(idx))
    return l2n(np.stack(out)), np.array(nrest)
def prep(name):
    c, off, W, act = load(R / name); A = activity_weights(act); sq = seqs(off, W, act)
    Wm = l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in sq])); B = binned(sq, 2); r, nr = rest(off, W, A)
    return c, Wm, B, r, nr
gc, GW, GB, Gr, _ = prep(sys.argv[1]); qc, QW, QB, Qr, qnr = prep(sys.argv[2])
gl_ = sorted({c["gloss"] for c in gc}); lab = {g: i for i, g in enumerate(gl_)}
gy = np.array([lab[c["gloss"]] for c in gc]); qy = np.array([lab[c["gloss"]] for c in qc]); sg = np.array([c["signer"] for c in qc])
print("query clips with >=1 window γ<0.2: %.3f; mean rest windows %.1f" % ((qnr > 0).mean(), qnr.mean()))
base = {}
for name, G, Q in (("wmean", GW, QW), ("bins2", GB, QB)):
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        if name == "bins2":
            d = G.shape[1] // 2
            f = lambda X, r: l2n(np.concatenate([l2n(X[:, :d] - alpha * r), l2n(X[:, d:] - alpha * r)], 1))
        else:
            f = lambda X, r: l2n(X - alpha * r)
        r_ = ranks_from_scores(class_max_scores(f(Q, Qr), f(G, Gr), gy, len(gl_)), qy)
        if alpha == 0: base[name] = r_
        b = paired_cluster_bootstrap(per_query(base[name])["R@1"], per_query(r_)["R@1"], sg)
        print(f"{name:6s} alpha {alpha:.2f} R@1 {metrics_from_ranks(r_)['R@1']:.4f}  vs alpha0 {b['diff']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] pos {b['n_clusters_positive']}/{b['n_clusters']}", flush=True)
