"""EXPLORATORY C004 (ASL Citizen val, official gallery): label-free post-hoc transforms of
frozen clip vectors fitted on the TRAIN gallery only (no gloss, no signer IDs):
  center      : subtract gallery mean
  abtt-k      : center + remove top-k principal directions ("all-but-the-top")
  whiten-k    : center + PCA-whiten keeping k dims (shrinkage eps)
Applied to wmean (active span) and to each half of bins2."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import class_max_scores, l2n, metrics_from_ranks, ranks_from_scores, per_query, paired_cluster_bootstrap
from diag_order import load, seqs, binned
R = Path(__file__).resolve().parents[1] / "runtime" / "feats"
def vecs(name):
    c, off, W, act = load(R / name); sq = seqs(off, W, act)
    return c, l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in sq])), binned(sq, 2)
gc, GW, GB = vecs("aslc_train_s2"); qc, QW, QB = vecs("aslc_val_s2")
gl_ = sorted({c["gloss"] for c in gc}); lab = {g: i for i, g in enumerate(gl_)}
gy = np.array([lab[c["gloss"]] for c in gc]); qy = np.array([lab[c["gloss"]] for c in qc]); sg = np.array([c["signer"] for c in qc])
def fit(X):
    mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False); return mu, S, Vt
def tf(X, mu, S, Vt, mode, k):
    Z = X - mu
    if mode == "abtt": Z = Z - (Z @ Vt[:k].T) @ Vt[:k]
    if mode == "whiten": Z = (Z @ Vt[:k].T) / (S[:k] / np.sqrt(len(S)) + 1e-2 * S[0] / np.sqrt(len(S)))
    return l2n(Z)
def run(G, Q, split):
    if split:
        d = G.shape[1] // 2; parts = []
        for sl in (slice(0, d), slice(d, 2 * d)):
            parts.append((G[:, sl], Q[:, sl], fit(G[:, sl])))
        return lambda mode, k: (l2n(np.concatenate([tf(g, *p, mode, k) for g, _, p in parts], 1)), l2n(np.concatenate([tf(q, *p, mode, k) for _, q, p in parts], 1)))
    p = fit(G); return lambda mode, k: (tf(G, *p, mode, k), tf(Q, *p, mode, k))
for name, G, Q, split in (("wmean", GW, QW, False), ("bins2", GB, QB, True)):
    f = run(G, Q, split); base = ranks_from_scores(class_max_scores(Q, G, gy, len(gl_)), qy)
    print(f"{name} raw R@1 {metrics_from_ranks(base)['R@1']:.4f}", flush=True)
    for mode, k in [("center", 0), ("abtt", 1), ("abtt", 2), ("abtt", 4), ("abtt", 8), ("abtt", 16), ("whiten", 128), ("whiten", 256), ("whiten", 512)]:
        Gt, Qt = f(mode, k); r = ranks_from_scores(class_max_scores(Qt, Gt, gy, len(gl_)), qy)
        b = paired_cluster_bootstrap(per_query(base)["R@1"], per_query(r)["R@1"], sg)
        print(f"  {mode:6s} k={k:3d} R@1 {metrics_from_ranks(r)['R@1']:.4f} ({b['diff']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] pos {b['n_clusters_positive']}/6)", flush=True)
