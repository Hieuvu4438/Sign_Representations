"""Training-free single-vector pools on the official gallery (train -> val), same windows."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.retrieval import activity_weights, class_max_scores, l2n, metrics_from_ranks, ranks_from_scores
R = Path(__file__).resolve().parents[1] / "runtime"
def load(d):
    d = R / "feats" / d
    c = [json.loads(l) for l in open(d / "clips.jsonl")]
    return c, np.load(d / "offsets.npy"), np.load(d / "windows_features.npy"), activity_weights(np.load(d / "windows_active.npy"))
def pools(off, W, g, thr=0.5):
    out = {k: [] for k in ["wmean_span", "nmean_span", "max_span", "gem3_span", "wnmean_span"]}
    for i in range(len(off) - 1):
        w = W[off[i]:off[i+1]].astype(np.float32); a = g[off[i]:off[i+1]]
        idx = np.where(a >= thr)[0]
        s, e = (idx[0], idx[-1] + 1) if len(idx) else (0, len(w))
        w, a = w[s:e], a[s:e] + 1e-6
        wn = l2n(w)
        out["wmean_span"].append((a[:, None] * w).sum(0))
        out["nmean_span"].append(wn.mean(0))
        out["wnmean_span"].append((a[:, None] * wn).sum(0))
        out["max_span"].append(wn.max(0))
        out["gem3_span"].append(np.sign(wn.mean(0)) * np.abs((np.abs(wn) ** 3).mean(0)) ** (1 / 3))
    return {k: l2n(np.stack(v)) for k, v in out.items()}
gc, goff, gw, ga = load("aslc_train_s2"); qc, qoff, qw, qa = load("aslc_val_s2")
gl_ = sorted({c["gloss"] for c in gc}); lab = {g: i for i, g in enumerate(gl_)}
gy = np.array([lab[c["gloss"]] for c in gc]); qy = np.array([lab[c["gloss"]] for c in qc])
G, Q = pools(goff, gw, ga), pools(qoff, qw, qa)
for k in G:
    r = ranks_from_scores(class_max_scores(Q[k], G[k], gy, len(gl_)), qy); m = metrics_from_ranks(r)
    np.save(R / "eval" / "EX01_readouts" / f"ranks_pool_{k}.npy", r)
    print(f"{k:12s} R@1 {m['R@1']:.4f} MRR {m['MRR']:.4f} DCG {m['DCG']:.4f}")
