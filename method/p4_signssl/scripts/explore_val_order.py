"""EXPLORATORY (val LOSO): order-aware training-free readouts vs activity-weighted mean."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.retrieval import activity_weights, class_max_scores, l2n, metrics_from_ranks, ranks_from_scores
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_order import seqs, binned
d = Path(sys.argv[1]); thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
clips = [json.loads(l) for l in open(d / "clips.jsonl")]
off = np.load(d / "offsets.npy"); act = np.load(d / "windows_active.npy"); win = np.load(d / "windows_features.npy")
gl_ = sorted({c["gloss"] for c in clips}); lab = {g: i for i, g in enumerate(gl_)}
y = np.array([lab[c["gloss"]] for c in clips]); sg = np.array([c["signer"] for c in clips])
sq = seqs(off, win, act, thr)
def loso(feat):
    r = np.zeros(len(y), int)
    for s in np.unique(sg):
        q, g = sg == s, sg != s
        r[q] = ranks_from_scores(class_max_scores(feat[q], feat[g], y[g], len(gl_)), y[q])
    return r
W = l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in sq]))
U = l2n(np.stack([w.mean(0) for w, g in sq]))
out = {"wmean_span": metrics_from_ranks(loso(W)), "mean_span": metrics_from_ranks(loso(U))}
for K in (2, 3, 4, 6):
    B = binned(sq, K)
    out[f"bins{K}"] = metrics_from_ranks(loso(B))
    out[f"bins{K}+wmean"] = metrics_from_ranks(loso(l2n(np.concatenate([B, W], 1))))
# negative control: bins with time order reversed within the query only is equivalent to
# comparing reversed bins; report shuffled-bin control (random bin permutation per clip)
rng = np.random.default_rng(0)
B4 = binned(sq, 4).reshape(len(sq), 4, -1)
perm = np.stack([rng.permutation(4) for _ in range(len(sq))])
B4s = np.take_along_axis(B4, perm[:, :, None], 1).reshape(len(sq), -1)
out["bins4_shuffled_order"] = metrics_from_ranks(loso(B4s))
for k, v in out.items():
    print(f"{k:22s} R@1 {v['R@1']:.4f} MRR {v['MRR']:.4f} DCG {v['DCG']:.4f}")
