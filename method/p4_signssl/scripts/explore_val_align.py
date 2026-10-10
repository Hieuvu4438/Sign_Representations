"""EXPLORATORY (val LOSO, 2000-query subset): chamfer / DTW rerank of top-M wmean classes."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.retrieval import class_max_scores, l2n, metrics_from_ranks, ranks_from_scores
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_order import seqs, pad, seq_scores, binned
d = Path(sys.argv[1]); M = 20
clips = [json.loads(l) for l in open(d / "clips.jsonl")]
off = np.load(d / "offsets.npy"); act = np.load(d / "windows_active.npy"); win = np.load(d / "windows_features.npy")
gl_ = sorted({c["gloss"] for c in clips}); lab = {g: i for i, g in enumerate(gl_)}
y = np.array([lab[c["gloss"]] for c in clips]); sg = np.array([c["signer"] for c in clips])
sub = set(l.strip() for l in open("method/p4_signssl/configs/val_subset2000.txt") if l.strip())
qidx = [i for i, c in enumerate(clips) if c["sample_id"] in sub]
sq = seqs(off, win, act, 0.5)
W = l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in sq])); B2 = binned(sq, 2)
X, Mk = pad(sq, 48); Xt, Mt = torch.from_numpy(X).cuda().half(), torch.from_numpy(Mk).cuda()
res = {k: [] for k in ["wmean", "bins2", "chamfer", "dtw", "dtw+wmean"]}
for n, i in enumerate(qidx):
    g = np.where(sg != sg[i])[0]
    base = class_max_scores(W[i:i+1], W[g], y[g], len(gl_), device="cuda")[0]
    b2 = class_max_scores(B2[i:i+1], B2[g], y[g], len(gl_), device="cuda")[0]
    res["wmean"].append(ranks_from_scores(base[None], [y[i]])[0]); res["bins2"].append(ranks_from_scores(b2[None], [y[i]])[0])
    top = np.argsort(-base)[:M]; items = g[np.isin(y[g], top)]
    for mode in ("chamfer", "dtw"):
        sc = seq_scores(Xt[i], Mt[i], Xt[items], Mt[items], mode).float().cpu().numpy()
        new = np.full(len(gl_), -1e9, np.float32); np.maximum.at(new, y[items], sc)
        res[mode].append(ranks_from_scores(np.where(new > -1e8, 10 + new, base)[None], [y[i]])[0])
        if mode == "dtw":
            ws = np.full(len(gl_), -1e9, np.float32); np.maximum.at(ws, y[items], sc + (W[items] @ W[i]))
            res["dtw+wmean"].append(ranks_from_scores(np.where(ws > -1e8, 10 + ws, base)[None], [y[i]])[0])
    if (n + 1) % 500 == 0: print(n + 1, flush=True)
for k, v in res.items():
    m = metrics_from_ranks(np.array(v)); print(f"{k:10s} R@1 {m['R@1']:.4f} MRR {m['MRR']:.4f} n {m['n']}")
