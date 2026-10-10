"""EX06-v1: SHuBERT layer-12 readouts on ASL Citizen val, leave-one-signer-out."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import class_max_scores, l2n, metrics_from_ranks, ranks_from_scores, per_query, paired_cluster_bootstrap
from diag_order import pad, seq_scores
D = Path(__file__).resolve().parents[1] / "runtime" / "feats" / (sys.argv[1] if len(sys.argv) > 1 else "shubert_val")
clips = [json.loads(l) for l in open(D / "clips.jsonl")]
assert all(c["status"] == "ok" for c in clips)
off = np.load(D / "offsets.npy"); F = np.load(D / "frames_l12.npy"); V = np.load(D / "frame_valid.npy")
gl_ = sorted({c["gloss"] for c in clips}); lab = {g: i for i, g in enumerate(gl_)}
y = np.array([lab[c["gloss"]] for c in clips]); sg = np.array([c["signer"] for c in clips])
S0, S1, S2, seqs = [], [], [], []
for i in range(len(clips)):
    f = F[off[i]:off[i+1]].astype(np.float32); v = V[off[i]:off[i+1]]
    idx = np.where(v)[0]; a, b = (idx[0], idx[-1] + 1) if len(idx) else (0, len(f))
    g = f[a:b]; S0.append(f.mean(0)); S1.append(g.mean(0))
    h = len(g) // 2 if len(g) > 1 else 1
    S2.append(np.concatenate([l2n(g[:h].mean(0, keepdims=True))[0], l2n(g[h:].mean(0, keepdims=True) if len(g) > 1 else g.mean(0, keepdims=True))[0]]) / np.sqrt(2))
    w = g[::2]
    if len(w) > 48: w = w[np.round(np.linspace(0, len(w) - 1, 48)).astype(int)]
    seqs.append((l2n(w), None))
S0, S1, S2 = l2n(np.stack(S0)), l2n(np.stack(S1)), np.stack(S2).astype(np.float32)
def loso(Fm):
    r = np.zeros(len(y), int); S = np.zeros((len(y), len(gl_)), np.float32)
    for s in np.unique(sg):
        q, g = sg == s, sg != s
        S[q] = class_max_scores(Fm[q], Fm[g], y[g], len(gl_)); r[q] = ranks_from_scores(S[q], y[q])
    return r, S
res, ranks = {}, {}
for n, Fm in (("S0_mean_all", S0), ("S1_mean_valid", S1), ("S2_bins2", S2)):
    ranks[n], sc = loso(Fm)
    if n == "S1_mean_valid": base = sc
X, M = pad(seqs, 48); Xt, Mt = torch.from_numpy(X).cuda().half(), torch.from_numpy(M).cuda()
r3 = []
for i in range(len(y)):
    g = np.where(sg != sg[i])[0]; top = np.argsort(-base[i])[:20]; items = g[np.isin(y[g], top)]
    sc = seq_scores(Xt[i], Mt[i], Xt[items], Mt[items], "chamfer").float().cpu().numpy()
    new = np.full(len(gl_), -1e9, np.float32); np.maximum.at(new, y[items], sc)
    r3.append(ranks_from_scores(np.where(new > -1e8, 10 + new, base[i])[None], [y[i]])[0])
ranks["S3_chamfer20"] = np.array(r3)
out = Path(__file__).resolve().parents[1] / "runtime" / "eval" / ("EX06_" + D.name); out.mkdir(parents=True, exist_ok=True)
for n, r in ranks.items():
    np.save(out / f"ranks_{n}.npy", r); res[n] = metrics_from_ranks(r, y)
    print(f"{n:14s} R@1 {res[n]['R@1']:.4f} MRR {res[n]['MRR']:.4f} DCG {res[n]['DCG']:.4f}", flush=True)
for n in ("S1_mean_valid", "S2_bins2", "S3_chamfer20"):
    b = paired_cluster_bootstrap(per_query(ranks["S0_mean_all"])["R@1"], per_query(ranks[n])["R@1"], sg)
    res[f"{n}-S0"] = b; print(f"{n}-S0 R@1 {b['diff']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] pos {b['n_clusters_positive']}/6")
json.dump(res, open(out / "metrics.json", "w"), indent=1)
