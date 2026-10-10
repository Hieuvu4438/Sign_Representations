"""C005 diagnostic: label-free pseudo-positive mining among ASL Citizen TRAIN clips.
For each clip: candidates = top-50 other clips by active-span wmean; rerank by chamfer; pick top-1.
Labels/signers used ONLY to measure mining quality (same gloss / same signer rates)."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import l2n
from diag_order import load, seqs, pad, seq_scores
RT = Path(__file__).resolve().parents[1] / "runtime"
c, off, W, act = load(RT / "feats" / "aslc_train_s2"); sq = seqs(off, W, act)
gl = np.array([x["gloss"] for x in c]); sg = np.array([x["signer"] for x in c])
Wm = torch.from_numpy(l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in sq]))).cuda()
X, M = pad(sq, 48); Xt, Mt = torch.from_numpy(X.astype(np.float16)), torch.from_numpy(M)
rng = np.random.default_rng(0); idx = rng.choice(len(c), 4000, replace=False)
res = {"wmean_top1": [], "chamfer_top1": [], "chamfer_score": []}
for i in idx:
    s = Wm[i] @ Wm.T; s[i] = -9; cand = s.topk(50).indices.cpu().numpy()
    res["wmean_top1"].append(cand[0])
    sc = seq_scores(Xt[i].cuda(), Mt[i].cuda(), Xt[cand].cuda(), Mt[cand].cuda(), "chamfer").float().cpu().numpy()
    res["chamfer_top1"].append(cand[sc.argmax()]); res["chamfer_score"].append(sc.max())
for k in ("wmean_top1", "chamfer_top1"):
    j = np.array(res[k])
    print(f"{k:13s} same-gloss {np.mean(gl[j]==gl[idx]):.3f}  same-signer {np.mean(sg[j]==sg[idx]):.3f}  same-gloss&diff-signer {np.mean((gl[j]==gl[idx])&(sg[j]!=sg[idx])):.3f}")
sc = np.array(res["chamfer_score"]); j = np.array(res["chamfer_top1"])
for q in (0.5, 0.75, 0.9):
    t = np.quantile(sc, q); m = sc >= t
    print(f"chamfer score >= q{q:.2f}: kept {m.mean():.2f}, same-gloss {np.mean(gl[j][m]==gl[idx][m]):.3f}, same-signer {np.mean(sg[j][m]==sg[idx][m]):.3f}")
