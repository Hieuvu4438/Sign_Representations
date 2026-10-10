"""EXPLORATORY (val LOSO, 2000-query subset): chamfer with a temporal-position penalty.
sim(i,j) = sym-mean over windows of max_k [ S - beta*|tau_q - tau_g| ]; tau = normalised
position inside the active span. beta=0 is plain (order-free) chamfer."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.retrieval import class_max_scores, l2n, metrics_from_ranks, ranks_from_scores
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_order import seqs, pad
d = Path(sys.argv[1]); M = 20; L = 48
clips = [json.loads(l) for l in open(d / "clips.jsonl")]
off = np.load(d / "offsets.npy"); act = np.load(d / "windows_active.npy"); win = np.load(d / "windows_features.npy")
gl_ = sorted({c["gloss"] for c in clips}); lab = {g: i for i, g in enumerate(gl_)}
y = np.array([lab[c["gloss"]] for c in clips]); sg = np.array([c["signer"] for c in clips])
sub = set(l.strip() for l in open("method/p4_signssl/configs/val_subset2000.txt") if l.strip())
qidx = [i for i, c in enumerate(clips) if c["sample_id"] in sub]
sq = seqs(off, win, act, 0.5)
W = l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in sq]))
X, Mk = pad(sq, L); X, Mk = torch.from_numpy(X).cuda().half(), torch.from_numpy(Mk).cuda()
n = Mk.sum(1).clamp(min=1).float()
tau = (torch.arange(L, device="cuda")[None].float() / (n[:, None] - 1).clamp(min=1)).clamp(max=1)
betas = [0.0, 0.2]
rev = "--reverse-query" in sys.argv
res = {b: [] for b in betas}
for c, i in enumerate(qidx):
    g = np.where(sg != sg[i])[0]
    base = class_max_scores(W[i:i+1], W[g], y[g], len(gl_), device="cuda")[0]
    top = np.argsort(-base)[:M]; items = torch.from_numpy(g[np.isin(y[g], top)]).cuda()
    S = torch.einsum("qd,bld->bql", X[i], X[items]).float()
    tq = tau[i] if not rev else (1 - tau[i]) * (tau[i] <= 1)
    D = (tq[None, :, None] - tau[items][:, None, :]).abs()
    qm, gm = Mk[i][None, :, None], Mk[items][:, None, :]
    for b in betas:
        A = (S - b * D)
        s1 = A.masked_fill(~gm, -9).amax(2); s2 = A.masked_fill(~qm, -9).amax(1)
        sc = 0.5 * ((s1 * Mk[i][None]).sum(1) / Mk[i].sum() + (s2 * Mk[items]).sum(1) / Mk[items].sum(1).clamp(min=1))
        new = np.full(len(gl_), -1e9, np.float32); np.maximum.at(new, y[items.cpu().numpy()], sc.cpu().numpy())
        res[b].append(ranks_from_scores(np.where(new > -1e8, 10 + new, base)[None], [y[i]])[0])
base_r = []
for i in qidx:
    g = np.where(sg != sg[i])[0]
    base_r.append(ranks_from_scores(class_max_scores(W[i:i+1], W[g], y[g], len(gl_), device="cuda"), [y[i]])[0])
Path("method/p4_signssl/runtime/explore").mkdir(parents=True, exist_ok=True)
np.savez("method/p4_signssl/runtime/explore/chamfer_pos_ranks%s.npz" % ("_rev" if rev else ""), qidx=np.array(qidx),
         wmean=np.array(base_r), **{f"beta{b}": np.array(res[b]) for b in betas})
for b in betas:
    m = metrics_from_ranks(np.array(res[b])); print(f"beta {b:.1f}{' (query tau reversed)' if rev else ''} R@1 {m['R@1']:.4f} MRR {m['MRR']:.4f}")
