"""Index size and query latency of readouts R0 / R1 / R2 (ASL Citizen train gallery, 1000 val queries)."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import activity_weights, class_max_scores, l2n, pool_clips
from diag_order import load, seqs, binned, pad, seq_scores
R = Path(__file__).resolve().parents[1] / "runtime" / "feats"
gc, goff, gw, ga = load(R / "aslc_train_s2"); qc, qoff, qw, qa = load(R / "aslc_val_s2")
gl_ = sorted({c["gloss"] for c in gc}); lab = {g: i for i, g in enumerate(gl_)}; gy = np.array([lab[c["gloss"]] for c in gc])
G0 = l2n(pool_clips(gw, goff, activity_weights(ga))[0]); Q0 = l2n(pool_clips(qw, qoff, activity_weights(qa))[0])
gs, qs = seqs(goff, gw, ga), seqs(qoff, qw, qa); GB, QB = binned(gs, 2), binned(qs, 2)
GW = l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in gs])); QW = l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in qs]))
gX, gM = pad(gs, 48); gXt, gMt = torch.from_numpy(gX.astype(np.float16)), torch.from_numpy(gM); qX, qM = pad(qs, 48)
n = 1000; out = {}
win_per_clip = float(np.mean([len(w) for w, _ in gs]))
def t_single(Q, G):
    torch.cuda.synchronize(); t = time.time()
    for i in range(n): class_max_scores(Q[i:i+1], G, gy, len(gl_), device="cuda")
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
out["R0 averaged"] = {"floats_per_clip": 768, "ms_per_query": t_single(Q0, G0)}
out["R1 bins2"] = {"floats_per_clip": 1536, "ms_per_query": t_single(QB, GB)}
torch.cuda.synchronize(); t = time.time()
for i in range(n):
    base = class_max_scores(QW[i:i+1], GW, gy, len(gl_), device="cuda")[0]; top = np.argsort(-base)[:20]; items = np.where(np.isin(gy, top))[0]
    seq_scores(torch.from_numpy(qX[i]).cuda().half(), torch.from_numpy(qM[i]).cuda(), gXt[items].cuda(), gMt[items].cuda(), "chamfer")
torch.cuda.synchronize()
out["R2 chamfer@20"] = {"floats_per_clip": round(768 * (1 + win_per_clip)), "ms_per_query": (time.time() - t) / n * 1e3,
                        "note": f"stores active-span windows (mean {win_per_clip:.1f}/clip, stride 2) + one vector for candidate selection"}
out["gallery_clips"] = len(gy)
json.dump(out, open(Path(__file__).resolve().parents[1] / "runtime" / "eval" / "cost_table.json", "w"), indent=1)
print(json.dumps(out, indent=1))
