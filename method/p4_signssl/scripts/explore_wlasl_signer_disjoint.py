"""EXPLORATORY: WLASL val queries vs train gallery, signer-overlapping (official) vs a
signer-disjoint variant (gallery clips of the query's signer removed). Readouts R0/R1/R2."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import activity_weights, class_max_scores, l2n, metrics_from_ranks, pool_clips, ranks_from_scores
from diag_order import load, seqs, binned, pad, seq_scores
RT = Path(__file__).resolve().parents[1] / "runtime"
meta = json.load(open(RT / "external/wlasl/WLASL_v0.3.json"))
sig = {i["video_id"]: i["signer_id"] for e in meta for i in e["instances"]}
gc, goff, gw, ga = load(RT / "feats/wlasl_train_s2"); qc, qoff, qw, qa = load(RT / "feats/wlasl_val_s2")
gl_ = sorted({c["gloss"] for c in gc}); lab = {g: i for i, g in enumerate(gl_)}
gy = np.array([lab[c["gloss"]] for c in gc]); qy = np.array([lab[c["gloss"]] for c in qc])
gsg = np.array([sig.get(c["sample_id"], -1) for c in gc]); qsg = np.array([sig.get(c["sample_id"], -2) for c in qc])
G0 = l2n(pool_clips(gw, goff, activity_weights(ga))[0]); Q0 = l2n(pool_clips(qw, qoff, activity_weights(qa))[0])
gs, qs = seqs(goff, gw, ga), seqs(qoff, qw, qa)
GW = l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in gs])); QW = l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in qs]))
GB, QB = binned(gs, 2), binned(qs, 2)
gX, gM = pad(gs, 48); gXt, gMt = torch.from_numpy(gX.astype(np.float16)), torch.from_numpy(gM)
qX, qM = pad(qs, 48)
def run(disjoint):
    r0, r1, r2, possible = [], [], [], []
    for i in range(len(qy)):
        keep = np.where(gsg != qsg[i])[0] if disjoint else np.arange(len(gy))
        possible.append(qy[i] in set(gy[keep]))
        r0.append(ranks_from_scores(class_max_scores(Q0[i:i+1], G0[keep], gy[keep], len(gl_), device="cpu"), [qy[i]])[0])
        r1.append(ranks_from_scores(class_max_scores(QB[i:i+1], GB[keep], gy[keep], len(gl_), device="cpu"), [qy[i]])[0])
        base = class_max_scores(QW[i:i+1], GW[keep], gy[keep], len(gl_), device="cpu")[0]
        top = np.argsort(-base)[:20]; items = keep[np.isin(gy[keep], top)]
        sc = seq_scores(torch.from_numpy(qX[i]).cuda().half(), torch.from_numpy(qM[i]).cuda(), gXt[items].cuda(), gMt[items].cuda(), "chamfer").float().cpu().numpy()
        new = np.full(len(gl_), -1e9, np.float32); np.maximum.at(new, gy[items], sc)
        r2.append(ranks_from_scores(np.where(new > -1e8, 10 + new, base)[None], [qy[i]])[0])
    m = [metrics_from_ranks(np.array(r))["R@1"] for r in (r0, r1, r2)]
    print(f"{'signer-disjoint' if disjoint else 'official (signers shared)':26s} R0 {m[0]:.4f} bins2 {m[1]-m[0]:+.4f} chamfer {m[2]-m[0]:+.4f}  (queries with class in gallery: {np.mean(possible):.3f})", flush=True)
print("val queries whose signer appears in train:", np.mean([s in set(gsg) for s in qsg]))
run(False); run(True)
