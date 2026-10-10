"""H010 diagnostic (no training): label-free teacher similarities for set distillation.
(a) Mining quality on unlabelled ASL Citizen TRAIN anchors (labels/signers used only to score):
    rerank each anchor's 64 wmean candidates (other train clips) by the teacher; report top-1 and
    top-5 rates of same-gloss, same-signer, same-gloss-different-signer.
(b) Teacher used directly as a readout on ASL Citizen VAL (official gallery, chamfer@20 rerank).
Teachers (all label-free, all on frozen SignRep active-span windows):
  T0 q2g chamfer on windows (H008 teacher)
  T1 q2g chamfer on temporal differences w[t+k]-w[t] (static appearance cancels to first order)
  T2 q2g chamfer on clip-centred windows w - mean_clip(w)
  T3 q2g chamfer on window-whitened windows (PCA fit on unlabelled train windows)
  T4 T0 + T1 (sum of similarities)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import class_max_scores, l2n, metrics_from_ranks, ranks_from_scores  # noqa: E402
from diag_order import load, pad, seqs  # noqa: E402

RT = Path(__file__).resolve().parents[1] / "runtime"
K_DIFF = 4  # windows (stride 2) -> 8 frames


def variants(sq, fw):
    out = {"T0": [], "T1": [], "T2": [], "T3": []}
    for w, g in sq:
        out["T0"].append(w)
        d = w[K_DIFF:] - w[:-K_DIFF] if len(w) > K_DIFF else (w[-1:] - w[:1] + 1e-6)
        out["T1"].append(l2n(d))
        out["T2"].append(l2n(w - w.mean(0, keepdims=True) + 1e-6))
        out["T3"].append(l2n(fw(w)).astype(np.float32))
    return out


def q2g(qx, qm, gx, gm):
    S = torch.einsum("qd,bld->bql", qx.float(), gx.float())
    s1 = S.masked_fill(~gm[:, None, :], -9).amax(2)
    return (s1 * qm[None]).sum(1) / qm.sum()


def main():
    gc, goff, gw, ga = load(RT / "feats" / "aslc_train_s2"); qc, qoff, qw, qa = load(RT / "feats" / "aslc_val_s2")
    gs, qs = seqs(goff, gw, ga), seqs(qoff, qw, qa)
    rng = np.random.default_rng(0)
    allw = np.concatenate([w for w, _ in gs]); samp = allw[rng.choice(len(allw), 300000, replace=False)]
    mu = samp.mean(0); _, S, Vt = np.linalg.svd(samp - mu, full_matrices=False); s = S / np.sqrt(len(samp))
    fw = lambda Z: ((Z - mu) @ Vt[:512].T) / (s[:512] + 0.01 * s[0])
    GV, QV = variants(gs, fw), variants(qs, fw)
    wm = lambda sq: l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in sq]))
    GW, QW = wm(gs), wm(qs)
    gl = np.array([c["gloss"] for c in gc]); sg = np.array([c["signer"] for c in gc])
    PAD = {}
    for k in ("T0", "T1", "T2", "T3"):
        X, M = pad([(v, None) for v in GV[k]], 48); PAD[k] = (torch.from_numpy(X.astype(np.float16)), torch.from_numpy(M))
    res = {"mining_train": {}, "readout_val": {}}
    # (a) mining quality on 3000 train anchors
    anchors = rng.choice(len(gs), 3000, replace=False)
    Wt = torch.from_numpy(GW).cuda()
    tops = {k: [] for k in ("wmean", "T0", "T1", "T2", "T3", "T4")}
    for i in anchors:
        sc = Wt[i] @ Wt.T; sc[i] = -9; cand = sc.topk(64).indices.cpu().numpy()
        tops["wmean"].append(cand[:5])
        sims = {}
        for k in ("T0", "T1", "T2", "T3"):
            X, M = PAD[k]
            sims[k] = q2g(X[i].cuda(), M[i].cuda(), X[cand].cuda(), M[cand].cuda()).float().cpu().numpy()
        sims["T4"] = sims["T0"] + sims["T1"]
        for k, v in sims.items():
            tops[k].append(cand[np.argsort(-v)[:5]])
    for k, t in tops.items():
        t = np.array(t)
        r = {}
        for n in (1, 5):
            sel = t[:, :n]
            same_g = gl[sel] == gl[anchors][:, None]; same_s = sg[sel] == sg[anchors][:, None]
            r[f"top{n}_same_gloss"] = float(same_g.mean()); r[f"top{n}_same_signer"] = float(same_s.mean())
            r[f"top{n}_same_gloss_diff_signer"] = float((same_g & ~same_s).mean())
        res["mining_train"][k] = r
        print(f"{k:6s} top1 gloss {r['top1_same_gloss']:.3f} signer {r['top1_same_signer']:.3f} gloss&diffsigner {r['top1_same_gloss_diff_signer']:.3f} | top5 gloss&diffsigner {r['top5_same_gloss_diff_signer']:.3f}", flush=True)
    # (b) readout on val (official gallery), 3000-query subset, chamfer@20 rerank of wmean top-20 classes
    labs = sorted(set(gl)); lab = {g: i for i, g in enumerate(labs)}
    gy = np.array([lab[x] for x in gl]); qy = np.array([lab[c["gloss"]] for c in qc])
    sub = rng.choice(len(qs), 3000, replace=False)
    QP = {}
    for k in ("T0", "T1", "T2", "T3"):
        X, M = pad([(v, None) for v in QV[k]], 48); QP[k] = (torch.from_numpy(X.astype(np.float16)), torch.from_numpy(M))
    rr = {k: [] for k in ("wmean", "T0", "T1", "T2", "T3", "T4")}
    for i in sub:
        base = class_max_scores(QW[i:i + 1], GW, gy, len(labs), device="cuda")[0]
        rr["wmean"].append(ranks_from_scores(base[None], [qy[i]])[0])
        top = np.argsort(-base)[:20]; items = np.where(np.isin(gy, top))[0]
        sims = {}
        for k in ("T0", "T1", "T2", "T3"):
            gx, gm = PAD[k]; qx, qm = QP[k]
            a_ = q2g(qx[i].cuda(), qm[i].cuda(), gx[items].cuda(), gm[items].cuda())
            sims[k] = a_.float().cpu().numpy()
        sims["T4"] = sims["T0"] + sims["T1"]
        for k, v in sims.items():
            new = np.full(len(labs), -1e9, np.float32); np.maximum.at(new, gy[items], v)
            rr[k].append(ranks_from_scores(np.where(new > -1e8, 10 + new, base)[None], [qy[i]])[0])
    for k, v in rr.items():
        m = metrics_from_ranks(np.array(v)); res["readout_val"][k] = m["R@1"]
        print(f"readout {k:6s} val R@1 {m['R@1']:.4f}", flush=True)
    json.dump(res, open(RT / "explore" / "teachers_h010.json", "w"), indent=1)


if __name__ == "__main__":
    main()
