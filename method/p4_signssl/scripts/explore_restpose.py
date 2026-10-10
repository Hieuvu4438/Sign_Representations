"""H012 diagnostic (no training): rest-pose pseudo-signer clusters for cross-signer label-free mining.
Rest descriptor of a clip = mean of L2-normalised windows OUTSIDE the active span (gamma >= 0.5);
fallback first+last window when the active span covers the clip. K-means on unlabelled ASL Citizen
train rest descriptors; K chosen by silhouette (label-free) on a K grid. Signer IDs / glosses only score.
 P1 NMI(cluster, signer) for each K.
 P2 mining on the same 3,000 train anchors as explore_teachers.py: T4 rerank of the 64 wmean candidates,
    without / with exclusion of candidates in the anchor's pseudo-signer cluster (+ oracle true-signer exclusion).
 P3 rest descriptor as a val readout (class-max over train gallery): gloss R@1 must be low.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import class_max_scores, l2n, metrics_from_ranks, ranks_from_scores  # noqa: E402
from diag_order import activity_weights, active_span, load, pad, seqs  # noqa: E402
from explore_teachers import K_DIFF, q2g  # noqa: E402

RT = Path(__file__).resolve().parents[1] / "runtime"


def rest_desc(off, win, act, thr=0.5):
    out, fb = [], 0
    for i in range(len(off) - 1):
        w = l2n(win[off[i]:off[i + 1]].astype(np.float32)); g = activity_weights(act[off[i]:off[i + 1]])
        a, b = active_span(g, thr)
        r = np.concatenate([w[:a], w[b:]])
        if len(r) == 0:
            r = w[[0, -1]]; fb += 1
        out.append(r.mean(0))
    return l2n(np.stack(out)), fb


def tdiff(w):
    return l2n(w[K_DIFF:] - w[:-K_DIFF]) if len(w) > K_DIFF else l2n(w[-1:] - w[:1] + 1e-6)


def main():
    gc, goff, gw, ga = load(RT / "feats" / "aslc_train_s2"); qc, qoff, qw, qa = load(RT / "feats" / "aslc_val_s2")
    res = {}
    GR, fb = rest_desc(goff, gw, ga); QR, qfb = rest_desc(qoff, qw, qa)
    res["fallback_rate_train"] = fb / len(GR); res["fallback_rate_val"] = qfb / len(QR)
    sg = np.array([c["signer"] for c in gc]); gl = np.array([c["gloss"] for c in gc])
    res["n_train_signers"] = int(len(set(sg)))
    srng = np.random.default_rng(1); samp = srng.choice(len(GR), 6000, replace=False)
    res["K"] = {}
    best = None
    for K in (20, 40, 60, 80, 120):
        km = KMeans(K, n_init=4, random_state=0).fit(GR)
        sil = float(silhouette_score(GR[samp], km.labels_[samp], metric="cosine"))
        nmi = float(normalized_mutual_info_score(sg, km.labels_))
        purity = float(sum(np.bincount(np.unique(sg, return_inverse=True)[1][km.labels_ == k]).max() for k in range(K) if (km.labels_ == k).any()) / len(sg))
        res["K"][K] = {"silhouette": sil, "nmi": nmi, "purity": purity}
        print(f"K={K} silhouette {sil:.3f} NMI {nmi:.3f} purity {purity:.3f}", flush=True)
        if best is None or sil > best[1]:
            best = (K, sil, km.labels_)
    K, _, cl = best; res["K_selected"] = K
    # P3: rest descriptor as a val gloss readout
    labs = sorted(set(gl)); lab = {g: i for i, g in enumerate(labs)}
    gy = np.array([lab[x] for x in gl]); qy = np.array([lab[c["gloss"]] for c in qc])
    res["P3_rest_readout_val_R@1"] = metrics_from_ranks(ranks_from_scores(class_max_scores(QR, GR, gy, len(labs), device="cuda"), qy))["R@1"]
    print("P3 rest readout val R@1", res["P3_rest_readout_val_R@1"], flush=True)
    # P2: mining
    gs = seqs(goff, gw, ga)
    rng = np.random.default_rng(0)
    rng.choice(sum(len(w) for w, _ in gs), 300000, replace=False)
    anchors = rng.choice(len(gs), 3000, replace=False)
    GW = l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in gs]))
    P0 = pad(gs, 48); P1 = pad([(tdiff(w), None) for w, _ in gs], 48)
    P0 = (torch.from_numpy(P0[0].astype(np.float16)), torch.from_numpy(P0[1]))
    P1 = (torch.from_numpy(P1[0].astype(np.float16)), torch.from_numpy(P1[1]))
    Wt = torch.from_numpy(GW).cuda()
    tops = {"T4": [], "T4_xcluster": [], "T4_xsigner_ORACLE": [], "wmean_xcluster": []}
    for i in anchors:
        sc = Wt[i] @ Wt.T; sc[i] = -9
        cand = sc.topk(64).indices.cpu().numpy()
        v = sum(q2g(X[i].cuda(), M[i].cuda(), X[cand].cuda(), M[cand].cuda()).float().cpu().numpy() for X, M in (P0, P1))
        o = cand[np.argsort(-v)]
        tops["T4"].append(o[:5])
        for k, keep in (("T4_xcluster", cl[o] != cl[i]), ("T4_xsigner_ORACLE", sg[o] != sg[i])):
            oo = o[keep]; tops[k].append(np.pad(oo[:5], (0, 5 - len(oo[:5])), constant_values=-1))
        cand_x = cand[cl[cand] != cl[i]]
        tops["wmean_xcluster"].append(np.pad(cand_x[:5], (0, 5 - len(cand_x[:5])), constant_values=-1))
    res["mining"] = {}
    for k, t in tops.items():
        t = np.array(t); valid = t >= 0; tt = np.where(valid, t, 0)
        r = {}
        for n in (1, 5):
            sel, vv = tt[:, :n], valid[:, :n]
            same_g = (gl[sel] == gl[anchors][:, None]) & vv; same_s = (sg[sel] == sg[anchors][:, None]) & vv
            r[f"top{n}_same_gloss"] = float(same_g.sum() / vv.sum()); r[f"top{n}_same_signer"] = float(same_s.sum() / vv.sum())
            r[f"top{n}_same_gloss_diff_signer"] = float((same_g & ~same_s).sum() / vv.sum())
        r["coverage_top1"] = float(valid[:, 0].mean())
        res["mining"][k] = r
        print(f"{k:18s} top1 gloss {r['top1_same_gloss']:.3f} signer {r['top1_same_signer']:.3f} gloss&diffsigner {r['top1_same_gloss_diff_signer']:.3f} | top5 g&ds {r['top5_same_gloss_diff_signer']:.3f} cov {r['coverage_top1']:.3f}", flush=True)
    np.save(RT / "explore" / "restpose_clusters_train.npy", cl)
    json.dump(res, open(RT / "explore" / "restpose_h012.json", "w"), indent=1)


if __name__ == "__main__":
    main()
