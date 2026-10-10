"""D-ORD: does cross-window temporal order carry retrieval information beyond the
(activity-weighted) bag of SignRep windows? Training-free readouts on the SAME cached
window embeddings:
  - wmean: SignRep Eq.12 activity-weighted mean (reference)
  - binsK: split the active span into K equal time bins (activity-weighted means per bin),
           concatenate L2-normalised bins (order-aware, rigid alignment)
  - chamfer / dtw rerank: for each query, take the top-M classes by wmean, rescore every
           gallery clip of those classes by chamfer similarity or DTW alignment of
           window sequences (active windows only), class score = max over clips.
Reports metrics on val queries. Diagnostic only (no selection of a method from this).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.retrieval import activity_weights, class_max_scores, l2n, metrics_from_ranks, ranks_from_scores  # noqa


def load(d, key="features"):
    d = Path(d)
    clips = [json.loads(l) for l in open(d / "clips.jsonl")]
    return clips, np.load(d / "offsets.npy"), np.load(d / f"windows_{key}.npy"), np.load(d / "windows_active.npy")


def active_span(g, thr):
    idx = np.where(g >= thr)[0]
    return (idx[0], idx[-1] + 1) if len(idx) else (0, len(g))


def seqs(off, win, act, thr=0.5):
    out = []
    for i in range(len(off) - 1):
        w = win[off[i]:off[i + 1]].astype(np.float32)
        g = activity_weights(act[off[i]:off[i + 1]])
        a, b = active_span(g, thr)
        out.append((l2n(w[a:b]), g[a:b]))
    return out


def binned(sq, K):
    out = np.zeros((len(sq), K * sq[0][0].shape[1]), np.float32)
    for i, (w, g) in enumerate(sq):
        n = len(w)
        edges = np.linspace(0, n, K + 1)
        parts = []
        for k in range(K):
            a, b = int(np.floor(edges[k])), max(int(np.ceil(edges[k + 1])), int(np.floor(edges[k])) + 1)
            ww, gg = w[a:b], g[a:b] + 1e-6
            parts.append(l2n((gg[:, None] * ww).sum(0, keepdims=True))[0])
        out[i] = np.concatenate(parts) / np.sqrt(K)
    return out


def pad(sq, L=None):
    L = L or max(len(w) for w, _ in sq)
    X = np.zeros((len(sq), L, sq[0][0].shape[1]), np.float32)
    M = np.zeros((len(sq), L), bool)
    for i, (w, _) in enumerate(sq):
        n = min(len(w), L)
        X[i, :n], M[i, :n] = w[:n], True
    return X, M


@torch.no_grad()
def seq_scores(qX, qM, gX, gM, mode):
    """qX [Lq,D], gX [B,Lg,D] -> [B] similarity."""
    S = torch.einsum("qd,bld->bql", qX, gX)  # B,Lq,Lg
    qm, gm = qM[None, :, None], gM[:, None, :]
    if mode == "chamfer":
        S1 = S.masked_fill(~gm, -9).amax(2)  # best gallery window per query window
        S2 = S.masked_fill(~qm, -9).amax(1)
        a = (S1 * qM[None]).sum(1) / qM.sum()
        b = (S2 * gM).sum(1) / gM.sum(1).clamp(min=1)
        return 0.5 * (a + b)
    # DTW on cost 1-S, normalised by path length via (Lq+Lg); band-free
    C = (1 - S).masked_fill(~(qm & gm), 1e3)
    B, Lq, Lg = C.shape
    D = torch.full((B, Lq + 1, Lg + 1), float("inf"), device=C.device)
    D[:, 0, 0] = 0
    for i in range(1, Lq + 1):
        for j in range(1, Lg + 1):
            D[:, i, j] = C[:, i - 1, j - 1] + torch.minimum(torch.minimum(D[:, i - 1, j], D[:, i, j - 1]), D[:, i - 1, j - 1])
    lq = int(qM.sum())
    lg = gM.sum(1)
    end = D[torch.arange(B), lq, lg]
    return -end / (lq + lg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topm", type=int, default=20)
    ap.add_argument("--maxq", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--modes", default="chamfer")
    a = ap.parse_args()
    gc, goff, gw, ga = load(a.gallery)
    qc, qoff, qw, qa = load(a.query)
    glosses = sorted({c["gloss"] for c in gc})
    lab = {g: i for i, g in enumerate(glosses)}
    gl = np.array([lab[c["gloss"]] for c in gc])
    ql = np.array([lab[c["gloss"]] for c in qc])
    gs, qs = seqs(goff, gw, ga, a.thr), seqs(qoff, qw, qa, a.thr)
    res = {}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def wmean(sq):
        return l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in sq]))

    gW, qW = wmean(gs), wmean(qs)
    base = class_max_scores(qW, gW, gl, len(glosses))
    r0 = ranks_from_scores(base, ql)
    res["wmean_activespan"] = metrics_from_ranks(r0, ql)
    np.save(out / "ranks_wmean.npy", r0)
    for K in (2, 3, 4):
        s = class_max_scores(binned(qs, K), binned(gs, K), gl, len(glosses))
        r = ranks_from_scores(s, ql)
        res[f"bins{K}"] = metrics_from_ranks(r, ql)
        res[f"bins{K}+wmean"] = metrics_from_ranks(ranks_from_scores(s + base, ql), ql)
        np.save(out / f"ranks_bins{K}.npy", r)
        print(K, res[f"bins{K}"], res[f"bins{K}+wmean"], flush=True)
    # rerank within top-M classes
    L = 48
    gX, gM = pad(gs, L)
    # gallery kept on CPU (pinned fp16); per-query items are moved to GPU (memory-safe on a shared GPU)
    gXt, gMt = torch.from_numpy(gX.astype(np.float16)).pin_memory(), torch.from_numpy(gM)
    by_class = [np.where(gl == c)[0] for c in range(len(glosses))]
    nq = a.maxq or len(qs)
    for mode in a.modes.split(","):
        rr, rr_fused = [], []
        for i in range(nq):
            top = np.argsort(-base[i])[: a.topm]
            items = np.concatenate([by_class[c] for c in top])
            qX, qM = pad([qs[i]], L)
            sc = seq_scores(torch.from_numpy(qX[0]).cuda().half(), torch.from_numpy(qM[0]).cuda(),
                            gXt[items].cuda(non_blocking=True), gMt[items].cuda(), mode).float().cpu().numpy()
            new = np.full(len(glosses), -1e9, np.float32)
            np.maximum.at(new, gl[items], sc)
            # outside top-M keep base order below all reranked classes
            fused_scores = np.where(new > -1e8, 10 + new, base[i])
            rr.append(ranks_from_scores(fused_scores[None], [ql[i]])[0])
            rr_fused.append(ranks_from_scores(np.where(new > -1e8, 10 + new + base[i], base[i])[None], [ql[i]])[0])
            if (i + 1) % 1000 == 0:
                print(mode, i + 1, metrics_from_ranks(np.array(rr)), flush=True)
        res[f"{mode}_rerank_top{a.topm}"] = metrics_from_ranks(np.array(rr), ql[:nq])
        res[f"{mode}+wmean_rerank_top{a.topm}"] = metrics_from_ranks(np.array(rr_fused), ql[:nq])
        res[f"wmean_on_same_{nq}"] = metrics_from_ranks(r0[:nq], ql[:nq])
        np.save(out / f"ranks_{mode}.npy", np.array(rr))
        print(mode, res[f"{mode}_rerank_top{a.topm}"], res[f"{mode}+wmean_rerank_top{a.topm}"], flush=True)
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
