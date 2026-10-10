"""EX03 robustness diagnostic: linear ridge G0 vs G1 (same examples) for hand edges, to
check that the null gate result is not an artefact of the neural forecasters. Train on
signer folds 1-4, test on fold 0; also a Huber-robust error summary (median abs error)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4 import cip_forecast as CF
RT = Path(__file__).resolve().parents[1] / "runtime" / "cip"
S = np.load(RT / "states.npy", mmap_mode="r"); V = np.load(RT / "valid.npy", mmap_mode="r"); off = np.load(RT / "offsets.npy")
vids = [json.loads(l) for l in open(RT / "videos.jsonl")]; fold = np.array([v["fold"] for v in vids])
def build(vset, edge, hi, n, seed):
    a, b = CF.EDGES[edge]; d = CF.HORIZONS[hi]; rng = np.random.default_rng(seed)
    X0, X1, Y, M = [], [], [], []
    vset = vset[(off[vset + 1] - off[vset]) >= CF.K + d]
    while len(Y) < n:
        v = vset[rng.integers(len(vset))]; T = off[v + 1] - off[v]; t = int(rng.integers(CF.K - 1, T - d))
        s = off[v] + t - CF.K + 1
        mb, ma = V[s:s + CF.K, b, :43], V[s:s + CF.K, a, :43]
        if mb.any(-1).sum() < 6 or ma.any(-1).sum() < 6 or not V[off[v] + t + d, b, :43].any(): continue
        hb = np.where(mb, S[s:s + CF.K, b, :43], 0).astype(np.float32).ravel(); ha = np.where(ma, S[s:s + CF.K, a, :43], 0).astype(np.float32).ravel()
        X0.append(np.r_[hb, mb.ravel()]); X1.append(np.r_[hb, mb.ravel(), ha, ma.ravel()])
        Y.append(S[off[v] + t + d, b, :43].astype(np.float32)); M.append(V[off[v] + t + d, b, :43])
    return np.array(X0), np.array(X1), np.array(Y), np.array(M)
def ridge(X, Y, lam=10.0):
    X = np.c_[X, np.ones(len(X))]; return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
pred = lambda W, X: np.c_[X, np.ones(len(X))] @ W
for name, edge in (("LH->RH", CF.EDGES.index((0, 1))), ("RH->LH", CF.EDGES.index((1, 0)))):
    for hi in range(3):
        X0, X1, Y, M = build(np.where((fold >= 1))[0], edge, hi, 60000, 1)
        T0, T1, TY, TM = build(np.where(fold == 0)[0], edge, hi, 20000, 2)
        r = {}
        for k, (Xtr, Xte) in {"g0": (X0, T0), "g1": (X1, T1)}.items():
            e = (pred(ridge(Xtr, np.where(M, Y, 0)), Xte) - TY); e = np.where(TM, e, 0)
            r[k] = ((e ** 2).sum(1) / TM.sum(1).clip(1)).mean(); r[k + "_medabs"] = np.median(np.abs(e[TM]))
        print(f"{name}@{CF.HORIZONS[hi]} ridge mse g0 {r['g0']:.3f} g1 {r['g1']:.3f} rel {(r['g0']-r['g1'])/r['g0']:+.3f} | med|e| g0 {r['g0_medabs']:.3f} g1 {r['g1_medabs']:.3f}", flush=True)
