"""proposal16 training-free diagnostics, step 2: which skeleton signals carry lexical information?

Compares hand-crafted clip descriptors on ASL Citizen (no training, no labels used to build features):
static positions, time-binned coordinates, velocities, and path log-signatures (level 1 + level 2 Levy areas)
over Path-JEPA-style path families (joint / edge / chain) plus sign-specific relational paths (hand-face,
hand-hand) with/without global+local windows, basepoint, lead-lag and time channels.

Metrics (all on official val, labels used only for scoring):
- VALDICT R@1: gallery = every 2nd official-train clip (20k, 2,731 glosses), query = all val (10,304),
  class-max cosine, pessimistic ties (mirrors the official dictionary protocol on a dev split).
- LOSO R@1: query signer vs. the other val signers.
- signerNN: share of nearest neighbours (excluding same gloss) that come from the query's signer (lower = less
  signer leakage); chance = sum_s p_s^2.
- tempo cos: mean cosine between a clip descriptor and the same clip resampled to 0.67x / 1.5x length.
- eps2 gloss / signer: adjusted variance share (epsilon^2) of descriptor dims explained by gloss / signer (train half).

Usage: python sig_diagnostics.py --val <val_pose.npz> --train <train_pose_half.npz> --out results.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

# joint layout from prep_pose.py
NOSE, DSHO, NSHO, DELB, NELB, DWRI, NWRI = range(7)
DH = list(range(7, 28))
NH = list(range(28, 49))
FACE = list(range(49, 66))
HAND_PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]


# ----------------------------------------------------------------------------------------------- paths
def path_families(x: np.ndarray) -> dict[str, np.ndarray]:
    """x [T,66,2] body frame -> {family: [T,P,d]} (paths of one family share d)."""
    out = {}
    out["J_wrist"] = x[:, [DWRI, NWRI]]                                    # absolute wrists (body frame)
    out["J_handD"] = x[:, DH[1:]] - x[:, DH[:1]]                          # D-hand joints rel. hand root
    out["J_handN"] = x[:, NH[1:]] - x[:, NH[:1]]
    dh = x[:, DH]
    nh = x[:, NH]
    out["E_handD"] = np.stack([dh[:, c] - dh[:, p] for c, p in enumerate(HAND_PARENT) if p >= 0], 1)
    out["E_handN"] = np.stack([nh[:, c] - nh[:, p] for c, p in enumerate(HAND_PARENT) if p >= 0], 1)
    out["E_arm"] = np.stack([x[:, DELB] - x[:, DSHO], x[:, DWRI] - x[:, DELB],
                             x[:, NELB] - x[:, NSHO], x[:, NWRI] - x[:, NELB]], 1)
    # chains: concatenated edge displacements (d = 2 * #edges)
    out["C_armD"] = np.concatenate([x[:, DELB] - x[:, DSHO], x[:, DWRI] - x[:, DELB]], -1)[:, None]
    out["C_armN"] = np.concatenate([x[:, NELB] - x[:, NSHO], x[:, NWRI] - x[:, NELB]], -1)[:, None]
    out["C_fingD"] = np.stack([np.concatenate([dh[:, f[k + 1]] - dh[:, f[k]] for k in range(4)], -1)
                               for f in FINGERS], 1)
    out["C_fingN"] = np.stack([np.concatenate([nh[:, f[k + 1]] - nh[:, f[k]] for k in range(4)], -1)
                               for f in FINGERS], 1)
    # sign-specific relational paths
    out["R_handFace"] = np.stack([x[:, DWRI] - x[:, NOSE], x[:, DH[8]] - x[:, NOSE],
                                  x[:, NWRI] - x[:, NOSE]], 1)            # wrist/index-tip vs nose
    out["R_handHand"] = np.stack([x[:, DWRI] - x[:, NWRI], x[:, DH[8]] - x[:, NH[0]]], 1)
    out["X_twoHand"] = np.concatenate([x[:, DWRI], x[:, NWRI]], -1)[:, None]  # 4-D: cross-hand Levy areas
    out["F_face"] = x[:, FACE] - x[:, [NOSE]]
    return out


GROUPS = {
    "joint": ["J_wrist", "J_handD", "J_handN"],
    "edge": ["E_handD", "E_handN", "E_arm"],
    "chain": ["C_armD", "C_armN", "C_fingD", "C_fingN"],
    "relation": ["R_handFace", "R_handHand", "X_twoHand"],
    "face": ["F_face"],
}


# --------------------------------------------------------------------------------------- signatures
def logsig2(p: np.ndarray) -> np.ndarray:
    """Exact degree-2 log-signature of the piecewise-linear path p [T,P,d] -> [P, d + d(d-1)/2]."""
    if len(p) < 2:
        d = p.shape[-1]
        return np.zeros((p.shape[1], d + d * (d - 1) // 2))
    inc = np.diff(p, axis=0)
    rel = p[:-1] - p[:1]
    a = np.einsum("tpi,tpj->pij", rel, inc)
    area = 0.5 * (a - np.swapaxes(a, 1, 2))
    iu = np.triu_indices(p.shape[-1], 1)
    return np.concatenate([p[-1] - p[0], area[:, iu[0], iu[1]]], -1)


def lead_lag(p: np.ndarray, delay: int) -> np.ndarray:
    lag = np.concatenate([np.repeat(p[:1], delay, 0), p[:-delay]], 0)
    return np.concatenate([p, lag], -1)


def with_time(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.concatenate([p, np.broadcast_to(t[:, None, None], p.shape[:2] + (1,))], -1)


def windows(T: int, k: int):
    edges = np.linspace(0, T - 1, k + 1).round().astype(int)
    return [(edges[i], max(edges[i + 1], edges[i] + 1)) for i in range(k)]


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    k = np.ones(w) / w
    pad = np.concatenate([np.repeat(x[:1], w // 2, 0), x, np.repeat(x[-1:], w - 1 - w // 2, 0)], 0)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, pad)


# --------------------------------------------------------------------------------------- descriptors
def describe(x: np.ndarray, fps: float, cfg: dict) -> dict[str, np.ndarray]:
    """One clip -> {block_name: vector}. cfg keys: kind, groups, k_local, glob, base, ll, time, smooth, lv."""
    x = np.nan_to_num(x.astype(np.float64))
    if cfg.get("smooth", 1) > 1:
        x = smooth(x, cfg["smooth"])
    fam = path_families(x)
    T = len(x)
    t = np.arange(T) / max(fps, 1.0)
    blocks = {}
    for g in cfg["groups"]:
        for name in GROUPS[g]:
            p = fam[name]
            vecs = []
            kind = cfg["kind"]
            if kind == "static":
                vecs += [p.mean(0).ravel(), p.std(0).ravel()]
            elif kind == "bins":
                for a, b in windows(T, cfg.get("k_local", 8)):
                    vecs.append(p[a:b + 1].mean(0).ravel())
            elif kind == "vel":
                v = np.diff(p, axis=0) * fps if T > 1 else np.zeros_like(p)
                for a, b in windows(max(T - 1, 2), cfg.get("k_local", 8)):
                    vecs.append(v[a:b + 1].mean(0).ravel())
                    vecs.append(np.abs(v[a:b + 1]).mean(0).ravel())
            elif kind == "sig":
                q = p
                if cfg.get("time"):
                    q = with_time(q, t)
                if cfg.get("ll"):
                    q = lead_lag(q, cfg["ll"])
                lv = cfg.get("lv", 2)
                def ls(seg):
                    s = logsig2(seg)
                    d = q.shape[-1]
                    if lv == 1:
                        s = s[:, :d]
                    elif lv == -2:  # level-2 only
                        s = s[:, d:]
                    return s.ravel()
                if cfg.get("glob", True):
                    vecs.append(ls(q))
                for a, b in windows(T, cfg.get("k_local", 0)) if cfg.get("k_local", 0) else []:
                    vecs.append(ls(q[a:b + 1]))
                if cfg.get("base"):
                    vecs += [p[0].ravel(), p.mean(0).ravel()]
            blocks[name] = np.concatenate(vecs)
    return blocks


def build(data, cfg, resample: float = 1.0, idx=None):
    x, lens, meta = data["x"], data["lens"], data["meta"]
    off = np.concatenate([[0], np.cumsum(lens)])
    ids = range(len(lens)) if idx is None else idx
    rows = []
    for i in ids:
        clip = x[off[i]:off[i + 1]].astype(np.float64)
        if resample != 1.0:
            T = len(clip)
            n = max(2, int(round(T * resample)))
            src = np.linspace(0, T - 1, n)
            lo = np.floor(src).astype(int)
            hi = np.minimum(lo + 1, T - 1)
            w = (src - lo)[:, None, None]
            clip = clip[lo] * (1 - w) + clip[hi] * w
            fps = meta[i]["fps"] * resample
        else:
            fps = meta[i]["fps"]
        rows.append(describe(clip, fps, cfg))
    names = list(rows[0].keys())
    return {n: np.stack([r[n] for r in rows]).astype(np.float32) for n in names}


class Normaliser:
    """z-score per dim and equal weight per block (fit on the gallery set only)."""

    def fit(self, blocks):
        self.stats = {}
        for n, b in blocks.items():
            mu, sd = b.mean(0), b.std(0)
            sd[sd < 1e-6] = 1.0
            self.stats[n] = (mu, sd, np.sqrt(b.shape[1]))
        return self

    def __call__(self, blocks):
        parts = [(blocks[n] - mu) / sd / sq for n, (mu, sd, sq) in self.stats.items()]
        f = np.concatenate(parts, 1)
        return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)


# ------------------------------------------------------------------------------------------ metrics
def dict_r1(q, qy, g, gy, q_sig=None, g_sig=None, chunk=1024):
    """Class-max cosine retrieval; returns R@1, R@5 (pessimistic ties). Optional signer exclusion (LOSO)."""
    order = np.argsort(gy, kind="stable")
    g, gy_s = g[order], gy[order]
    g_sig_s = None if g_sig is None else g_sig[order]
    classes, starts = np.unique(gy_s, return_index=True)
    cpos = {c: i for i, c in enumerate(classes)}
    hits1 = hits5 = 0
    for a in range(0, len(q), chunk):
        s = q[a:a + chunk] @ g.T
        if q_sig is not None:
            s = np.where(q_sig[a:a + chunk, None] == g_sig_s[None, :], -np.inf, s)
        cs = np.maximum.reduceat(s, starts, axis=1)
        for r, y in enumerate(qy[a:a + chunk]):
            j = cpos.get(y)
            if j is None or not np.isfinite(cs[r, j]):
                continue
            rank = int((cs[r] >= cs[r, j]).sum()) - 1
            hits1 += rank == 0
            hits5 += rank < 5
    return 100.0 * hits1 / len(q), 100.0 * hits5 / len(q)


def signer_nn(f, y, sig, chunk=1024):
    same = 0
    for a in range(0, len(f), chunk):
        s = f[a:a + chunk] @ f.T
        s[y[a:a + chunk, None] == y[None, :]] = -np.inf
        nn = s.argmax(1)
        same += int((sig[nn] == sig[a:a + chunk]).sum())
    _, c = np.unique(sig, return_counts=True)
    return 100.0 * same / len(f), 100.0 * float(((c / c.sum()) ** 2).sum())


def eps2(f, groups):
    """Mean adjusted epsilon^2 over dims of f [n,D] for a grouping."""
    _, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    k, n = len(cnt), len(f)
    mu = f.mean(0)
    sums = np.zeros((k, f.shape[1]))
    np.add.at(sums, inv, f)
    means = sums / cnt[:, None]
    ssb = (cnt[:, None] * (means - mu) ** 2).sum(0)
    sst = ((f - mu) ** 2).sum(0)
    msw = (sst - ssb) / max(n - k, 1)
    e = (ssb - (k - 1) * msw) / np.maximum(sst, 1e-12)
    return float(np.mean(e))


# -------------------------------------------------------------------------------------------- configs
ALL = ["joint", "edge", "chain", "relation", "face"]
MAN = ["joint", "edge", "chain", "relation"]
CONFIGS = {
    # baselines
    "static(all)": dict(kind="static", groups=ALL),
    "bins8(all)": dict(kind="bins", groups=ALL, k_local=8),
    "vel8(all)": dict(kind="vel", groups=ALL, k_local=8),
    # Path-JEPA style signatures, manual families
    "sig glob L1 (man)": dict(kind="sig", groups=MAN, lv=1),
    "sig glob L2only (man)": dict(kind="sig", groups=MAN, lv=-2),
    "sig glob L1+L2 (man)": dict(kind="sig", groups=MAN),
    "sig glob+loc4 (man)": dict(kind="sig", groups=MAN, k_local=4),
    "sig glob+loc4+base (man)": dict(kind="sig", groups=MAN, k_local=4, base=True),
    "sig glob+loc4+base+LL2 (man)": dict(kind="sig", groups=MAN, k_local=4, base=True, ll=2),
    "sig glob+loc4+base+time (man)": dict(kind="sig", groups=MAN, k_local=4, base=True, time=True),
    "sig glob+loc4+base smooth5 (man)": dict(kind="sig", groups=MAN, k_local=4, base=True, smooth=5),
    "sig glob+loc4+base (all)": dict(kind="sig", groups=ALL, k_local=4, base=True),
    "sig loc4 only+base (man)": dict(kind="sig", groups=MAN, k_local=4, glob=False, base=True),
    # one family at a time (glob+loc4+base)
    "fam joint": dict(kind="sig", groups=["joint"], k_local=4, base=True),
    "fam edge": dict(kind="sig", groups=["edge"], k_local=4, base=True),
    "fam chain": dict(kind="sig", groups=["chain"], k_local=4, base=True),
    "fam relation": dict(kind="sig", groups=["relation"], k_local=4, base=True),
    "fam face": dict(kind="sig", groups=["face"], k_local=4, base=True),
    "fam relation nobase": dict(kind="sig", groups=["relation"], k_local=4),
    "fam relation static": dict(kind="static", groups=["relation"]),
    "fam joint static": dict(kind="static", groups=["joint"]),
    # combos
    "bins8+sig (all)": "COMBO",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--tempo_n", type=int, default=1000)
    a = ap.parse_args()

    def load(p):
        z = np.load(p, allow_pickle=False)
        return {"x": z["x"], "lens": z["lens"], "meta": json.loads(str(z["meta"]))}

    val, tr = load(a.val), load(a.train)
    vy = np.array([m["class_id"] for m in val["meta"]])
    vs = np.array([m["signer_id"] for m in val["meta"]])
    ty = np.array([m["class_id"] for m in tr["meta"]])
    tsg = np.array([m["signer_id"] for m in tr["meta"]])
    rng = np.random.default_rng(0)
    tempo_idx = rng.choice(len(vy), size=min(a.tempo_n, len(vy)), replace=False)
    results = {}
    names = [n for n in CONFIGS if not a.only or n in a.only.split(",")]
    cache = {}
    for name in names:
        t0 = time.time()
        cfg = CONFIGS[name]
        if cfg == "COMBO":
            gb = {**{f"b:{k}": v for k, v in cache["bins8(all)"][0].items()},
                  **{f"s:{k}": v for k, v in cache["sig glob+loc4+base (all)"][0].items()}}
            vb = {**{f"b:{k}": v for k, v in cache["bins8(all)"][1].items()},
                  **{f"s:{k}": v for k, v in cache["sig glob+loc4+base (all)"][1].items()}}
            tempo = None
        else:
            gb, vb = build(tr, cfg), build(val, cfg)
            cache[name] = (gb, vb)
            tempo = []
            norm0 = Normaliser().fit(gb)
            base = norm0({k: v[tempo_idx] for k, v in vb.items()})
            for f in (0.67, 1.5):
                rb = build(val, cfg, resample=f, idx=tempo_idx)
                tempo.append(float((base * norm0(rb)).sum(1).mean()))
        norm = Normaliser().fit(gb)
        G, Q = norm(gb), norm(vb)
        r1, r5 = dict_r1(Q, vy, G, ty)
        l1, l5 = dict_r1(Q, vy, Q, vy, q_sig=vs, g_sig=vs)
        snn, chance = signer_nn(Q, vy, vs)
        Graw = np.concatenate([(gb[n] - norm.stats[n][0]) / norm.stats[n][1] for n in gb], 1)
        e_g, e_s = eps2(Graw, ty), eps2(Graw, tsg)
        results[name] = dict(dim=int(G.shape[1]), valdict_r1=r1, valdict_r5=r5, loso_r1=l1, loso_r5=l5,
                             signer_nn=snn, signer_chance=chance,
                             tempo_cos_067=None if tempo is None else tempo[0],
                             tempo_cos_15=None if tempo is None else tempo[1],
                             eps2_gloss=e_g, eps2_signer=e_s, sec=time.time() - t0)
        print(f"{name:40s} D={G.shape[1]:5d} VALDICT R1={r1:5.2f} R5={r5:5.2f} LOSO R1={l1:5.2f} "
              f"sNN={snn:5.1f}/{chance:4.1f} tempo={results[name]['tempo_cos_067']} "
              f"eps2 g={e_g:.3f} s={e_s:.3f} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
