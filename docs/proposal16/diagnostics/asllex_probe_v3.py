"""D-LEX v3 (after independent review): gloss-disjoint AND signer-independent ASL-LEX probes, equal PCA dimension for
every descriptor, a mean-only static descriptor, and bootstrap CIs over held-out glosses.

Split: glosses are randomly partitioned (seed 0) into probe-train glosses (70 %) and probe-test glosses (30 %).
Probes are fit on train-half clips of probe-train glosses and scored on val clips of probe-test glosses, so a probe can
only succeed by encoding the phonological parameter, not by recognising the gloss. Metric = balanced accuracy;
95 % CI = percentile bootstrap over test glosses (500 resamples).
Reuses feature caches of asllex_probe.py (<cache>.feat{i}.npz) and builds the mean-only static descriptor itself.
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

import asllex_probe as P
from sig_diagnostics import build

META = "/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc/metadata_{}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--asllex", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    with open(a.asllex, encoding="latin-1") as f:
        lex = {r["Code"]: r for r in csv.DictReader(f)}
    stem = {}
    for sp in ("val", "train"):
        for l in open(META.format(sp)):
            r = json.loads(l)
            stem[r["sample_id"]] = r["source_stem"]
    cmap = P.code_map()

    def load(p):
        z = np.load(p, allow_pickle=False)
        return {"x": z["x"], "lens": z["lens"], "meta": json.loads(str(z["meta"]))}

    val, tr = load(a.val), load(a.train)
    gl_t = np.array([m["gloss"] for m in tr["meta"]])
    gl_v = np.array([m["gloss"] for m in val["meta"]])
    rng = np.random.default_rng(0)
    glosses = np.unique(np.concatenate([gl_t, gl_v]))
    test_g = set(rng.choice(glosses, size=int(0.3 * len(glosses)), replace=False).tolist())
    mt = np.array([g not in test_g for g in gl_t])
    mv = np.array([g in test_g for g in gl_v])

    def labels(meta, col, classes):
        out = []
        for m in meta:
            row = lex.get(cmap.get(stem.get(m["sample_id"], ""), "NA"))
            v = None if row is None else (row.get(col) or "").strip()
            if v and v.replace(".", "", 1).isdigit():
                v = str(int(float(v)))
            out.append(classes.index(v) if v in classes else -1)
        return np.array(out)

    f = {}
    for i, s in enumerate(P.SETS):
        z = np.load(f"{a.cache}.feat{i}.npz")
        f[s] = (z["g"], z["v"])
    cfg = dict(kind="bins", groups=["joint", "edge", "chain", "relation", "face"], k_local=1)  # = clip mean only
    g, v = build(tr, cfg), build(val, cfg)
    f["static mean-only"] = (np.concatenate([g[k] for k in g], 1), np.concatenate([v[k] for k in v], 1))
    st = f["static(all)"]
    sets = {
        "static mean-only": f["static mean-only"],
        "static mean+std": st,
        "bins8": f["bins8(all)"],
        "vel8": f["vel8(all)"],
        "sig+base smooth5": f["sig glob+loc4+base smooth5 (man)"],
        "sig+base LL2": f["sig glob+loc4+base+LL2 (man)"],
        "sig no-base (glob+loc4)": f["sig glob+loc4 (man)"],
        "mean-only + sig smooth5": tuple(np.concatenate([f["static mean-only"][j], f["sig glob+loc4+base smooth5 (man)"][j]], 1) for j in (0, 1)),
        "mean+std + sig smooth5": tuple(np.concatenate([st[j], f["sig glob+loc4+base smooth5 (man)"][j]], 1) for j in (0, 1)),
    }
    res = {"split": {"n_test_glosses": len(test_g), "train_clips": int(mt.sum()), "val_clips": int(mv.sum()),
                     "pca": a.pca}}
    for s, (G, V) in sets.items():
        G, V = G[mt], V[mv]
        sc = StandardScaler().fit(G)
        Gs, Vs = np.nan_to_num(sc.transform(G)), np.nan_to_num(sc.transform(V))
        pca = PCA(min(a.pca, Gs.shape[1]), random_state=0).fit(Gs)
        Gp, Vp = pca.transform(Gs), pca.transform(Vs)
        res[s] = {"dim": int(G.shape[1])}
        for name, (col, classes) in P.PARAMS.items():
            yg = labels([m for m, k in zip(tr["meta"], mt) if k], col, classes)
            yv = labels([m for m, k in zip(val["meta"], mv) if k], col, classes)
            okg, okv = yg >= 0, yv >= 0
            clf = LogisticRegression(max_iter=3000, C=0.05, class_weight="balanced").fit(Gp[okg], yg[okg])
            pred = clf.predict(Vp[okv])
            yt, gv = yv[okv], gl_v[mv][okv]
            ba = 100 * balanced_accuracy_score(yt, pred)
            ug = np.unique(gv)
            idx = {u: np.where(gv == u)[0] for u in ug}
            bs = []
            for _ in range(a.boot):
                pick = np.concatenate([idx[u] for u in rng.choice(ug, size=len(ug), replace=True)])
                if len(np.unique(yt[pick])) < 2:
                    continue
                bs.append(100 * balanced_accuracy_score(yt[pick], pred[pick]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            res[s][name] = [round(ba, 1), round(lo, 1), round(hi, 1)]
        print(s, {k: v for k, v in res[s].items()}, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
