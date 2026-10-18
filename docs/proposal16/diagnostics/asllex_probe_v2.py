"""D-LEX v2: re-run the ASL-LEX probes with a wider bottleneck (PCA 1024, or all dims when smaller) and with
'static + X' combinations, to test whether signature descriptors add information *beyond* static position statistics.
Reuses the feature caches written by asllex_probe.py (<out>.feat{i}.npz). Same signer-independent split and metric.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

import asllex_probe as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--asllex", required=True)
    ap.add_argument("--cache", required=True, help="prefix used by asllex_probe.py --out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca", type=int, default=1024)
    a = ap.parse_args()
    import csv
    with open(a.asllex, encoding="latin-1") as f:
        lex = {r["Code"]: r for r in csv.DictReader(f)}
    stem = {}
    for p in (P.__dict__.get("META_VAL", "/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc/metadata_val.jsonl"),
              "/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc/metadata_train.jsonl"):
        for l in open(p):
            r = json.loads(l)
            stem[r["sample_id"]] = r["source_stem"]
    cmap = P.code_map()
    meta_v = json.loads(str(np.load(a.val)["meta"]))
    meta_t = json.loads(str(np.load(a.train)["meta"]))

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
    st = f["static(all)"]
    sets = {
        "static": st,
        "bins8": f["bins8(all)"],
        "sig+base smooth5": f["sig glob+loc4+base smooth5 (man)"],
        "sig+base LL2": f["sig glob+loc4+base+LL2 (man)"],
        "static+vel8": tuple(np.concatenate([st[j], f["vel8(all)"][j]], 1) for j in (0, 1)),
        "static+sig smooth5": tuple(np.concatenate([st[j], f["sig glob+loc4+base smooth5 (man)"][j]], 1) for j in (0, 1)),
        "static+sig LL2": tuple(np.concatenate([st[j], f["sig glob+loc4+base+LL2 (man)"][j]], 1) for j in (0, 1)),
    }
    res = {}
    for s, (G, V) in sets.items():
        sc = StandardScaler().fit(G)
        Gs, Vs = np.nan_to_num(sc.transform(G)), np.nan_to_num(sc.transform(V))
        k = min(a.pca, Gs.shape[1])
        pca = PCA(k, random_state=0).fit(Gs)
        Gp, Vp = pca.transform(Gs), pca.transform(Vs)
        res[s] = {"pca": k}
        for name, (col, classes) in P.PARAMS.items():
            yg, yv = labels(meta_t, col, classes), labels(meta_v, col, classes)
            mg, mv = yg >= 0, yv >= 0
            clf = LogisticRegression(max_iter=3000, C=0.05, class_weight="balanced").fit(Gp[mg], yg[mg])
            res[s][name] = 100 * balanced_accuracy_score(yv[mv], clf.predict(Vp[mv]))
        print(s, {k2: (round(v, 1) if isinstance(v, float) else v) for k2, v in res[s].items()}, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
