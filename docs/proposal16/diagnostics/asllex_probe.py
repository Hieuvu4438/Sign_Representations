"""proposal16 training-free diagnostic, step 3: which descriptor family linearly exposes which ASL-LEX phonological
parameter? Signer-independent: probes are fit on (half of) official train and scored on official val.

Probe = StandardScaler -> PCA(256, fit on train) -> multinomial logistic regression (class_weight balanced).
Metric = balanced accuracy on val (chance = 1 / #classes). Labels come from ASL-LEX 2.0 `signdata.csv`
(join on the `ASL-LEX Code` column of the ASL Citizen split CSVs).

Usage: python asllex_probe.py --val val_pose.npz --train train_pose_half.npz --asllex signdata.csv --out probe.json
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from sig_diagnostics import CONFIGS, build

SPLITS = "/home/dongvk/datasets/ASL_Citizen/splits"
PARAMS = {
    "SignType": ("SignType.2.0", ["OneHanded", "SymmetricalOrAlternating", "AsymmetricalDifferentHandshape",
                                  "AsymmetricalSameHandshape"]),
    "Movement": ("Movement.2.0", ["Straight", "Curved", "Circular", "Z-shaped", "BackAndForth"]),
    "Repeated": ("RepeatedMovement.2.0", ["0", "1"]),
    "MajorLoc": ("MajorLocation.2.0", ["Neutral", "Head", "Hand", "Body", "Arm"]),
    "Contact": ("Contact.2.0", ["0", "1"]),
    "FlexChange": ("FlexionChange.2.0", ["0", "1"]),
    "WristTwist": ("UlnarRotation.2.0", ["0", "1"]),
    "SelFingers": ("SelectedFingers.2.0", ["imrp", "i", "im", "t", "m", "p"]),
    "Flexion": ("Flexion.2.0", ["FullyOpen", "Curved", "Flat", "Bent", "FullyClosed"]),
}
SETS = ["static(all)", "bins8(all)", "vel8(all)", "sig glob+loc4 (man)", "sig glob+loc4+base (man)",
        "sig glob+loc4+base+LL2 (man)", "sig glob+loc4+base smooth5 (man)"]


def code_map():
    m = {}
    for sp in ("train", "val", "test"):
        with open(f"{SPLITS}/{sp}.csv") as f:
            for r in csv.DictReader(f):
                m[r["Video file"].rsplit(".", 1)[0]] = r["ASL-LEX Code"]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--asllex", required=True)
    ap.add_argument("--meta_val", default="/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc/metadata_val.jsonl")
    ap.add_argument("--meta_train", default="/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc/metadata_train.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.asllex, encoding="latin-1") as f:
        lex = {r["Code"]: r for r in csv.DictReader(f)}
    stem = {}
    for p in (a.meta_val, a.meta_train):
        for l in open(p):
            r = json.loads(l)
            stem[r["sample_id"]] = r["source_stem"]
    cmap = code_map()

    def load(p):
        z = np.load(p, allow_pickle=False)
        return {"x": z["x"], "lens": z["lens"], "meta": json.loads(str(z["meta"]))}

    val, tr = load(a.val), load(a.train)

    def labels(data, col, classes):
        out = []
        for m in data["meta"]:
            row = lex.get(cmap.get(stem.get(m["sample_id"], ""), "NA"))
            v = None if row is None else (row.get(col) or "").strip()
            if v and v.replace(".", "", 1).isdigit():
                v = str(int(float(v)))
            out.append(classes.index(v) if v in classes else -1)
        return np.array(out)

    feats = {}
    for i, s in enumerate(SETS):
        t0 = time.time()
        cache = f"{a.out}.feat{i}.npz"
        try:
            z = np.load(cache)
            feats[s] = (z["g"], z["v"])
        except FileNotFoundError:
            g, v = build(tr, CONFIGS[s]), build(val, CONFIGS[s])
            feats[s] = (np.concatenate([g[k] for k in g], 1), np.concatenate([v[k] for k in v], 1))
            np.savez(cache, g=feats[s][0], v=feats[s][1])
        print("built", s, feats[s][0].shape, f"{time.time() - t0:.0f}s", flush=True)
    feats["bins8+sigLL2"] = (np.concatenate([feats["bins8(all)"][0], feats["sig glob+loc4+base+LL2 (man)"][0]], 1),
                             np.concatenate([feats["bins8(all)"][1], feats["sig glob+loc4+base+LL2 (man)"][1]], 1))
    res = {}
    for s, (G, V) in feats.items():
        sc = StandardScaler().fit(G)
        Gs, Vs = np.nan_to_num(sc.transform(G)), np.nan_to_num(sc.transform(V))
        pca = PCA(256, random_state=0).fit(Gs)
        Gp, Vp = pca.transform(Gs), pca.transform(Vs)
        res[s] = {}
        for name, (col, classes) in PARAMS.items():
            yg, yv = labels(tr, col, classes), labels(val, col, classes)
            mg, mv = yg >= 0, yv >= 0
            if mg.sum() == 0 or mv.sum() == 0:
                print("no labels for", name, flush=True)
                continue
            clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(Gp[mg], yg[mg])
            ba = balanced_accuracy_score(yv[mv], clf.predict(Vp[mv]))
            res[s][name] = dict(bal_acc=100 * ba, chance=100 / len(classes), n_val=int(mv.sum()))
        print(s, {k: round(v["bal_acc"], 1) for k, v in res[s].items()}, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
