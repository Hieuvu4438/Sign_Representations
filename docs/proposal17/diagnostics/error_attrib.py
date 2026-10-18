"""D-ERR: which phonological parameters do wrong retrievals agree / disagree on? (proposal17 diagnostic G4)

For every val query whose top-1 retrieved gloss is wrong, compare the ASL-LEX 2.0 parameters of the true and the retrieved gloss
and compare the agreement rate to the base rate of two random glosses. lift > 1 means the embedding preserves that parameter
(errors stay inside the phonological neighbourhood); lift ~ 1 means the parameter is NOT encoded in the geometry used by retrieval.
Run on three geometries of the same 768-d embedding: raw (protocol), label-free PCA-whitened, and gloss-LDA (supervised ceiling),
to see which parameters supervision / isotropy recover.  Labels are used for scoring only.
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np

from common import *  # noqa
import geo_headroom as G

SPLITS = "/home/dongvk/datasets/ASL_Citizen/splits"
LEX = "/home/haipd/Sign_Representations/docs/proposal16/data/signdata.csv"
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/error_attrib.json"
PARAMS = {
    "SignType": "SignType.2.0", "Movement": "Movement.2.0", "Repeated": "RepeatedMovement.2.0", "MajorLoc": "MajorLocation.2.0",
    "Contact": "Contact.2.0", "FlexChange": "FlexionChange.2.0", "WristTwist": "UlnarRotation.2.0", "SelFingers": "SelectedFingers.2.0",
    "Flexion": "Flexion.2.0", "Handshape(full)": "Handshape.2.0", "MinorLoc": "MinorLocation.2.0",
}


def gloss_lex(d):
    cm = {}
    for sp in ("train", "val"):
        for r in csv.DictReader(open(f"{SPLITS}/{sp}.csv")):
            cm[r["Video file"].rsplit(".", 1)[0]] = (r["Gloss"] if "Gloss" in r else None, r["ASL-LEX Code"])
    lex = {r["Code"]: r for r in csv.DictReader(open(LEX, encoding="latin-1"))}
    codes = {}
    for split in ("train", "val"):
        for stem, g in zip(d[split]["ids"], d[split]["y"]):
            if stem in cm:
                codes.setdefault(g, []).append(cm[stem][1])
    out = {}
    for g, cs in codes.items():
        c = max(set(cs), key=cs.count)
        row = lex.get(c)
        if row is None:
            continue
        out[g] = {p: (row.get(col) or "").strip() for p, col in PARAMS.items()}
    return out


def class_scores(Q, G_, yg):
    Q, G_ = l2(Q.astype(np.float32)), l2(G_.astype(np.float32))
    order = np.argsort(yg, kind="stable")
    Gs, ygs = G_[order], yg[order]
    classes, starts = np.unique(ygs, return_index=True)
    sc = np.empty((len(Q), len(classes)), np.float32)
    for a in range(0, len(Q), 512):
        sc[a : a + 512] = np.maximum.reduceat(Q[a : a + 512] @ Gs.T, starts, axis=1)
    return sc, classes


def main():
    d = load_baseline()
    tr, va = d["train"], d["val"]
    lexg = gloss_lex(d)
    rng = np.random.default_rng(0)
    # geometries
    geos = {}
    geos["raw"] = (va["X"], tr["X"])
    mu_w, W = G.fit_pca_whiten(tr["X"], 512, 1.0)
    geos["whitened"] = (G.apply_lin(va["X"], mu_w, W), G.apply_lin(tr["X"], mu_w, W))
    mu_d, Wd = G.fit_lda(tr["X"], tr["y"], 256, 0.05)
    geos["lda"] = (G.apply_lin(va["X"], mu_d, Wd), G.apply_lin(tr["X"], mu_d, Wd))
    gl = sorted(lexg)
    # base rates: random pairs of distinct glosses present in val queries
    qg = np.unique(va["y"])
    qg = np.array([g for g in qg if g in lexg])
    pairs = rng.choice(qg, size=(200000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    base = {}
    for p in PARAMS:
        a = [lexg[i][p] for i, j in pairs[:50000]]
        b = [lexg[j][p] for i, j in pairs[:50000]]
        ok = [(x != "" and y != "") for x, y in zip(a, b)]
        base[p] = float(np.mean([x == y for x, y, o in zip(a, b, ok) if o]))
    core = [p for p in PARAMS if p not in ("Handshape(full)", "MinorLoc")]
    hbase = []
    for i, j in pairs[:50000]:
        hbase.append(sum(lexg[i][p] != lexg[j][p] for p in core if lexg[i][p] != "" and lexg[j][p] != ""))
    hbase = np.array(hbase)
    out = {"base_agreement": base, "base_hamming_mean": float(hbase.mean()), "base_hamming_le1": float((hbase <= 1).mean()),
           "n_val_glosses_with_lex": int(len(qg))}
    for name, (Q, Gl) in geos.items():
        sc, classes = class_scores(Q, Gl, tr["y"])
        pred = classes[sc.argmax(1)]
        err = (pred != va["y"]) & np.array([(g in lexg) and (p in lexg) for g, p in zip(va["y"], pred)])
        res = {"R1": float((pred == va["y"]).mean() * 100), "n_err_scored": int(err.sum())}
        agree, lift = {}, {}
        for p in PARAMS:
            m = [(lexg[g][p] == lexg[q][p]) for g, q in zip(va["y"][err], pred[err]) if lexg[g][p] != "" and lexg[q][p] != ""]
            agree[p] = float(np.mean(m))
            lift[p] = float(np.mean(m) / base[p])
        res["error_agreement"] = agree
        res["lift_over_random_pair"] = lift
        h = np.array([sum(lexg[g][p] != lexg[q][p] for p in core if lexg[g][p] != "" and lexg[q][p] != "") for g, q in zip(va["y"][err], pred[err])])
        res["hamming_mean_errors"] = float(h.mean())
        res["hamming_le1_errors"] = float((h <= 1).mean())
        # accuracy by parameter value of the true gloss
        by = {}
        for p in ("SignType", "Movement", "Repeated", "MajorLoc", "Contact", "SelFingers", "WristTwist", "Flexion"):
            vals = np.array([lexg[g][p] if g in lexg else "" for g in va["y"]])
            by[p] = {v: dict(n=int((vals == v).sum()), R1=float((pred[vals == v] == va["y"][vals == v]).mean() * 100))
                     for v in np.unique(vals) if v != "" and (vals == v).sum() >= 100}
        res["R1_by_true_param_value"] = by
        out[name] = res
        print(name, res["R1"], "hamming", res["hamming_mean_errors"], "vs base", out["base_hamming_mean"], flush=True)
        print({p: round(v, 2) for p, v in lift.items()}, flush=True)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
