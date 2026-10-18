"""D-LEXEMB: what does the trained encoder expose about each ASL-LEX 2.0 phonological parameter, compared with its own input and with a
random encoder?  (proposal17 diagnostic G20; same protocol as proposal16 D-LEX v3)

Protocol (identical to proposal16/diagnostics/asllex_probe_v3.py): glosses are partitioned 70/30 with seed 0; probes are fit on gallery-side
(train-half) clips of the 70 % glosses and scored on query-side (val-half) clips of the 30 % glosses, so a probe can only succeed by encoding
the phonological parameter (gloss-disjoint) for unseen signers (signer-disjoint).  StandardScaler -> PCA(256) -> multinomial logistic
regression (C=0.05, class_weight balanced); metric = balanced accuracy; 95 % CI = bootstrap over held-out glosses.
Descriptors compared: PLctx teacher grp_fin / hands-only / CONF-only, RANDOM-INIT encoder, INPUT-token mean (no encoder).
The v3 raw-descriptor numbers (static mean+std, bins8, ...) are in proposal16/diagnostics/results/probe_v3_results.json.
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from analyze_emb import meta
from common import *  # noqa

SPLITS = "/home/dongvk/datasets/ASL_Citizen/splits"
LEX = "/home/haipd/Sign_Representations/docs/proposal16/data/signdata.csv"
EMB = f"{SCRATCH}/emb"
PARAMS = {
    "SignType": ("SignType.2.0", ["OneHanded", "SymmetricalOrAlternating", "AsymmetricalDifferentHandshape", "AsymmetricalSameHandshape"]),
    "Movement": ("Movement.2.0", ["Straight", "Curved", "Circular", "Z-shaped", "BackAndForth"]),
    "Repeated": ("RepeatedMovement.2.0", ["0", "1"]),
    "MajorLoc": ("MajorLocation.2.0", ["Neutral", "Head", "Hand", "Body", "Arm"]),
    "Contact": ("Contact.2.0", ["0", "1"]),
    "FlexChange": ("FlexionChange.2.0", ["0", "1"]),
    "WristTwist": ("UlnarRotation.2.0", ["0", "1"]),
    "SelFingers": ("SelectedFingers.2.0", ["imrp", "i", "im", "t", "m", "p"]),
    "Flexion": ("Flexion.2.0", ["FullyOpen", "Curved", "Flat", "Bent", "FullyClosed"]),
}


def main():
    man = manifests()
    lex = {r["Code"]: r for r in csv.DictReader(open(LEX, encoding="latin-1"))}
    cmap = {}
    for sp in ("train", "val"):
        for r in csv.DictReader(open(f"{SPLITS}/{sp}.csv")):
            cmap[r["Video file"].rsplit(".", 1)[0]] = r["ASL-LEX Code"]
    glosses = np.unique([m["gloss"] for sp in man.values() for m in sp])
    rng = np.random.default_rng(0)
    test_g = set(rng.choice(glosses, size=int(0.3 * len(glosses)), replace=False).tolist())
    zt, zv = np.load(f"{EMB}/train_teacher.npz"), np.load(f"{EMB}/val_teacher.npz")
    mt, mv = meta(zt["ids"], man["train"]), meta(zv["ids"], man["val"])
    gt, gv = np.array([m["gloss"] for m in mt]), np.array([m["gloss"] for m in mv])
    keep_t = np.array([g not in test_g for g in gt])
    keep_v = np.array([g in test_g for g in gv])
    print("test glosses", len(test_g), "train clips", int(keep_t.sum()), "val clips", int(keep_v.sum()), flush=True)

    def labels(metas, col, classes):
        out = []
        for m in metas:
            row = lex.get(cmap.get(m["sample_id"], "NA"))
            v = None if row is None else (row.get(col) or "").strip()
            if v and v.replace(".", "", 1).isdigit():
                v = str(int(float(v)))
            out.append(classes.index(v) if v in classes else -1)
        return np.array(out)

    sets = {}
    ch_t, ch_v = zt["ch_fin"].astype(np.float32).reshape(len(gt), 7, 256), zv["ch_fin"].astype(np.float32).reshape(len(gv), 7, 256)
    sets["PLctx grp_fin"] = (zt["grp_fin"].astype(np.float32), zv["grp_fin"].astype(np.float32))
    sets["PLctx hands-only"] = (ch_t[:, :4].reshape(len(gt), -1), ch_v[:, :4].reshape(len(gv), -1))
    sets["PLctx CONF-only"] = (ch_t[:, [0, 2]].reshape(len(gt), -1), ch_v[:, [0, 2]].reshape(len(gv), -1))
    sets["PLctx ord4"] = (zt["ord4"].astype(np.float32), zv["ord4"].astype(np.float32))
    zr, zrv = np.load(f"{EMB}/train_random.npz"), np.load(f"{EMB}/val_random.npz")
    sets["RANDOM-INIT grp_fin"] = (zr["grp_fin"].astype(np.float32), zrv["grp_fin"].astype(np.float32))
    ii, iv = np.load(f"{EMB}/train_inputs.npz"), np.load(f"{EMB}/val_inputs.npz")
    sets["INPUT tokens mean"] = (ii["inp"].astype(np.float32), iv["inp"].astype(np.float32))
    res = {"n_test_glosses": len(test_g)}
    brng = np.random.default_rng(1)
    for name, (G_, V_) in sets.items():
        G_, V_ = G_[keep_t], V_[keep_v]
        sc = StandardScaler().fit(G_)
        Gs, Vs = np.nan_to_num(sc.transform(G_)), np.nan_to_num(sc.transform(V_))
        pca = PCA(min(256, Gs.shape[1]), random_state=0).fit(Gs)
        Gp, Vp = pca.transform(Gs), pca.transform(Vs)
        res[name] = {"dim": int(G_.shape[1])}
        for pname, (col, classes) in PARAMS.items():
            yg = labels([m for m, k in zip(mt, keep_t) if k], col, classes)
            yv = labels([m for m, k in zip(mv, keep_v) if k], col, classes)
            okg, okv = yg >= 0, yv >= 0
            clf = LogisticRegression(max_iter=3000, C=0.05, class_weight="balanced").fit(Gp[okg], yg[okg])
            pred = clf.predict(Vp[okv])
            yt, gg = yv[okv], gv[keep_v][okv]
            ba = 100 * balanced_accuracy_score(yt, pred)
            ug = np.unique(gg)
            idx = {u: np.where(gg == u)[0] for u in ug}
            bs = []
            for _ in range(300):
                pick = np.concatenate([idx[u] for u in brng.choice(ug, size=len(ug), replace=True)])
                if len(np.unique(yt[pick])) < 2:
                    continue
                bs.append(100 * balanced_accuracy_score(yt[pick], pred[pick]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            res[name][pname] = [round(ba, 1), round(lo, 1), round(hi, 1)]
        print(name, {k: v for k, v in res[name].items()}, flush=True)
        json.dump(res, open("results/lex_probe_emb.json", "w"), indent=1)


if __name__ == "__main__":
    main()
