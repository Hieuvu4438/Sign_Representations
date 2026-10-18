"""D-STYLE: which interpretable style/morphology variables carry the signer cluster seen in the PLctx embedding?
(proposal17 diagnostic G3). Pure metadata + pose statistics, no training.

For each clip (ASL Citizen train half + val, canonical body-frame arrays from proposal16/prep_pose.py) we compute ~30 interpretable
per-clip variables in three groups:
  MORPH   body/face/hand proportions that do not depend on what is signed (hand size / shoulder width, upper-arm/forearm ratio,
          eye distance, nose-chin distance, shoulder tilt ...) - medians over observed frames
  BEHAV   how the signer signs: signing-space amplitude, wrist speed, rest height, active fraction, duration, hand dominance
  CAM     camera framing: nose height, shoulder-width in px proxy via resolution, head offset
and ask (a) how well each group alone decodes the signer id (35-way, held-out clips), (b) how much of the PLctx embedding's
SIGNER component (gloss-residualised signer means) is linearly predicted from each group (R^2 on held-out clips).
Positive result = signer identity in the skeleton is explained by a small set of nameable nuisance variables, which tells us
which perturbations a signer-invariance objective should simulate.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from common import *  # noqa

S = SCRATCH
META = "/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc/metadata_{}.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/style_vars.json"
# canonical layout: 0 nose,1 Dsho,2 Nsho,3 Delb,4 Nelb,5 Dwri,6 Nwri, 7..27 Dhand, 28..48 Nhand, 49..65 face(17)
# face list indices (FACE order): brows 17,19,21,22,24,26 -> 49..54 ; nose tip 30 ->55 ; eyes 36,39,42,45 -> 56..59 ; mouth 48,51,54,57,62,66 -> 60..65
NOSE, DS, NS, DE, NE, DW, NW = 0, 1, 2, 3, 4, 5, 6
DH0 = 7


def med(a):
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else np.nan


def clip_vars(x, ob, frac, ts, right_dom, res_w, fps):
    x = x.astype(np.float32)
    T = len(x)
    d = lambda i, j: np.linalg.norm(x[:, i] - x[:, j], axis=-1)
    v = {}
    # --- MORPH (proportions in shoulder-width units, medians over frames)
    v["upper_arm_D"] = med(d(DS, DE))
    v["forearm_D"] = med(d(DE, DW))
    v["arm_ratio_D"] = v["forearm_D"] / max(v["upper_arm_D"], 1e-3) if np.isfinite(v["forearm_D"]) else np.nan
    v["palm_D"] = med(d(DH0, DH0 + 9))  # wrist -> middle MCP (index 9 of hand layout)
    v["handspan_D"] = med(d(DH0 + 4, DH0 + 20))  # thumb tip -> pinky tip (spread open or not, noisy)
    v["middle_len_D"] = med(d(DH0 + 9, DH0 + 12))
    v["eye_dist"] = med(d(56, 59))
    v["nose_chin_proxy"] = med(d(55, 63))
    v["mouth_width"] = med(d(60, 62)) if False else med(d(60, 63))
    v["brow_width"] = med(d(49, 54))
    v["shoulder_tilt"] = med(x[:, DS, 1] - x[:, NS, 1])
    v["nose_y"] = med(x[:, NOSE, 1])  # head height above shoulder line, in shoulder widths (also lean/camera)
    v["nose_x"] = med(x[:, NOSE, 0])  # head offset (head turn / lean)
    # --- BEHAV
    w = x[:, DW]
    vel = np.linalg.norm(np.diff(w, axis=0), axis=-1) * fps if T > 1 else np.zeros(1)
    vel = vel[np.isfinite(vel)]
    rest_y = np.nanpercentile(w[:, 1], 10) if np.isfinite(w[:, 1]).any() else np.nan
    v["rest_height_D"] = float(rest_y)
    act = (w[:, 1] > rest_y + 0.25) if np.isfinite(rest_y) else np.zeros(T, bool)
    v["active_frac"] = float(act.mean())
    v["duration_s"] = float(T / max(fps, 1.0))
    v["wrist_speed_med"] = float(np.median(vel)) if len(vel) else np.nan
    v["wrist_speed_p90"] = float(np.percentile(vel, 90)) if len(vel) else np.nan
    wa = w[act] if act.sum() > 3 else w
    v["amp_x_std"] = float(np.nanstd(wa[:, 0]))
    v["amp_y_std"] = float(np.nanstd(wa[:, 1]))
    v["amp_range_x"] = float(np.nanpercentile(wa[:, 0], 95) - np.nanpercentile(wa[:, 0], 5))
    v["amp_range_y"] = float(np.nanpercentile(wa[:, 1], 95) - np.nanpercentile(wa[:, 1], 5))
    v["wrist_mean_x"] = float(np.nanmean(wa[:, 0]))
    v["wrist_mean_y"] = float(np.nanmean(wa[:, 1]))
    v["right_dom"] = float(right_dom)
    # --- CAM
    v["res_w"] = float(res_w)
    v["fps"] = float(fps)
    v["obs_frac_Dhand"] = float(frac[DH0:DH0 + 21].mean())
    v["obs_frac_Nhand"] = float(frac[DH0 + 21:DH0 + 42].mean())
    v["obs_frac_face"] = float(frac[49:66].mean())
    return v


GROUPS = {
    "MORPH": ["upper_arm_D", "forearm_D", "arm_ratio_D", "palm_D", "handspan_D", "middle_len_D", "eye_dist", "nose_chin_proxy",
              "mouth_width", "brow_width", "shoulder_tilt"],
    "POSTURE": ["nose_y", "nose_x", "rest_height_D"],
    "BEHAV": ["active_frac", "duration_s", "wrist_speed_med", "wrist_speed_p90", "amp_x_std", "amp_y_std", "amp_range_x",
              "amp_range_y", "wrist_mean_x", "wrist_mean_y", "right_dom"],
    "CAM": ["res_w", "fps", "obs_frac_Dhand", "obs_frac_Nhand", "obs_frac_face"],
}


def build(split, every):
    z = np.load(f"{S}/{'val_pose' if split == 'val' else 'train_pose_half'}.npz", allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    stem = {}
    for l in open(META.format(split)):
        r = json.loads(l)
        stem[r["sample_id"]] = (r["source_stem"], r["width"])
    off = np.concatenate([[0], np.cumsum(z["lens"])])
    X_, OB_, TS_, FR_, RD_ = z["x"], z["ob"], z["ts"], z["frac"], z["right_dom"]
    rows, ids = [], []
    for i, m in enumerate(meta):
        a, b = off[i], off[i + 1]
        st, wpx = stem[m["sample_id"]]
        try:
            rows.append(clip_vars(X_[a:b], OB_[a:b], FR_[i], TS_[a:b], bool(RD_[i]), wpx, m["fps"]))
            ids.append(st)
        except Exception as e:  # pragma: no cover
            continue
    return ids, rows


def main():
    d = load_baseline()
    names = [n for g in GROUPS.values() for n in g]
    out = {}
    tab = {}
    for split in ("train", "val"):
        ids, rows = build(split, 1)
        tab[split] = (ids, np.array([[r.get(n, np.nan) for n in names] for r in rows], dtype=np.float64))
        print(split, len(ids), flush=True)
    # align with embeddings
    pos = {i: k for k, i in enumerate(d["train"]["ids"])}
    ti = [pos[i] for i in tab["train"][0] if i in pos]
    sel = np.array([k for k, i in enumerate(tab["train"][0]) if i in pos])
    F = tab["train"][1][sel]
    E = d["train"]["X"][ti]
    sg = d["train"]["signer"][ti]
    y = d["train"]["y"][ti]
    ok = np.isfinite(F).all(1)
    F, E, sg, y = F[ok], E[ok], sg[ok], y[ok]
    out["n_clips"] = int(len(F))
    Z = StandardScaler().fit_transform(F)

    # --- (a) signer identity from style variables only (GroupKFold over clips: same signers in train/test folds,
    # held-out CLIPS; also report gloss-balanced chance)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Z))
    cut = int(0.7 * len(Z))
    tr_i, te_i = perm[:cut], perm[cut:]
    res_a = {}
    for gname, cols in list(GROUPS.items()) + [("ALL", names)]:
        idx = [names.index(c) for c in cols]
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Z[tr_i][:, idx], sg[tr_i])
        res_a[gname] = float((clf.predict(Z[te_i][:, idx]) == sg[te_i]).mean() * 100)
    # leave-one-group-out
    for gname in GROUPS:
        idx = [names.index(c) for c in names if c not in GROUPS[gname]]
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Z[tr_i][:, idx], sg[tr_i])
        res_a["ALL minus " + gname] = float((clf.predict(Z[te_i][:, idx]) == sg[te_i]).mean() * 100)
    out["signer_id_acc_from_style_vars"] = res_a
    out["signer_id_chance"] = float(np.max(np.bincount(np.unique(sg, return_inverse=True)[1])) / len(sg) * 100)
    # also: same decode with the PLctx embedding itself (for scale)
    En = l2(E)
    clf = LogisticRegression(max_iter=300, C=1.0).fit(En[tr_i], sg[tr_i])
    out["signer_id_acc_from_embedding"] = float((clf.predict(En[te_i]) == sg[te_i]).mean() * 100)
    # R1-F7: the embedding decode above is on unscaled L2-normalised features; the style variables are standardised.  Same standardisation:
    Es = StandardScaler().fit_transform(En)
    clf = LogisticRegression(max_iter=300, C=1.0).fit(Es[tr_i], sg[tr_i])
    out["signer_id_acc_from_embedding_standardised"] = float((clf.predict(Es[te_i]) == sg[te_i]).mean() * 100)

    # --- (b) how much of the embedding's signer component do style vars predict?
    # gloss-residualise the embedding, take signer means, build the per-clip 'signer component' target (the signer mean of the residual)
    gm = np.zeros((y.max() + 1, En.shape[1]))
    cnt = np.bincount(y, minlength=y.max() + 1)
    np.add.at(gm, y, En)
    gm = gm / np.maximum(cnt, 1)[:, None]
    R = En - gm[y]
    sm = {s: R[sg == s].mean(0) for s in np.unique(sg)}
    Tsig = np.stack([sm[s] for s in sg])  # signer-offset component per clip
    # PCA of the signer-offset component to 34 dims (it has rank <= 34)
    Tc = Tsig - Tsig.mean(0)
    U, s_, Vt = np.linalg.svd(Tc[::10], full_matrices=False)
    P = Vt[:34].T
    tgt = Tc @ P
    # signer-grouped CV: held-out SIGNERS (offset of an unseen signer predicted from its style vars) and held-out clips
    res_b = {}
    ug = np.unique(sg)
    for gname, cols in list(GROUPS.items()) + [("ALL", names)]:
        idx = [names.index(c) for c in cols]
        # (i) held-out clips (signers seen)
        m = Ridge(alpha=10.0).fit(Z[tr_i][:, idx], tgt[tr_i])
        pr = m.predict(Z[te_i][:, idx])
        r2c = 1 - ((tgt[te_i] - pr) ** 2).sum() / ((tgt[te_i] - tgt[te_i].mean(0)) ** 2).sum()
        # (ii) held-out signers: 5-fold GroupKFold over signers
        gkf = GroupKFold(n_splits=5)
        num = den = 0.0
        for a, b in gkf.split(Z, groups=sg):
            m = Ridge(alpha=10.0).fit(Z[a][:, idx], tgt[a])
            pr = m.predict(Z[b][:, idx])
            num += ((tgt[b] - pr) ** 2).sum()
            den += ((tgt[b] - tgt[a].mean(0)) ** 2).sum()
        res_b[gname] = dict(R2_heldout_clips=float(r2c * 100), R2_heldout_signers=float((1 - num / den) * 100))
    out["signer_component_R2_from_style_vars"] = res_b

    # --- (c) variance of each variable between vs within signers (is it a signer trait?)
    icc = {}
    for j, n in enumerate(names):
        col = F[:, j]
        tot = col.var()
        if tot == 0:
            continue
        between = sum(((col[sg == s].mean() - col.mean()) ** 2) * (sg == s).sum() for s in ug) / len(col)
        icc[n] = float(between / tot)
    out["between_signer_variance_share_per_variable"] = dict(sorted(icc.items(), key=lambda kv: -kv[1]))
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
