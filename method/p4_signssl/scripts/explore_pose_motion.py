"""EXPLORATORY (val LOSO): a hand-trajectory descriptor from the local COCO-WholeBody 2-D
keypoints, used as (a) a control representation for the ASL-LEX error-sharing lift
(does a motion-centric descriptor show high movement lift?), and (b) a complementarity
check: does adding trajectory information to SignRep scores help (fusion weight swept on
the same val queries -> exploratory upper bound, not a method).

Descriptor per clip: wrists + index/middle fingertips of both hands, centred on the
shoulder midpoint and scaled by shoulder width; frames with 1 detected person; linear
interpolation over missing frames; resampled to 16 steps over the span where either wrist
is above the hip line; features = positions (16x8x2) + velocities (15x8x2), z-scored with
val statistics (exploratory), L2-normalised.
"""
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.retrieval import activity_weights, class_max_scores, l2n, metrics_from_ranks, pool_clips, ranks_from_scores  # noqa
from p4.phonology import error_attribute_sharing, gloss_codes, load_asllex, minimal_pairs  # noqa

KP = Path("/home/dongvk/datasets/ASL_Citizen/keypoints")
# COCO-WholeBody: 5/6 shoulders, 11/12 hips, 9/10 wrists(body); hands 91-111 left, 112-132 right
JOINTS = [91, 91 + 8, 91 + 12, 112, 112 + 8, 112 + 12]  # wrist, index tip, middle tip (L, R)
STEPS = 16


def descriptor(sample_id):
    try:
        d = pickle.load(open(KP / f"{sample_id}.pkl", "rb"))
    except Exception:
        return None
    pts, ok = [], []
    for k in d["keypoints"]:
        k = np.asarray(k)
        if k.ndim == 3 and k.shape[0] == 1:
            pts.append(k[0]); ok.append(True)
        else:
            pts.append(np.zeros((133, 2))); ok.append(False)
    pts, ok = np.stack(pts), np.array(ok)
    if ok.sum() < 4:
        return None
    sh = (pts[:, 5] + pts[:, 6]) / 2
    sw = np.linalg.norm(pts[:, 5] - pts[:, 6], axis=1)
    scale = np.median(sw[ok]) + 1e-6
    hip_y = np.median(((pts[:, 11, 1] + pts[:, 12, 1]) / 2)[ok])
    X = (pts[:, JOINTS] - sh[:, None]) / scale  # T,6,2
    t = np.arange(len(X))
    for j in range(X.shape[1]):
        for c in range(2):
            X[:, j, c] = np.interp(t, t[ok], X[ok, j, c])
    wy = np.minimum(pts[:, 91, 1], pts[:, 112, 1])
    act = ok & (wy < hip_y)
    a, b = (np.where(act)[0][[0, -1]] if act.any() else (0, len(X) - 1))
    if b - a < 2:
        a, b = 0, len(X) - 1
    tt = np.linspace(a, b, STEPS)
    R = np.stack([np.stack([np.interp(tt, t, X[:, j, c]) for c in range(2)], -1) for j in range(X.shape[1])], 1)  # 16,6,2
    V = np.diff(R, axis=0)
    return np.concatenate([R.ravel(), 2.0 * V.ravel()]).astype(np.float32)


def main():
    d = Path("method/p4_signssl/runtime/feats/aslc_val_s2")
    clips = [json.loads(l) for l in open(d / "clips.jsonl")]
    with ProcessPoolExecutor(12) as ex:
        desc = list(ex.map(descriptor, [c["sample_id"] for c in clips], chunksize=64))
    miss = [i for i, x in enumerate(desc) if x is None]
    D = np.stack([x if x is not None else np.zeros_like(desc[0 if desc[0] is not None else 1]) for x in desc])
    D = (D - D.mean(0)) / (D.std(0) + 1e-6)
    D = l2n(D)
    off = np.load(d / "offsets.npy")
    S_feat, _ = pool_clips(np.load(d / "windows_features.npy"), off, activity_weights(np.load(d / "windows_active.npy")))
    S_feat = l2n(S_feat)
    gl_ = sorted({c["gloss"] for c in clips}); lab = {g: i for i, g in enumerate(gl_)}
    y = np.array([lab[c["gloss"]] for c in clips]); sg = np.array([c["signer"] for c in clips])
    _, sigs = minimal_pairs(gl_, gloss_codes(clips), load_asllex())

    def loso_scores(F):
        S = np.zeros((len(y), len(gl_)), np.float32)
        for s in np.unique(sg):
            q, g = sg == s, sg != s
            S[q] = class_max_scores(F[q], F[g], y[g], len(gl_))
        return S

    Sp, Sr = loso_scores(D), loso_scores(S_feat)
    out = {"n_missing_pose": len(miss)}
    for name, S in (("pose_motion", Sp), ("signrep", Sr)):
        out[name] = {"retrieval": metrics_from_ranks(ranks_from_scores(S, y)), "err": error_attribute_sharing(S, y, sigs)}
    for lam in (0.1, 0.2, 0.3, 0.5):
        out[f"fused_{lam}"] = {"retrieval": metrics_from_ranks(ranks_from_scores(Sr + lam * Sp, y))}
    print(json.dumps(out, indent=1))
    json.dump(out, open("method/p4_signssl/runtime/explore/val_pose_motion_loso.json", "w"), indent=1)


if __name__ == "__main__":
    main()
