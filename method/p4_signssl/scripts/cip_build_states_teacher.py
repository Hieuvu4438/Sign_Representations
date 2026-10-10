"""EX09: build proposal5-style states (D=224: appearance PCA-128 | geometry <=96) from
`runtime/cip_teacher_how2sign/*.npz`. Output `runtime/cip_teacher_states/` in the same cache
format as cip_build_states.py (states/valid/offsets/videos.jsonl/states_meta.json)."""
import json
from pathlib import Path

import numpy as np

RT = Path(__file__).resolve().parents[1] / "runtime"
SRC, OUT = RT / "cip_teacher_how2sign", RT / "cip_teacher_states"
D, A = 224, 128
POSE_BODY = [0, 7, 8, 11, 12, 23, 24]


def geometry(hand_lm, pose_lm):
    """-> geo [T,4,96] float32, gvalid [T,4,96] bool (FACE geometry absent)."""
    T = len(pose_lm)
    geo = np.zeros((T, 4, 96), np.float32)
    gv = np.zeros((T, 4, 96), bool)
    sh = (pose_lm[:, 11, :2] + pose_lm[:, 12, :2]) / 2
    sT = np.linalg.norm(pose_lm[:, 11, :2] - pose_lm[:, 12, :2], axis=1)
    pose_ok = np.isfinite(sT) & (sT > 1e-3)
    for p in (0, 1):
        h = hand_lm[:, p]  # T,21,3 (normalised x,y; relative z)
        sH = np.linalg.norm(h[:, 9, :2] - h[:, 0, :2], axis=1)
        ok = np.isfinite(h).all((1, 2)) & np.isfinite(sH) & (sH > 1e-4)
        rel = (h[:, 1:] - h[:, :1]) / np.where(ok, sH, 1)[:, None, None]
        geo[ok, p, :60] = rel[ok].reshape(-1, 60)
        gv[ok, p, :60] = True
        ok2 = ok & pose_ok
        geo[ok2, p, 60:62] = (h[ok2, 0, :2] - sh[ok2]) / sT[ok2, None]
        geo[ok2, p, 62] = np.log(sH[ok2] / sT[ok2])
        gv[ok2, p, 60:63] = True
    b = (pose_lm[:, POSE_BODY, :2] - sh[:, None]) / np.where(pose_ok, sT, 1)[:, None, None]
    geo[pose_ok, 3, :14] = b[pose_ok].reshape(-1, 14)
    geo[pose_ok, 3, 14:16] = (pose_lm[pose_ok, 12, :2] - pose_lm[pose_ok, 11, :2]) / sT[pose_ok, None]
    geo[pose_ok, 3, 16] = np.log(sT[pose_ok])
    gv[pose_ok, 3, :17] = True
    return geo, gv


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in SRC.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    names = [p.stem for p in files]
    groups = sorted({n[:11] for n in names})
    rng = np.random.default_rng(2026)
    perm = list(rng.permutation(groups))
    ncal = max(2, len(perm) // 20)
    calib, rest = set(perm[:ncal]), perm[ncal:]
    fold_of = {g: i % 5 for i, g in enumerate(rest)}
    raw = []
    for p in files:
        z = np.load(p)
        app = z["app"].astype(np.float32)  # T,3,384 (LH,RH,FACE)
        obs = z["observed"]  # T,4: face, left, right, pose
        appv = np.stack([obs[:, 1], obs[:, 2], obs[:, 0]], 1)  # LH,RH,FACE observed
        geo, gv = geometry(z["hand_lm"], z["pose_lm"])
        raw.append((app, appv, geo, gv))
    is_cal = np.array([n[:11] in calib for n in names])
    # PCA-128 per appearance stream on calibration groups (observed frames only)
    pcas = []
    for s in range(3):
        X = np.concatenate([r[0][r[1][:, s], s] for r, c in zip(raw, is_cal) if c])
        X = X[rng.choice(len(X), min(200000, len(X)), replace=False)]
        mu = X.mean(0)
        _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
        pcas.append((mu, Vt[:A]))
    S_list, V_list = [], []
    for app, appv, geo, gv in raw:
        T = len(app)
        S = np.zeros((T, 4, D), np.float32)
        V = np.zeros((T, 4, D), bool)
        for s, part in enumerate((0, 1, 2)):  # LH, RH, FACE appearance
            mu, W = pcas[s]
            S[:, part, :A] = (app[:, s] - mu) @ W.T
            V[:, part, :A] = appv[:, s][:, None]
        S[:, :, A:] = geo
        V[:, :, A:] = gv
        S_list.append(S); V_list.append(V)
    lens = np.array([len(s) for s in S_list]); off = np.concatenate([[0], np.cumsum(lens)])
    S = np.concatenate(S_list); V = np.concatenate(V_list)
    cal_rows = np.repeat(is_cal, lens)
    Sc, Vc = S[cal_rows], V[cal_rows]
    for p in range(4):
        for d in range(D):
            m = Vc[:, p, d]
            if m.sum() > 1000:
                x = Sc[:, p, d][m]; mu, sd = x.mean(), max(x.std(), 1e-3)
                S[:, p, d] = np.where(V[:, p, d], (S[:, p, d] - mu) / sd, 0)
            else:
                V[:, p, d] = False; S[:, p, d] = 0
    np.save(OUT / "states.npy", S.astype(np.float16)); np.save(OUT / "valid.npy", V); np.save(OUT / "offsets.npy", off)
    with open(OUT / "videos.jsonl", "w") as f:
        for n, c in zip(names, is_cal):
            f.write(json.dumps({"sample_id": n, "signer": n[:11], "partition": "calibration" if c else "source_train",
                                "fold": -1 if c else fold_of[n[:11]]}) + "\n")
    cov = V.reshape(len(V), 4, -1)
    json.dump({"n_clips": len(names), "n_frames": int(len(S)), "D": D, "blocks": [[0, A], [A, D]],
               "appearance_coverage": cov[:, :, :A].any(-1).mean(0).tolist(), "geometry_coverage": cov[:, :, A:].any(-1).mean(0).tolist()},
              open(OUT / "states_meta.json", "w"), indent=1)
    print("clips", len(names), "frames", len(S), "app cov", cov[:, :, :A].any(-1).mean(0), "geo cov", cov[:, :, A:].any(-1).mean(0))


if __name__ == "__main__":
    main()
