"""Build the geometry-only teacher-state cache for the CIP gate test on ASL Citizen TRAIN
videos (labels never read; Participant ID used only as source group, as proposal5 §5.3
allows "known signer" grouping).

Partitions (by signer, seed 2026): calibration = 2 signers (score thresholds + z-score
statistics only); the other 33 signers -> 5 cross-fitting folds.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4 import cip_states as C  # noqa: E402
from p4.data import aslc_rows  # noqa: E402

DATASET = sys.argv[1] if len(sys.argv) > 1 else "aslc"
OUT = Path(__file__).resolve().parents[1] / "runtime" / ("cip" if DATASET == "aslc" else "cip_how2sign")
H2S_KP = Path("/home/shared_data/sign_language/How2Sign/train/train_pose")
KPDIR = None if DATASET == "aslc" else H2S_KP
THR = None


def _init(thr):
    global THR
    THR = thr


def _one(sid):
    try:
        return C.video_states(sid, THR, KPDIR)
    except Exception as e:  # missing / corrupt pickle
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    if DATASET == "aslc":
        rows = aslc_rows("train")
    else:  # How2Sign train: random 8000 sentence clips; group = YouTube video_id (text never read)
        names = sorted(p.stem for p in H2S_KP.glob("*.pkl"))
        names = list(rng.choice(names, 8000, replace=False))
        rows = [{"sample_id": n, "signer": n[:11]} for n in names]  # YouTube video ids are 11 chars
    signers = sorted({r["signer"] for r in rows})
    perm = list(rng.permutation(signers))
    ncal = 2 if DATASET == "aslc" else max(2, len(perm) // 20)
    calib, rest = perm[:ncal], perm[ncal:]
    fold_of = {s: i % 5 for i, s in enumerate(rest)}
    cal_ids = [r["sample_id"] for r in rows if r["signer"] in calib]
    thr = C.score_quantiles(rng.choice(cal_ids, min(600, len(cal_ids)), replace=False), q=0.2, kp_dir=KPDIR)
    print("score thresholds (hand, face, body)", thr, flush=True)
    with ProcessPoolExecutor(2, initializer=_init, initargs=(thr,)) as ex:  # user rule: <= 2 processes
        res = list(ex.map(_one, [r["sample_id"] for r in rows], chunksize=64))
    keep = [i for i, x in enumerate(res) if x is not None]
    print("videos", len(rows), "with pose", len(keep), flush=True)
    S = np.concatenate([res[i][0] for i in keep]).astype(np.float32)
    V = np.concatenate([res[i][1] for i in keep])
    lens = np.array([len(res[i][0]) for i in keep])
    off = np.concatenate([[0], np.cumsum(lens)])
    is_cal = np.repeat([rows[i]["signer"] in calib for i in keep], lens)
    mean = np.zeros((4, C.D), np.float32)
    std = np.ones((4, C.D), np.float32)
    Sc, Vc = S[is_cal], V[is_cal]
    for p in range(4):
        for d in range(C.D):
            m = Vc[:, p, d]
            if m.sum() > 1000:
                x = Sc[:, p, d][m]
                mean[p, d], std[p, d] = x.mean(), max(x.std(), 1e-3)
            else:
                V[:, p, d] = False  # no calibration coverage -> globally invalid
    S = np.where(V, (S - mean) / std, 0.0).astype(np.float16)
    meta = [{"sample_id": rows[i]["sample_id"], "signer": rows[i]["signer"],
             "partition": "calibration" if rows[i]["signer"] in calib else "source_train",
             "fold": -1 if rows[i]["signer"] in calib else fold_of[rows[i]["signer"]]} for i in keep]
    np.save(OUT / "states.npy", S)
    np.save(OUT / "valid.npy", V)
    np.save(OUT / "offsets.npy", off)
    np.savez(OUT / "calibration.npz", mean=mean, std=std, thr=np.array(thr))
    with open(OUT / "videos.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    cov = V.reshape(len(V), 4, -1).any(-1).mean(0)
    json.dump({"calibration_signers": calib, "fold_of": fold_of, "thresholds": thr, "n_frames": int(len(S)),
               "frame_part_coverage": cov.tolist(), "part_dims": C.PART_DIMS},
              open(OUT / "states_meta.json", "w"), indent=1)
    print("coverage per part (LH,RH,FACE,BODY)", cov, flush=True)


if __name__ == "__main__":
    main()
