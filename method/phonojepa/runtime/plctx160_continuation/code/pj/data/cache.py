"""Keypoint loading (COCO-WholeBody pickles -> 103-joint pixel arrays) and the training-corpus cache.

Per clip: xy [F, 103, 2] float32 source pixels (NaN = not observed), conf [F, 103] float32 in [0, 1], fps, W, H.
Observation rule (thresholds of obs_jepa B): joint score >= 0.5 x part threshold, plus a part-mean gate on joint sets
that no mask splits (palm for hands, each face channel separately; none for body joints) — see gate().
conf = clip(score / (2 x part threshold), 0, 1) — uncalibrated, not a probability.
Labels (gloss / signer) are never stored in the training cache.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from pj.joints import COCO_INDEX, J, PARTS, THR_OF_PART
from pj.paths import ASLC002_CORPUS, ASLC_KP, OBS_META, RUNTIME


def load_clip(sample_id: str, width: int, height: int, kp_dir: Path = ASLC_KP):
    d = pickle.load(open(Path(kp_dir) / f"{sample_id}.pkl", "rb"))
    F = len(d["keypoints"])
    xy = np.full((F, J, 2), np.nan, np.float32)
    sc = np.zeros((F, J), np.float32)
    for t, (k, s) in enumerate(zip(d["keypoints"], d["scores"])):
        k, s = np.asarray(k), np.asarray(s)
        if k.ndim == 3 and k.shape[0] == 1:                  # single detected person only
            xy[t] = k[0, COCO_INDEX] * np.array([width, height], np.float32)
            sc[t] = s[0, COCO_INDEX]
    obs, conf = gate(xy, sc)
    xy[~obs] = np.nan
    conf[~obs] = 0.0
    return xy, conf


def gate(xy, sc):
    """-> observed [F,J] bool, conf [F,J]."""
    F = len(sc)
    obs = np.isfinite(xy).all(-1)
    conf = np.zeros((F, J), np.float32)
    # Observation gates are evaluated on joint sets that are never split by a mask (code-review finding #7: a part mean
    # over joints of several channels would let the scores of hidden joints decide whether visible joints are observed).
    #   body joints : joint threshold only (0.5 x 5.876). Deviation from obs_jepa B, whose body part gate rejected 36% of
    #                 frames although the wrist passed its joint threshold in 89% of hand-valid frames.
    #   hands       : part gate on the palm joints (wrist, CMC, MCPs; never hidden by R1) + joint threshold
    #   face        : part gate per face channel (F_UP, F_MO, F_HD joint sets) + joint threshold
    from pj.joints import F_HD, F_MO, F_UP, LH0, OWNED, RH0
    gates = [("body", PARTS["body"], None)]
    for base in (LH0, RH0):
        gates.append(("hand", list(range(base, base + 21)), [base + k for k in (0, 1, 5, 9, 13, 17)]))
    for ch in (F_UP, F_MO, F_HD):
        gates.append(("face", OWNED[ch], OWNED[ch]))
    for part, idx, ref in gates:
        thr = THR_OF_PART[part if part != "hand" else "lhand"]
        ok = sc[:, idx] >= 0.5 * thr
        if ref is not None:
            ok &= (sc[:, ref].mean(1) >= thr)[:, None]
        obs[:, idx] &= ok
        conf[:, idx] = np.clip(sc[:, idx] / (2.0 * thr), 0.0, 1.0)
    return obs, conf


def read_corpus(path=ASLC002_CORPUS):
    return [{"sample_id": Path(json.loads(l)["video"]).stem, "video": json.loads(l)["video"]} for l in open(path)]


def build_train_cache(out=RUNTIME / "cache" / "aslc002_train.npz", corpus=ASLC002_CORPUS, meta_path=OBS_META):
    """concatenated arrays for the 3,000-video ASLC002 corpus (proposal11 / EX16 protocol)."""
    rows = read_corpus(corpus)
    meta = json.load(open(meta_path))
    xs, cs, offs, fps, wh, ids = [], [], [0], [], [], []
    for r in rows:
        W, H, f, n = meta[r["sample_id"]]
        xy, conf = load_clip(r["sample_id"], W, H)
        xs.append(xy); cs.append(conf); offs.append(offs[-1] + len(xy)); fps.append(f); wh.append((W, H)); ids.append(r["sample_id"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, xy=np.concatenate(xs), conf=np.concatenate(cs).astype(np.float16), offsets=np.array(offs),
             fps=np.array(fps, np.float64), wh=np.array(wh), ids=np.array(ids))
    return out


class ClipCache:
    """read-only view on the training cache; clip(i) -> (xy, conf, fps)."""

    def __init__(self, path=RUNTIME / "cache" / "aslc002_train.npz"):
        z = np.load(path, allow_pickle=False)
        self.xy, self.conf, self.offs = z["xy"], z["conf"], z["offsets"]
        self.fps, self.wh, self.ids = z["fps"], z["wh"], z["ids"]

    def __len__(self):
        return len(self.offs) - 1

    def clip(self, i):
        a, b = self.offs[i], self.offs[i + 1]
        return self.xy[a:b], self.conf[a:b].astype(np.float32), float(self.fps[i])


if __name__ == "__main__":
    print(build_train_cache())
