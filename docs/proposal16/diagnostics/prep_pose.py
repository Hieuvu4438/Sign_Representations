"""proposal16 training-free diagnostics, step 1: compact body-frame pose arrays for ASL Citizen clips.

Reads the asm_jepa per-clip cache (xy pixels [T,133,2], observed flags) read-only, applies the audit-driven
observation filters (in-image + hand-wrist distance 0.30 shoulder widths), body-frame normalisation
(origin = shoulder midpoint, scale = median shoulder width, pixel units so aspect ratio is correct),
dominant-hand canonicalisation (hand with longer body-wrist path is "D"; left-dominant clips are mirrored),
and linear interpolation of short gaps. Output: one npz with ragged arrays concatenated.

Usage: python prep_pose.py --split val --out <scratch>/val_pose.npz [--limit N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc")
LH = list(range(91, 112))
RH = list(range(112, 133))
# face subset (COCO-WholeBody face 23..90 = 68 iBUG points): brows 17-26, nose tip 30, mouth 48-67 corners/lips
FACE = [23 + i for i in (17, 19, 21, 22, 24, 26, 30, 36, 39, 42, 45, 48, 51, 54, 57, 62, 66)]
BODY = [0, 5, 6, 7, 8, 9, 10]  # nose, shoulders, elbows, wrists
# output joint layout (after canonicalisation): BODY_D order = nose, D_sho, N_sho, D_elb, N_elb, D_wri, N_wri
N_OUT = 7 + 21 + 21 + len(FACE)


def load(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return z["xy"].astype(np.float64), z["observed"].copy(), z["image_wh"], z["timestamps_sec"].copy()


def interp_gaps(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """x [T,J,2], ok [T,J] -> filled x (NaN where a joint is never observed)."""
    T = len(x)
    out = np.full_like(x, np.nan)
    t = np.arange(T)
    for j in range(x.shape[1]):
        m = ok[:, j]
        if m.sum() == 0:
            continue
        for c in range(2):
            out[:, j, c] = np.interp(t, t[m], x[m, j, c])
    return out


def process(path: Path):
    xy, ob, wh, ts = load(path)
    T = len(xy)
    whv = np.asarray(wh, np.float64)
    whv = np.broadcast_to(whv if whv.ndim == 2 else whv[None], (T, 2))[:, None, :]
    with np.errstate(invalid="ignore"):
        ob &= ((xy >= 0) & (xy <= whv)).all(-1)
    shok = ob[:, 5] & ob[:, 6]
    if shok.sum() < 3:
        return None
    sw = float(np.median(np.linalg.norm(xy[shok, 5] - xy[shok, 6], axis=-1)))
    if sw < 1e-3:
        return None
    for hand, bw in ((LH, 9), (RH, 10)):
        ref = ob[:, hand[0]] & ob[:, bw]
        with np.errstate(invalid="ignore"):
            far = ref & (np.linalg.norm(xy[:, hand[0]] - xy[:, bw], axis=-1) > 0.30 * sw)
        ob[np.ix_(far, hand)] = False
    centre = np.nanmedian(np.where(shok[:, None], 0.5 * (xy[:, 5] + xy[:, 6]), np.nan), axis=0)
    z = (xy - centre) / sw
    z[..., 1] *= -1.0  # y up
    # dominant hand = longer observed body-wrist path
    def plen(j):
        m = ob[:, j]
        p = z[m, j]
        return float(np.linalg.norm(np.diff(p, axis=0), axis=-1).sum()) if len(p) > 1 else 0.0
    right_dom = plen(10) >= plen(9)
    if right_dom:
        body = [0, 6, 5, 8, 7, 10, 9]
        dh, nh = RH, LH
        mirror = False
    else:
        body = [0, 5, 6, 7, 8, 9, 10]
        dh, nh = LH, RH
        mirror = True
    idx = body + dh + nh + FACE
    zz = z[:, idx].copy()
    oo = ob[:, idx].copy()
    if mirror:
        zz[..., 0] *= -1.0
    filled = interp_gaps(zz, oo)
    frac = oo.mean(0)
    return filled.astype(np.float16), oo, frac.astype(np.float32), ts, right_dom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="keep every k-th clip (deterministic subsample)")
    a = ap.parse_args()
    meta = [json.loads(l) for l in open(ROOT / f"metadata_{a.split}.jsonl")]
    meta = meta[:: a.every]
    if a.limit:
        meta = meta[: a.limit]
    xs, obs, fracs, lens, keep, rdom, tss = [], [], [], [], [], [], []
    for i, m in enumerate(meta):
        p = ROOT / m["pose_relpath"]
        if not p.exists():
            continue
        r = process(p)
        if r is None:
            continue
        x, o, f, ts, rd = r
        xs.append(x); obs.append(o); fracs.append(f); lens.append(len(x)); rdom.append(rd); tss.append(ts)
        keep.append({k: m[k] for k in ("sample_id", "signer_id", "gloss", "class_id", "fps", "duration_sec")})
        if i % 1000 == 0:
            print(i, len(meta), flush=True)
    np.savez(a.out, x=np.concatenate(xs), ob=np.concatenate(obs), ts=np.concatenate(tss),
             frac=np.stack(fracs), lens=np.array(lens), right_dom=np.array(rdom),
             meta=np.array(json.dumps(keep)))
    print("saved", a.out, len(keep), "clips")


if __name__ == "__main__":
    main()
