"""Multi-variant embedding extraction for proposal17 diagnostics G5-G7 (CPU; no training, no test split).

For ASL Citizen train (every 2nd clip) and val (every 2nd clip) we run three frozen encoders on the SAME preprocessed windows:
  teacher  = EMA teacher of the PLctx full160 checkpoint (the protocol's weights)
  student  = online student of the same checkpoint
  random   = identical architecture, random init (seed 0)  -> floor / "retention ratio" control
and store, per clip (mean over windows, stride 16 samples, T = 64 samples @ min(fps, 25) as the protocol),
  grp_L1..L6 : group-mean token readout (D hand, N hand, face; 768-d) after every encoder block (pre final LayerNorm)
  grp_fin    : same after the final LayerNorm (= the protocol's `_grp`, but stride 16 instead of 2)
  ord2/ord4  : group readout per temporal segment of the 32 bins (order-preserving), last layer
  ord4_L4    : same on block 4
  ch_fin     : per-channel mean (7 x 256)
  cls_fin    : CLS token
An optional morphology-perturbed pass (clip-constant analytic 2D perturbation of face scale, arm length, hand scale, applied to the
raw keypoints BEFORE canonicalisation) on a val subset measures how much the teacher embedding / retrieval depends on morphology.
Outputs: <scratch>/emb/<split>_<variant>.npz (+ ids). Labels are never used here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/haipd/Sign_Representations/method/phonojepa")
sys.path.insert(0, "/home/haipd/Sign_Representations/method/obs_jepa")
from pj import features as Fe  # noqa: E402
from pj.data import window as W  # noqa: E402
from pj.data.cache import load_clip  # noqa: E402
from pj.joints import F0, LELB, LH0, LSHO, LWRI, NC, NF, NOSE_B, NOSE_TIP, RELB, RH0, RSHO, RWRI  # noqa: E402
from pj.model import Encoder  # noqa: E402
from oj.data.clips import video_meta  # noqa: E402

from common import SCRATCH, manifests  # noqa: E402

CKPT = "/home/haipd/Sign_Representations/method/phonojepa/runtime/runs/PJ_PLctx_FULL_160ep/final.pt"
KP = Path("/home/haipd/Sign_Representations/../../home/dongvk/datasets/ASL_Citizen/keypoints").resolve()
KP = Path("/home/dongvk/datasets/ASL_Citizen/keypoints")
T = 64
STRIDE = 16
TB = 32
GROUPS = ((0, 1), (2, 3), (4, 5, 6))
FACE = list(range(F0, F0 + NF))
LHAND, RHAND = list(range(LH0, LH0 + 21)), list(range(RH0, RH0 + 21))


def morph(xy, rng, mode, scale=1.15):
    """clip-constant analytic 2D morphology perturbation of raw pixel keypoints xy [F,J,2] (NaN = unobserved).
    mode in {'face','arm','hand','all','nonuni'}; factors log-uniform in [1/scale, scale] (scale 1.15 = +-15 %).
    'nonuni' = independent scale per arm, per hand and per finger (about the finger base joint), no face change.
    NaN-SAFE (review R2-N1): an unobserved body wrist/elbow never turns observed hand keypoints into NaN; the hand is translated by the
    finite displacement of the wrist (else of the elbow, else 0) and scaled about its own root joint; observation flags are unchanged."""
    orig = xy
    xy = xy.copy()
    s = lambda: float(np.exp(rng.uniform(np.log(1.0 / scale), np.log(scale))))
    if mode in ("face", "all"):
        sf = s()
        c = orig[:, NOSE_TIP : NOSE_TIP + 1]
        fn = c + sf * (orig[:, FACE] - c)
        xy[:, FACE] = np.where(np.isfinite(fn) & np.isfinite(orig[:, FACE]), fn, orig[:, FACE])
    if mode in ("arm", "hand", "all", "nonuni"):
        for sho, elb, wri, hand in ((LSHO, LELB, LWRI, LHAND), (RSHO, RELB, RWRI, RHAND)):
            sa = s() if mode in ("arm", "all", "nonuni") else 1.0
            sh = s() if mode in ("hand", "all", "nonuni") else 1.0
            e_old, w_old = orig[:, elb], orig[:, wri]
            e_new = orig[:, sho] + sa * (e_old - orig[:, sho])
            w_new = e_new + sa * (w_old - e_old)
            xy[:, elb] = np.where(np.isfinite(e_new), e_new, e_old)
            xy[:, wri] = np.where(np.isfinite(w_new), w_new, w_old)
            dw, de = w_new - w_old, e_new - e_old
            delta = np.where(np.isfinite(dw), dw, np.where(np.isfinite(de), de, 0.0))
            h = orig[:, hand]
            root = h[:, :1]
            hn = root + delta[:, None] + sh * (h - root)
            if mode == "nonuni":
                for base in (1, 5, 9, 13, 17):  # per-finger length about the finger base joint
                    sf_ = s()
                    hn[:, base : base + 4] = hn[:, base : base + 1] + sf_ * (hn[:, base : base + 4] - hn[:, base : base + 1])
            xy[:, hand] = np.where(np.isfinite(hn) & np.isfinite(h), hn, h)
    return xy


def prepare(row, meta, rng=None, mode=None, scale=1.15):
    width, height, fps, nf = meta
    xy, conf = load_clip(row["sample_id"], width, height, kp_dir=KP)
    if mode is not None:
        xy = morph(xy, rng, mode, scale)
    xy, conf, _ = W.canonicalise(xy, conf)
    eff = min(fps, W.TARGET_FPS)
    nsamp = max(1, int(np.floor(nf / fps * eff)))
    starts = list(range(0, max(1, nsamp - T + 1), STRIDE))
    arrays = {k: [] for k in ("xy", "obs", "conf", "valid")}
    for start in starts:
        i0, i1, weights, valid = W.window_positions(nf, fps, T, start / eff, 1.0)
        x, c, _ = W.resample(xy, conf, i0, i1, weights, valid)
        arrays["xy"].append(np.nan_to_num(x))
        arrays["obs"].append(np.isfinite(x).all(-1))
        arrays["conf"].append(c)
        arrays["valid"].append(valid)
    return {k: np.stack(v) for k, v in arrays.items()}


class DS(torch.utils.data.Dataset):
    def __init__(self, rows, metas, mode=None, seed=0, scale=1.15):
        self.rows, self.metas, self.mode, self.seed, self.scale = rows, metas, mode, seed, scale

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1000003 + i)
        return prepare(self.rows[i], self.metas[i], rng, self.mode, self.scale)


def load_encoders(device="cpu"):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    c = ck["config"]
    encs = {}
    for name in ("teacher", "student"):
        e = Encoder(c["d"], c["depth"], c["heads"]).to(device).eval()
        e.load_state_dict(ck[name])
        encs[name] = e
    torch.manual_seed(0)
    encs["random"] = Encoder(c["d"], c["depth"], c["heads"]).to(device).eval()
    return encs, c


def group_mean(h, use):  # h [B,TB,NC,d], use [B,TB,NC] -> [B, 3*d]
    out = []
    for g in GROUPS:
        wg = use[:, :, list(g)]
        out.append((h[:, :, list(g)] * wg[..., None]).sum((1, 2)) / wg.sum((1, 2)).clamp(min=1)[:, None])
    return torch.cat(out, -1)


def ordered(h, use, K):
    seg = TB // K
    out = []
    for k in range(K):
        out.append(group_mean(h[:, k * seg : (k + 1) * seg], use[:, k * seg : (k + 1) * seg]))
    return torch.cat(out, -1)


@torch.inference_mode()
def encode_windows(encs, p, dev="cpu", only=None):
    xy = torch.as_tensor(p["xy"], device=dev)
    obs = torch.as_tensor(p["obs"], device=dev)
    conf = torch.as_tensor(p["conf"], device=dev)
    valid = torch.as_tensor(p["valid"], device=dev)
    f = Fe.frame_features(xy, obs, conf, torch.zeros_like(obs))
    ins, pres = Fe.tokens(f, valid)
    B = xy.shape[0]
    vb = valid.view(B, -1, 2).any(-1)
    cells = vb[:, None, :].expand(-1, NC, -1).clone()
    use = (cells & pres).transpose(1, 2).flatten(1).float().view(B, TB, NC)
    res = {}
    for name, enc in encs.items():
        if only and name not in only:
            continue
        hfin, cls, layers = enc(ins, pres, cells, return_layers=len(enc.blocks))
        r = {}
        for i, L in enumerate(layers):
            r[f"grp_L{i + 1}"] = group_mean(L.view(B, TB, NC, -1), use)
        hf = hfin.view(B, TB, NC, -1)
        r["grp_fin"] = group_mean(hf, use)
        r["ord2"] = ordered(hf, use, 2)
        r["ord4"] = ordered(hf, use, 4)
        r["ord4_L4"] = ordered(layers[3].view(B, TB, NC, -1), use, 4)
        wch = use[..., None]
        r["ch_fin"] = ((hf * wch).sum(1) / wch.sum(1).clamp(min=1)).flatten(1)
        r["cls_fin"] = cls
        res[name] = {k: v.float().cpu().numpy() for k, v in r.items()}
    return res


def run(rows, metas, tag, out_dir, workers, mode=None, only=None, limit=0, scale=1.15):
    encs, _ = load_encoders()
    if only:
        encs = {k: v for k, v in encs.items() if k in only}
    dl = torch.utils.data.DataLoader(DS(rows, metas, mode, 0, scale), batch_size=None, num_workers=workers,
                                     worker_init_fn=lambda _: (torch.set_num_threads(1), __import__("cv2").setNumThreads(1)),
                                     prefetch_factor=4 if workers else None)
    acc = {}
    t0 = time.time()
    n = 0
    for i, p in enumerate(dl):
        p = {k: v.numpy() if torch.is_tensor(v) else v for k, v in p.items()}
        res = encode_windows(encs, p, only=only)
        for name, r in res.items():
            for k, v in r.items():
                a = acc.setdefault(name, {}).setdefault(k, np.zeros((len(rows), v.shape[1]), np.float32))
                a[i] = v.mean(0)
        n += 1
        if i % 200 == 0:
            print(tag, i, len(rows), f"{time.time() - t0:.0f}s", flush=True)
        if limit and n >= limit:
            break
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, r in acc.items():
        np.savez(out_dir / f"{tag}_{name}.npz", **{k: v[:n].astype(np.float16) for k, v in r.items()}, ids=np.array([r_["sample_id"] for r_ in rows[:n]]))
    print("saved", tag, n, f"{time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--morph", default="", help="'' | face | arm | hand | all | nonuni (val subset only)")
    ap.add_argument("--morph_scale", type=float, default=1.15)
    ap.add_argument("--balanced", type=int, default=0, help="val only: N clips per signer (P13 excluded) from the every-k subset")
    ap.add_argument("--n_morph", type=int, default=1500)
    ap.add_argument("--only", default="", help="comma list of variants")
    ap.add_argument("--out", default=f"{SCRATCH}/emb")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rows = manifests()[a.split][:: a.every]
    if a.balanced:
        cnt, keep = {}, []
        for r in rows:
            if r["signer"] == "P13":
                continue
            if cnt.get(r["signer"], 0) < a.balanced:
                cnt[r["signer"]] = cnt.get(r["signer"], 0) + 1
                keep.append(r)
        rows = keep
    elif a.morph:
        rows = rows[: a.n_morph]
    metas = [video_meta(r["video"])[:4] for r in rows]
    only = [x for x in a.only.split(",") if x] or None
    tag = a.split + (f"_bal{a.balanced}" if a.balanced else "") + (f"_morph-{a.morph}-s{a.morph_scale}" if a.morph else "") + (f"_lim{a.limit}" if a.limit else "")
    run(rows, metas, tag, a.out, a.workers, mode=a.morph or None, only=only, limit=a.limit, scale=a.morph_scale)


if __name__ == "__main__":
    main()
