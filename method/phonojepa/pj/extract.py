"""Frozen window features in the obs_jepa format, so every model is scored by the SAME evaluation code
(method/obs_jepa/oj/evaluate.py: LOSO-val whitened retrieval, ASLC300 retrieval + true linear probe, probes, signer).

Windows: T = 64 samples at min(fps, 25), one every 32 samples, speed 1, no augmentation (same as obs_jepa graph B).
Readouts (written as separate feature names, proposal12 §6.14):
  <name>_grp   concat of mean present-token features for dominant hand (D_CONF, D_LOC), non-dominant hand, face  [primary]
  <name>_mean  mean over present tokens
  <name>_cls   CLS token
kind = model (final.pt, --weights teacher|student) | descriptor (training-free B0: bin-pair means of the level features)
Output: runtime/feats/<name>[_readout]/<set>/{windows.npy, offsets.npy, src_span.npy, clips.jsonl, meta.json}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pj import features as Fe  # noqa: E402
from pj.data import window as W  # noqa: E402
from pj.data.cache import load_clip  # noqa: E402
from pj.joints import FACE_CH, NC  # noqa: E402
from pj.paths import OBS_JEPA, RUNTIME  # noqa: E402

sys.path.insert(0, str(OBS_JEPA))
from oj.data.clips import video_meta  # noqa: E402
from oj.extract import rows_for  # noqa: E402

T = 64
GROUPS = ((0, 1), (2, 3), (4, 5, 6))


def prepare(r):
    Wd, H, fps, nf = video_meta(r["video"])
    xy, conf = load_clip(r["sample_id"], Wd, H)
    n = min(nf, len(xy)) if nf > 0 else len(xy)
    xy, conf = xy[:n], conf[:n]
    xy, conf, _ = W.canonicalise(xy, conf)
    eff = min(fps, W.TARGET_FPS)
    nsamp = max(1, int(np.floor(n / fps * eff)))
    starts = list(range(0, max(1, nsamp - T + 1), 32))
    out = {k: [] for k in ("xy", "obs", "conf", "valid")}
    spans = []
    for s in starts:
        i0, i1, w, valid = W.window_positions(n, fps, T, s / eff, 1.0)
        x, c, _ = W.resample(xy, conf, i0, i1, w, valid)
        out["xy"].append(np.nan_to_num(x)); out["obs"].append(np.isfinite(x).all(-1)); out["conf"].append(c); out["valid"].append(valid)
        src = np.concatenate([i0[valid], i1[valid]]) if valid.any() else np.array([0])
        spans.append([int(src.min()), int(src.max()) + 1])
    return {k: np.stack(v) for k, v in out.items()} | {"spans": np.array(spans)}


class _Prep(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return prepare(self.rows[i])


class Extractor:
    def __init__(self, kind, ckpt=None, weights="teacher", device="cuda"):
        self.kind, self.dev = kind, device
        self.hands_only = False
        if kind == "model":
            from pj.model import Encoder
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            c = ck["config"]
            self.enc = Encoder(c["d"], c["depth"], c["heads"]).to(device).eval()
            self.enc.load_state_dict(ck[weights])
            self.hands_only = c.get("hands_only", False)

    @torch.no_grad()
    def encode(self, p):
        xy = torch.as_tensor(p["xy"], device=self.dev); obs = torch.as_tensor(p["obs"], device=self.dev)
        conf = torch.as_tensor(p["conf"], device=self.dev); valid = torch.as_tensor(p["valid"], device=self.dev)
        f = Fe.frame_features(xy, obs, conf, torch.zeros_like(obs))
        ins, pres = Fe.tokens(f, valid)
        B = xy.shape[0]
        if self.kind == "descriptor":
            lev = []
            for c, (l, _, fv) in enumerate(f):
                fv = (fv & valid).float()
                lb, fb = l.view(B, 16, 4, -1), fv.view(B, 16, 4, 1)
                lev.append((lb * fb).sum(2) / fb.sum(2).clamp(min=1))
            d = torch.cat(lev, -1).flatten(1)
            return {"": d.cpu().numpy().astype(np.float16)}
        vb = valid.view(B, -1, 2).any(-1)
        cells = vb[:, None, :].expand(-1, NC, -1).clone()
        if self.hands_only:
            cells[:, list(FACE_CH)] = False
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.startswith("cuda")):
            h, cls, _ = self.enc(ins, pres, cells)
        h, cls = h.float(), cls.float()
        use = (cells & pres).transpose(1, 2).flatten(1).float()                      # present tokens in valid bins
        hh = h.view(B, -1, NC, h.shape[-1]); uu = use.view(B, -1, NC)
        grp = []
        for g in GROUPS:
            if self.hands_only and g == GROUPS[2]:
                continue
            wg = uu[:, :, list(g)]
            grp.append((hh[:, :, list(g)] * wg[..., None]).sum((1, 2)) / wg.sum((1, 2)).clamp(min=1)[:, None])
        mean = (h * use[..., None]).sum(1) / use.sum(1).clamp(min=1)[:, None]
        cv = lambda x: x.cpu().numpy().astype(np.float16)
        return {"_grp": cv(torch.cat(grp, -1)), "_mean": cv(mean), "_cls": cv(cls)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="model", choices=["model", "descriptor"])
    ap.add_argument("--ckpt")
    ap.add_argument("--weights", default="teacher")
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", default="val", choices=["val", "aslc300"])
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only: every k-th row")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--root", default=str(RUNTIME / "feats"))
    a = ap.parse_args(argv)
    ex = Extractor(a.kind, a.ckpt, a.weights, a.device)
    rows = rows_for(a.set)
    if a.limit:
        rows = rows[:: max(1, len(rows) // a.limit)]
    acc, offs, spans = {}, [0], []
    t0 = time.time()

    def winit(_):
        torch.set_num_threads(1)
    dl = torch.utils.data.DataLoader(_Prep(rows), batch_size=None, num_workers=a.workers, worker_init_fn=winit,
                                     prefetch_factor=4 if a.workers else None)
    for i, p in enumerate(dl):
        p = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in p.items()}
        feats = ex.encode(p)
        for k, v in feats.items():
            acc.setdefault(k, []).append(v)
        spans.append(p["spans"]); offs.append(offs[-1] + len(p["spans"]))
        if i % 1000 == 0:
            print(i, len(rows), f"{time.time() - t0:.0f}s", flush=True)
    for k, v in acc.items():
        out = Path(a.root) / f"{a.name}{k}" / a.set
        out.mkdir(parents=True, exist_ok=False)
        np.save(out / "windows.npy", np.concatenate(v)); np.save(out / "offsets.npy", np.array(offs))
        np.save(out / "src_span.npy", np.concatenate(spans))
        with open(out / "clips.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        json.dump({**vars(a), "readout": k, "seconds": time.time() - t0}, open(out / "meta.json", "w"), indent=1)
    print("DONE", offs[-1], "windows", f"{time.time() - t0:.0f}s", list(acc))


if __name__ == "__main__":
    main()
