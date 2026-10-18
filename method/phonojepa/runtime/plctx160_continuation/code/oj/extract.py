"""Frozen window features for evaluation (no test split). Each model is evaluated with ITS OWN training preprocessing.

kinds
  rgb       : A/C checkpoint (final.pt, --weights student|teacher) or 'vjepa' (A00 frozen foundation);
              letterbox S, effective fps min(src, 25), T=16, windows every 8 samples; feature = mean of final-layer
              tokens over VALID (non-padding) tokens
  graph     : B checkpoint (final.pt teacher); T=64 at 25 fps, windows every 32 samples; feature = 4 part means
  hiera     : SignRep-layout checkpoint (SignRep reproduction / EX16 SR3D / JH3D): squash 224, native frames,
              16 consecutive frames every 8; feature = fc(mean tokens)
  vjepa_fc  : EX16 JV3D: squash 224, native frames; feature = fc(mean final tokens)
sets: val (all ASL Citizen val clips; LOSO protocol) | aslc300 (ASLC002 gallery 3,000 train + 1,500 val queries)
Output: runtime/feats/<name>/<set>/{windows.npy, offsets.npy, src_span.npy, clips.jsonl, meta.json}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oj.data.clips import decode, letterbox_frames, make_letterbox, video_meta, window_indices  # noqa: E402
from oj.data.landmarks import PARTS, load_landmarks  # noqa: E402
from oj.paths import ASLC002_RUN, ASLC_ROOT, ASLC_SPLITS, REPO, RUNTIME  # noqa: E402


def rows_for(setname):
    if setname == "val":
        out = []
        with open(ASLC_SPLITS / "val.csv", newline="") as f:
            for r in csv.DictReader(f):
                out.append({"sample_id": Path(r["Video file"]).stem, "video": str(ASLC_ROOT / "videos" / r["Video file"]),
                            "gloss": r["Gloss"], "signer": r["Participant ID"], "split": "val"})
        return out
    out = []
    for fn, role in (("gallery.jsonl", "gallery"), ("queries.jsonl", "query")):
        for l in open(ASLC002_RUN / fn):
            r = json.loads(l)
            out.append({"sample_id": Path(r["video"]).stem, "video": r["video"], "gloss": r["gloss"], "label": r["label"],
                        "signer": r["signer_id"], "split": r["split"], "role": role})
    return out


def native_frames(path):
    import cv2
    cap = cv2.VideoCapture(str(path)); fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_LINEAR))
    cap.release()
    return np.stack(fr) if fr else np.zeros((1, 224, 224, 3), np.uint8)


class Extractor:
    def __init__(self, kind, ckpt, weights, size, device="cuda"):
        self.kind, self.size, self.dev = kind, size, device
        if kind == "rgb":
            from oj.models.rgb import build_encoder
            if ckpt == "vjepa":
                self.enc = build_encoder(size, 16, 1, pretrained=True, checkpointing=False)
            else:
                ck = torch.load(ckpt, map_location="cpu", weights_only=False)
                n_out = 1 if ck.get("recipe", "AR") == "AR" else 4
                self.enc = build_encoder(ck.get("size", size), 16, n_out, pretrained=False, checkpointing=False)
                self.enc.load_state_dict(ck[weights]); self.size = ck.get("size", size)
            self.enc = self.enc.to(self.dev).eval()
        elif kind == "graph":
            from oj.data.pose_dataset import PoseStats
            from oj.models.graph import GraphEncoder
            self.enc = GraphEncoder().to(self.dev).eval()
            self.enc.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)[weights or "teacher"])
            self.stats = PoseStats(RUNTIME / "b_stats_T64.npz")
        elif kind == "hiera":
            sys.path.insert(0, str(REPO / "method" / "jepa_baseline" / "src"))
            from signrep_baseline.model import SignRep
            m = SignRep(); m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)[weights], strict=True)
            self.m = m.to(self.dev).eval()
        elif kind == "vjepa_fc":
            from oj.models.rgb import build_encoder
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            e = build_encoder(224, 16, 1, pretrained=False, checkpointing=False); e.load_state_dict(ck[weights])
            fc = torch.nn.Sequential(torch.nn.LayerNorm(768), torch.nn.Linear(768, 768, bias=False)); fc.load_state_dict(ck["fc"])
            self.e, self.fc = e.to(self.dev).eval(), fc.to(self.dev).eval()

    # ------------------------------------------------------------- CPU side (DataLoader workers)
    def prepare(self, r):
        if self.kind in ("hiera", "vjepa_fc"):
            fr = native_frames(r["video"]); n = len(fr)
            if n < 16:
                fr = np.concatenate([fr, np.repeat(fr[-1:], 16 - n, 0)])
            starts = list(range(0, max(1, len(fr) - 15), 8))
            return {"x": np.stack([fr[s:s + 16] for s in starts]), "spans": np.array([[s, min(s + 16, n)] for s in starts])}
        W, H, fps, nf = video_meta(r["video"])
        eff = min(fps, 25.0)
        if self.kind == "rgb":
            T, step = 16, 8
            nsamp = max(1, int(np.floor(nf / fps * eff)))
            starts = list(range(0, max(1, nsamp - T + 1), step))
            lb = make_letterbox(W, H, self.size)
            idxs = [window_indices(nf, fps, T, 25.0, s / eff)[0] for s in starts]
            allidx = np.unique(np.concatenate(idxs)); allidx = allidx[allidx >= 0]
            fr_all, _ = decode(r["video"], allidx)  # decode every needed frame once
            lbf = letterbox_frames(fr_all, lb); pos = {int(i): k for k, i in enumerate(allidx)}
            blank = np.zeros((self.size, self.size, 3), np.uint8)
            clips, spans = [], []
            for idx in idxs:
                clips.append(np.stack([lbf[pos[int(i)]] if i >= 0 and int(i) in pos else blank for i in idx]))
                v = idx[idx >= 0]; spans.append([int(v.min()) if len(v) else 0, int(v.max()) + 1 if len(v) else 1])
            return {"x": np.stack(clips), "spans": np.array(spans), "tv": lb.token_valid().reshape(-1)}
        from oj.data.pose_dataset import PoseWindows
        ds = PoseWindows([{"sample_id": r["sample_id"], "video": r["video"]}], {r["sample_id"]: (W, H, fps, nf)}, T=64,
                         stats=self.stats, train=False)
        nsamp = max(1, int(np.floor(nf / fps * eff)))
        starts = list(range(0, max(1, nsamp - 64 + 1), 32))
        items, spans = [], []
        for s in starts:
            it = ds.build(0, np.random.default_rng(0), start=s / eff, mask=False)
            items.append({k: it[k] for k in ("te_node", "te_loc", "obs", "t_bin")})
            idx, _, _ = window_indices(nf, fps, 64, 25.0, s / eff); v = idx[idx >= 0]; spans.append([int(v.min()), int(v.max()) + 1])
        return {"items": items, "spans": np.array(spans)}

    # ------------------------------------------------------------- GPU side
    @torch.no_grad()
    def encode(self, p):
        from oj.models.rgb import normalize_video
        if self.kind == "graph":
            import torch.nn.functional as F
            out = []
            for it in p["items"]:
                b = {k: (v[None].to(self.dev) if torch.is_tensor(v) else {kk: vv[None].to(self.dev) for kk, vv in v.items()}) for k, v in it.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h, _, _, _ = self.enc(b["te_node"], b["te_loc"], b["obs"], b["t_bin"])
                w = b["obs"].float()[..., None] * F.one_hot(self.enc.part_of, 4).float()
                out.append((torch.einsum("btjp,btjd->bpd", w, h.float()) / w.sum((1, 2)).clamp(min=1.0)[..., None]).flatten(1).cpu())
            return torch.cat(out).numpy().astype(np.float16)
        x = normalize_video(torch.as_tensor(p["x"]).to(self.dev))
        feats = []
        for i in range(0, len(x), 16):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if self.kind == "hiera":
                    f = self.m.encode(x[i:i + 16], mask_ratio=0.0)["features"]
                elif self.kind == "vjepa_fc":
                    f = self.fc(self.e(x[i:i + 16]).float().mean(1))
                else:
                    h = self.enc(x[i:i + 16]).float()
                    tvn = torch.as_tensor(p["tv"], device=self.dev).repeat(8).float()
                    f = (h * tvn[None, :, None]).sum(1) / tvn.sum()
            feats.append(f.float().cpu())
        return torch.cat(feats).numpy().astype(np.float16)


class _Prep(torch.utils.data.Dataset):
    def __init__(self, ex, rows):
        self.ex, self.rows = ex, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.ex.prepare(self.rows[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["rgb", "graph", "hiera", "vjepa_fc"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default="student")
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", default="val", choices=["val", "aslc300"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only: every k-th row")
    a = ap.parse_args()
    out = RUNTIME / "feats" / a.name / a.set; out.mkdir(parents=True, exist_ok=False)
    ap_workers = 6
    ex = Extractor(a.kind, a.ckpt, a.weights, a.size)
    rows = rows_for(a.set)
    if a.limit:
        rows = rows[:: max(1, len(rows) // a.limit)]
    W, offs, spans = [], [0], []
    t0 = time.time()
    import cv2

    def winit(_):
        torch.set_num_threads(1); cv2.setNumThreads(1)
    dl = torch.utils.data.DataLoader(_Prep(ex, rows), batch_size=None, num_workers=ap_workers, prefetch_factor=4,
                                     worker_init_fn=winit)  # CPU decode/preprocess in parallel, GPU encode here
    for i, p in enumerate(dl):
        w = ex.encode(p); W.append(w); spans.append(p["spans"]); offs.append(offs[-1] + len(w))
        if i % 1000 == 0:
            print(i, len(rows), f"{time.time() - t0:.0f}s", flush=True)
    np.save(out / "windows.npy", np.concatenate(W)); np.save(out / "offsets.npy", np.array(offs)); np.save(out / "src_span.npy", np.concatenate(spans))
    with open(out / "clips.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump({**vars(a), "seconds": time.time() - t0}, open(out / "meta.json", "w"), indent=1)
    print("DONE", offs[-1], "windows", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
