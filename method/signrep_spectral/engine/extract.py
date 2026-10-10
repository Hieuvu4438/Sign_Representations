"""RGB-only feature extraction: 16-frame windows, uniform mean pooling (float64) then L2."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..data.manifest import read_jsonl
from ..data.video import decode_frames, normalize_frames, window_starts, worker_init

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_EPS = 1e-12


class VideoDataset(Dataset):
    def __init__(self, recs):
        self.recs = recs

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        try:
            frames, pts = decode_frames(str(Path(r["source_root"]) / r["relative_path"]))
            return {"i": i, "frames": torch.from_numpy(frames), "pts": pts, "error": None}
        except Exception as e:
            return {"i": i, "frames": None, "pts": None, "error": repr(e)}


def _collate(items):
    return items[0]


def build_encoder(cfg, encoder_path, reference: bool):
    from ..model import encoder_from_vendor_model, state_hash
    from ..vendor import load_vendor_model
    from .checkpoint import load_exported_encoder, sha256_file

    if reference:
        base, _ = load_vendor_model(cfg["paths"]["vendor_root"], cfg["paths"]["checkpoint"], cfg["checkpoint_sha256"], "cpu")
        enc = encoder_from_vendor_model(base)
        src = {"kind": "reference_signrep_checkpoint", "sha256": cfg["checkpoint_sha256"]}
    else:
        enc = load_exported_encoder(cfg["paths"]["vendor_root"], encoder_path)
        src = {"kind": "exported_encoder", "path": str(encoder_path), "sha256": sha256_file(encoder_path)}
    enc.eval()
    src["encoder_hash"] = state_hash(enc)
    return enc, src


def extract(cfg, ctx, encoder_path, reference: bool, splits, limit: int = 0) -> None:
    from ..model import attention_kernel

    ev = cfg["evaluation"]
    device = torch.device(cfg["train"]["device"])
    enc, src = build_encoder(cfg, encoder_path, reference)
    enc.to(device)
    T, stride, Bmax = ev["window_length"], ev["stride"], ev["batch_size"]
    amp = ev["amp"] and device.type == "cuda"
    manifest_path = Path(cfg["paths"]["manifest"])
    manifest_path = manifest_path if manifest_path.is_absolute() else REPO_ROOT / manifest_path
    all_recs = read_jsonl(manifest_path)
    meta = {"encoder": src, "window_length": T, "stride": stride, "pooling": ev["pooling"],
            "precision": "bf16_autocast" if amp else "fp32", "attention_kernel": cfg["model"]["attention_kernel"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "splits": {}}
    for split in splits:
        recs = [r for r in all_recs if r["official_split"] == split]
        if limit:
            recs = recs[:limit]
        N = len(recs)
        sums = np.zeros((N, 768), np.float64)
        lat_sums = np.zeros((N, 768), np.float64)
        nwin = np.zeros(N, np.int64)
        status = np.array(["pending"] * N, dtype=object)
        frames_decoded = np.zeros(N, np.int64)
        loader = DataLoader(VideoDataset(recs), batch_size=1, shuffle=False, num_workers=ev["num_workers"],
                            collate_fn=_collate, prefetch_factor=8, worker_init_fn=worker_init)
        buf_x, buf_owner = [], []
        t0 = time.time()

        def flush():
            if not buf_x:
                return
            x = torch.stack(buf_x)  # [b,T,H,W,3] uint8 on device
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = enc.encode(normalize_frames(x), mask_ratio=0.0)
            f = out["features"].float().cpu().numpy().astype(np.float64)
            l = out["latent"].float().cpu().numpy().astype(np.float64)
            if not (np.isfinite(f).all() and np.isfinite(l).all()):
                raise FloatingPointError("non-finite features")
            for j, owner in enumerate(buf_owner):
                sums[owner] += f[j]
                lat_sums[owner] += l[j]
                nwin[owner] += 1
            buf_x.clear()
            buf_owner.clear()

        with attention_kernel(cfg["model"]["attention_kernel"]):
            for k, item in enumerate(loader):
                i = item["i"]
                if item["error"]:
                    status[i] = f"decode_error:{item['error'][:80]}"
                    continue
                frames = item["frames"]
                frames_decoded[i] = len(frames)
                if recs[i].get("num_frames") is not None and len(frames) != recs[i]["num_frames"]:
                    status[i] = "frame_count_changed_since_audit"
                    continue
                starts = window_starts(len(frames), T, stride)
                if len(starts) == 0:
                    status[i] = "short_video"
                    continue
                fr = frames.to(device, non_blocking=True)
                for s in starts:
                    buf_x.append(fr[s : s + T])
                    buf_owner.append(i)
                    if len(buf_x) >= Bmax:
                        flush()
                status[i] = "ok"
                if k % 2000 == 0:
                    print(f"[{split}] {k}/{N} videos, {time.time()-t0:.0f}s", flush=True)
            flush()
        ok = status == "ok"
        pooled = np.zeros((N, 768), np.float32)
        pooled_lat = np.zeros((N, 768), np.float32)
        for arr, s in ((pooled, sums), (pooled_lat, lat_sums)):
            mean = s[ok] / nwin[ok, None]
            norm = np.linalg.norm(mean, axis=1, keepdims=True)
            bad = norm[:, 0] < POOL_EPS
            mean = mean / np.maximum(norm, POOL_EPS)
            arr[ok] = mean.astype(np.float32)
            if bad.any():
                idx = np.flatnonzero(ok)[bad]
                status[idx] = "zero_pooled_norm"
        out = ctx.dir / f"features_{split}.npz"
        with open(out, "xb") as fh:
            np.savez(fh, sample_id=np.array([r["sample_id"] for r in recs]),
                     label_id=np.array([r["label_id"] for r in recs], np.int64),
                     signer_id=np.array([r["signer_id"] for r in recs]),
                     pooled_features=pooled, pooled_latent=pooled_lat, n_windows=nwin,
                     status=status.astype(str), frames_decoded=frames_decoded)
        counts = {s: int((status == s).sum()) for s in set(status.tolist())}
        meta["splits"][split] = {"videos": N, "status_counts": counts, "windows": int(nwin.sum()),
                                 "seconds": time.time() - t0, "file": out.name,
                                 "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
        print(json.dumps({split: meta["splits"][split]}), flush=True)
        ctx.write_json("feature_manifest.json", meta)
