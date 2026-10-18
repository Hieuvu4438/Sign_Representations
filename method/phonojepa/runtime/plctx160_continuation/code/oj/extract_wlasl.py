"""Inference-only extraction against a locked WLASL manifest. Compact resumable shards.

Stores pooled video embeddings rather than all windows (under 1GB for this suite).
No parameters are updated, including normalization statistics and EMA teachers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oj.extract import Extractor
from oj.data.clips import make_letterbox, letterbox_frames, video_meta
from oj.data.landmarks import load_landmarks
from oj.data.pose_dataset import PoseWindows
from oj.wlasl_retrieval import DATA, DEFAULT_OUT, ROOT, evaluate_pooled, pool_windows, protocol, read_rows, save_json, sha256


def fast_no_cls_attention(self, x):
    """Equivalent zero-class-token Hiera attention using 4D SDPA (Flash-compatible).

    Only folds heads/windows into one axis; all projections and q-pooling stay fixed.
    The released public SignRep cache uses the same SDPA layout optimization.
    """
    if self.num_cls_tokens != 0:
        raise ValueError("Fast attention requires zero class tokens")
    B, N, _ = x.shape
    nw = N // (self.q_stride * self.window_size) if self.use_mask_unit_attn else 1
    qkv = self.qkv(x).reshape(B, -1, nw, 3, self.heads, self.head_dim).permute(3, 0, 4, 2, 1, 5)
    q, k, v = qkv.unbind(0)
    if self.q_stride > 1:
        q = q.view(B, self.heads, nw, self.q_stride, -1, self.head_dim).max(dim=3).values
    nq, nk = q.shape[3], k.shape[3]
    y = torch.nn.functional.scaled_dot_product_attention(
        q.reshape(B, self.heads * nw, nq, self.head_dim).contiguous(),
        k.reshape(B, self.heads * nw, nk, self.head_dim).contiguous(),
        v.reshape(B, self.heads * nw, nk, self.head_dim).contiguous())
    y = y.view(B, self.heads, nw, nq, self.head_dim).transpose(1, 3).reshape(B, -1, self.dim_out)
    return self.proj(y)


class WlaslPose(PoseWindows):
    def _lm(self, i):
        if i not in self._cache:
            row = self.rows[i]; W, H, fps, nf = self.meta[row["sample_id"]]
            lm = load_landmarks(row["sample_id"], W, H, root=DATA / "keypoint")
            if len(lm.xy) != nf:
                raise RuntimeError(f"Pose/video frame mismatch: {row['sample_id']}: {len(lm.xy)} != {nf}")
            self._cache[i] = (lm, (W, H, fps, nf))
        return self._cache[i]


class FrozenExtractor(Extractor):
    def __init__(self, spec, device):
        super().__init__(spec["kind"], spec["ckpt"], spec["weights"], spec["size"], device=device)
        self.activity = None
        if self.kind == "hiera":
            self.activity = self.m.active_head
            # Exact axis regrouping avoids the slow five-dimensional SDPA fallback.
            from signrep_baseline.upstream_signrep.models.hiera.hiera_clstoken import MaskUnitAttention
            for module in self.m.modules():
                if isinstance(module, MaskUnitAttention):
                    if module.num_cls_tokens != 0:
                        raise ValueError("Unexpected Hiera class tokens")
                    module.forward = types.MethodType(fast_no_cls_attention, module)
        elif spec["activity_weighted_available"]:
            state = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)["model"]
            self.activity = torch.nn.Sequential(torch.nn.Linear(768, 128), torch.nn.GELU(), torch.nn.Linear(128, 2))
            self.activity.load_state_dict({k.removeprefix("active_head."): v for k, v in state.items() if k.startswith("active_head.")}, strict=True)
            self.activity = self.activity.to(device).eval()
        for value in vars(self).values():
            if isinstance(value, torch.nn.Module):
                value.eval().requires_grad_(False)

    def prepare(self, row):
        W, H, fps, nf = video_meta(row["video"])
        if W <= 0 or H <= 0 or nf <= 0 or fps <= 0:
            raise RuntimeError(f"Invalid video metadata: {row['video']}")
        if self.kind == "graph":
            ds = WlaslPose([row], {row["sample_id"]: (W, H, fps, nf)}, T=64, stats=self.stats, train=False)
            n = max(1, int(np.floor(nf / fps * min(fps, 25))))
            starts = list(range(0, max(1, n - 63), 32))
            items = []
            for s in starts:
                item = ds.build(0, np.random.default_rng(0), start=s / min(fps, 25), mask=False)
                items.append({k: item[k] for k in ("te_node", "te_loc", "obs", "t_bin")})
            return dict(items=items, n_frames=nf, fps=fps, nwin=len(items))
        cap = cv2.VideoCapture(row["video"]); frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            if self.kind != "rgb":
                f = cv2.resize(f, (224, 224), interpolation=cv2.INTER_LINEAR)
            frames.append(f)
        cap.release()
        if len(frames) != nf:
            raise RuntimeError(f"Video decode mismatch: {row['sample_id']}: {len(frames)} != {nf}")
        frames = np.stack(frames)
        if self.kind == "rgb":
            lb = make_letterbox(W, H, self.size)
            frames = letterbox_frames(frames, lb)
            tv = lb.token_valid().reshape(-1)
        else:
            tv = np.ones((224 // 16) ** 2, bool)
        if len(frames) < 16:
            frames = np.concatenate([frames, np.repeat(frames[-1:], 16 - len(frames), axis=0)])
        starts = list(range(0, len(frames) - 15, 2))
        return dict(x=np.stack([frames[s:s + 16] for s in starts]), tv=tv,
                    n_frames=nf, fps=fps, nwin=len(starts))

    @torch.inference_mode()
    def encode_pooled(self, prepared, batch):
        from oj.models.rgb import normalize_video
        if self.kind == "graph":
            # Existing graph encoder and exact training geometry/part pooling.
            # Its autocast is valid on CUDA; use an explicit CPU path for smoke validation.
            features = []
            for item in prepared["items"]:
                b = {k: (v[None].to(self.dev) if torch.is_tensor(v) else {kk: vv[None].to(self.dev) for kk, vv in v.items()})
                     for k, v in item.items()}
                amp = torch.autocast("cuda", dtype=torch.bfloat16) if self.dev == "cuda" else nullcontext()
                with amp:
                    h, _, _, _ = self.enc(b["te_node"], b["te_loc"], b["obs"], b["t_bin"])
                w = b["obs"].float()[..., None] * torch.nn.functional.one_hot(self.enc.part_of, 4).float()
                f = (torch.einsum("btjp,btjd->bpd", w, h.float()) / w.sum((1, 2)).clamp(min=1)[..., None]).flatten(1)
                features.append(f.cpu().numpy())
            return pool_windows(np.concatenate(features))
        features, activities = [], []
        for i in range(0, len(prepared["x"]), batch):
            x = normalize_video(torch.as_tensor(prepared["x"][i:i + batch]).to(self.dev))
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if self.dev == "cuda" else nullcontext()
            with amp:
                if self.kind == "hiera":
                    f = self.m.encode(x, mask_ratio=0.0)["features"]
                elif self.kind == "vjepa_fc":
                    f = self.fc(self.e(x).float().mean(1))
                else:
                    h = self.enc(x).float()
                    tv = torch.as_tensor(prepared["tv"], device=self.dev).repeat(8).float()
                    f = (h * tv[None, :, None]).sum(1) / tv.sum()
                if self.activity is not None:
                    activities.append(self.activity(f).float().cpu().numpy())
            features.append(f.float().cpu().numpy())
        return pool_windows(np.concatenate(features), np.concatenate(activities) if activities else None)


class PrepareRows(torch.utils.data.Dataset):
    def __init__(self, ex, rows):
        self.ex, self.rows = ex, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.ex.prepare(self.rows[i])


def worker_init(_):
    torch.set_num_threads(1); cv2.setNumThreads(1)


def extract_split(ex, rows, target, provenance, workers, batch, shard_size):
    target.mkdir(parents=True, exist_ok=True)
    provenance_path = target / "provenance.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != provenance:
            raise RuntimeError("Refusing to resume mismatched extraction")
    else:
        if any(target.iterdir()):
            raise RuntimeError("Unrecognized partial directory")
        save_json(provenance_path, provenance)
    t0 = time.monotonic()
    for start in range(0, len(rows), shard_size):
        path = target / f"{start:07d}.npz"
        if path.exists():
            with np.load(path) as z:
                assert z["sample_id"].tolist() == [r["sample_id"] for r in rows[start:start + shard_size]]
            continue
        values, fallbacks, counts = {}, [], []
        part = rows[start:start + shard_size]
        kwargs = dict(batch_size=None, num_workers=workers, worker_init_fn=worker_init)
        if workers:
            kwargs.update(prefetch_factor=1)
        loader = torch.utils.data.DataLoader(PrepareRows(ex, part), **kwargs)
        for prepared in loader:
            pooled, fb = ex.encode_pooled(prepared, batch)
            for key, value in pooled.items():
                values.setdefault(key, []).append(value)
            fallbacks.append(fb); counts.append(prepared["nwin"])
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **{k: np.stack(v) for k, v in values.items()},
                 sample_id=np.array([r["sample_id"] for r in part]), nwin=np.asarray(counts), fallback=np.asarray(fallbacks))
        tmp.replace(path)
        elapsed = time.monotonic() - t0
        save_json(target / "progress.json", dict(completed=min(start + shard_size, len(rows)), total=len(rows), seconds_this_process=elapsed))
        print(target.name, min(start + shard_size, len(rows)), "/", len(rows), f"{elapsed:.1f}s", flush=True)
    arrays, total_windows, fb_count = {}, 0, 0
    for start in range(0, len(rows), shard_size):
        with np.load(target / f"{start:07d}.npz") as z:
            for k in ("avg", "weighted"):
                if k in z:
                    arrays.setdefault(k, []).append(z[k])
            total_windows += int(z["nwin"].sum()); fb_count += int(z["fallback"].sum())
    return {k: np.concatenate(v) for k, v in arrays.items()}, dict(total_windows=total_windows, zero_activity_fallback=fb_count,
                                                                zero_vectors={k: int((np.linalg.norm(np.concatenate(v), axis=1) == 0).sum()) for k, v in arrays.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shard", type=int, default=64)
    ap.add_argument("--scheduler-tag", default="", help="Compatibility marker for existing GPU schedulers")
    ap.add_argument("--smoke", action="store_true", help="Two clips on CPU/GPU; never writes evaluation results")
    a = ap.parse_args(); a.out = a.out.resolve()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    p = protocol(a.out); spec = p["models"][a.name]
    if spec["kind"] == "public":
        raise ValueError("Public model uses validated existing cache; run wlasl_retrieval.py public")
    if sha256(spec["checkpoint_file"]) != spec["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed after protocol lock")
    if spec["kind"] == "graph" and sha256(p["graph_statistics"]["path"]) != p["graph_statistics"]["sha256"]:
        raise RuntimeError("ASL Citizen graph normalization changed")
    ex = FrozenExtractor(spec, a.device)
    if a.smoke:
        rows = read_rows(a.out / "train.jsonl")
        # Include one normal clip and the known ten-frame short clip.
        ids = [rows[0], next(r for r in rows if r["sample_id"] == "18223")]
        modules = {key: value for key, value in vars(ex).items() if isinstance(value, torch.nn.Module)}
        snapshots = {name: {key: value.detach().clone() for key, value in module.state_dict().items()}
                     for name, module in modules.items()}
        for row in ids:
            prepared = ex.prepare(row)
            pooled, fb = ex.encode_pooled(prepared, a.batch)
            assert all(np.isfinite(v).all() for v in pooled.values())
            assert ("weighted" in pooled) == spec["activity_weighted_available"]
            print("SMOKE", a.name, row["sample_id"], prepared["nwin"], {k: list(v.shape) for k, v in pooled.items()}, "fallback", fb, flush=True)
        assert all(torch.equal(value, snapshots[name][key]) for name, module in modules.items()
                   for key, value in module.state_dict().items())
        assert not any(v.requires_grad for value in vars(ex).values() if isinstance(value, torch.nn.Module) for v in value.parameters())
        print("SMOKE PASS: no parameter/buffer changes; no result written", flush=True)
        return
    if a.device != "cuda":
        raise ValueError("Full CPU run disabled; use --smoke for CPU checks")
    # One model worker per output root. No concurrent writers or silent overwrites.
    import fcntl
    with (a.out / "extract.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (a.out / f"results/{a.name}.json").exists():
            print("Already evaluated", a.name); return
        pooled = {}
        meta = {"inference": dict(torch_version=torch.__version__, numpy_version=np.__version__,
                                  cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                                  batch_windows=a.batch, decoder_workers=a.workers,
                                  fast_zero_cls_hiera_attention=(spec["kind"] == "hiera"),
                                  parameters_frozen=True, optimizer_created=False)}
        for split in ("train", "test"):
            rows = read_rows(a.out / f"{split}.jsonl")
            provenance = dict(model=a.name, checkpoint_sha256=spec["checkpoint_sha256"],
                              manifest_sha256=p["manifest_sha256"][split], spec=spec, precision="bf16 inference/fp32 pooling", shard_size=a.shard,
                              implementation_sha256={str(path.relative_to(ROOT)): sha256(path)
                                                     for path in [Path(__file__), ROOT / "oj/extract.py", ROOT / "oj/wlasl_retrieval.py",
                                                                  ROOT / "oj/data/landmarks.py", ROOT / "oj/data/pose_dataset.py",
                                                                  ROOT / "oj/features/geometry.py", ROOT / "oj/models/rgb.py", ROOT / "oj/models/graph.py"]})
            pooled[split], meta[split] = extract_split(ex, rows, a.out / "embeddings" / a.name / split,
                                                       provenance, a.workers, a.batch, a.shard)
        if sha256(spec["checkpoint_file"]) != spec["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed during inference")
        evaluate_pooled(a.out, a.name, pooled["train"], pooled["test"], meta)


if __name__ == "__main__":
    main()
