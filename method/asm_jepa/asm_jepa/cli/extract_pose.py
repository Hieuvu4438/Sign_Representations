"""RTMW whole-body (COCO-133) pose extraction with signer tracking and provenance (§6).

Run in the pose environment (rtmlib + onnxruntime-gpu + opencv). Heavy imports are lazy.
Per frame: detector once -> boxes; pose model on those boxes (never a second `estimator(frame)` call).
Signer selection: first frame = large, central box; later frames = shoulder/box continuity with the previous
selection (ambiguity is flagged, never resolved by confidence or labels). Zero detections -> missing frame.
No interpolation or smoothing. Timestamps: CAP_PROP_POS_MSEC when monotonic, else index / fps (flagged).
"""
from __future__ import annotations

import argparse
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from asm_jepa import SCHEMA_VERSION
from asm_jepa.cli.common import add_config_args, config_from_args, run
from asm_jepa.data.manifest import SPLITS, load_split
from asm_jepa.data.pose_store import PoseClip, is_valid_cache, write_clip
from asm_jepa.utils import append_jsonl, now, read_json, resolve, sha256_file, write_json


def fetch_model(url: str, cache: Path, local_dir: Path | None = None) -> Path:
    """Return a local .onnx for `url` inside `cache` (download + unzip once; or copy from --model-source-dir)."""
    name = Path(url).stem
    target = cache / f"{name}.onnx"
    if target.is_file():
        return target
    cache.mkdir(parents=True, exist_ok=True)
    if local_dir is not None and (local_dir / f"{name}.onnx").is_file():
        shutil.copy2(local_dir / f"{name}.onnx", target)
        return target
    zpath = cache / f"{name}.zip"
    if not zpath.is_file():
        print(f"downloading {url}", flush=True)
        tmp = zpath.with_suffix(".zip.part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status} for {url}")
            shutil.copyfileobj(r, f)
        tmp.replace(zpath)
    with zipfile.ZipFile(zpath) as z:
        onnx = [n for n in z.namelist() if n.endswith(".onnx")]
        if len(onnx) != 1:
            raise RuntimeError(f"{zpath} holds {len(onnx)} onnx files")
        with z.open(onnx[0]) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return target


def build_estimator(pc: dict, cache: Path, local_dir: Path | None):
    import torch  # noqa: F401  (load CUDA libraries before creating ORT sessions)
    from rtmlib import Wholebody

    det = fetch_model(pc["detector_url"], cache, local_dir)
    pose = fetch_model(pc["pose_url"], cache, local_dir)
    est = Wholebody(det=str(det), det_input_size=tuple(pc["detector_input_size"]), pose=str(pose),
                    pose_input_size=tuple(pc["pose_input_size"]), mode=pc["mode"], to_openpose=False,
                    backend=pc["backend"], device=pc["device"])
    providers = {"det": est.det_model.session.get_providers() if hasattr(est.det_model, "session") else [],
                 "pose": est.pose_model.session.get_providers() if hasattr(est.pose_model, "session") else []}
    if pc["require_cuda_provider"] and pc["device"] == "cuda":
        for k, v in providers.items():
            if "CUDAExecutionProvider" not in v:
                raise RuntimeError(f"{k} session has no CUDA provider ({v}); refusing silent CPU fallback")
    hashes = {"detector_onnx_sha256": sha256_file(det), "pose_onnx_sha256": sha256_file(pose),
              "detector_file": det.name, "pose_file": pose.name}
    return est, providers, hashes


def _shoulder_mid(k, s, thr):
    ok = s[[5, 6]] >= thr
    if ok.all():
        return k[[5, 6]].mean(0), float(np.linalg.norm(k[5] - k[6]))
    return None, None


def select_signer(bboxes, kps, scores, prev, wh, thr):
    """-> (index or None, flags)."""
    n = len(bboxes)
    if n == 0:
        return None, {"zero_detections": True}
    W, H = wh
    if prev is None or n == 1:
        if n == 1:
            return 0, {}
        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        cx, cy = (bboxes[:, 0] + bboxes[:, 2]) / 2, (bboxes[:, 1] + bboxes[:, 3]) / 2
        central = np.exp(-(((cx - W / 2) / (0.35 * W)) ** 2 + ((cy - H / 2) / (0.5 * H)) ** 2))
        score = area * central
        order = np.argsort(-score)
        return int(order[0]), {"multi_person": True, "ambiguous": bool(score[order[1]] > 0.8 * score[order[0]])}
    pmid, pwidth = prev
    costs = []
    for i in range(n):
        mid, _ = _shoulder_mid(kps[i], scores[i], thr)
        if mid is None:
            c = (bboxes[i, :2] + bboxes[i, 2:]) / 2
            costs.append(np.linalg.norm(c - pmid) / max(pwidth, 1.0) + 1.0)
        else:
            costs.append(np.linalg.norm(mid - pmid) / max(pwidth, 1.0))
    costs = np.array(costs)
    order = np.argsort(costs)
    amb = bool(costs[order[1]] < 1.2 * costs[order[0]] + 0.05)
    return int(order[0]), {"multi_person": True, "ambiguous": amb}


def extract_video(est, path: Path, thr: float):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    xs, ss, msec, fv, flags = [], [], [], [], {"zero_detections": 0, "multi_person": 0, "ambiguous": 0}
    wh = None
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        msec.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        H, W = frame.shape[:2]
        wh = (W, H)
        bboxes = est.det_model(frame)
        bboxes = np.asarray(bboxes).reshape(-1, 4)
        if len(bboxes):
            kps, scs = est.pose_model(frame, bboxes=bboxes)
            kps, scs = np.asarray(kps), np.asarray(scs)
            assert kps.shape == (len(bboxes), 133, 2) and scs.shape == (len(bboxes), 133), (kps.shape, scs.shape)
        else:
            kps, scs = np.zeros((0, 133, 2)), np.zeros((0, 133))
        idx, fl = select_signer(bboxes, kps, scs, prev, wh, thr)
        for k in flags:
            flags[k] += int(bool(fl.get(k)))
        if idx is None:
            xs.append(np.full((133, 2), np.nan))
            ss.append(np.zeros(133))
            fv.append(False)
            continue
        xs.append(kps[idx])
        ss.append(scs[idx])
        fv.append(not fl.get("ambiguous", False))
        mid, width = _shoulder_mid(kps[idx], scs[idx], thr)
        if mid is not None:
            prev = (mid, width)
    cap.release()
    T = len(xs)
    if T == 0:
        raise RuntimeError("no decodable frames")
    msec = np.array(msec, np.float64) / 1000.0
    if T > 1 and np.all(np.diff(msec) > 0):
        ts, mode = msec, "container_pts_cap_prop_pos_msec"
    else:
        ts, mode = np.arange(T, dtype=np.float64) / (fps if fps > 0 else 30.0), "cfr_index_over_container_fps"
    return (np.array(xs, np.float32), np.array(ss, np.float32), np.array(fv, bool), ts, mode, wh, fps, flags)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default="configs/pose/rtmw133.yaml")
    ap.add_argument("--split", default="train", choices=SPLITS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--model-source-dir", help="directory with already-downloaded .onnx files to copy (hashed)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args, "pose")
    pc = cfg["pose"]
    if pc["to_openpose"] or pc["interpolate"] or pc["smooth"] or pc["detection_interval_frames"] != 1:
        raise ValueError("only raw COCO-133, per-frame detection, no interpolation/smoothing is implemented")
    root = resolve(cfg["data"]["processed_root"])
    version = read_json(root / "dataset_version.json")
    raw_root = Path(version["raw_root"])
    rows = load_split(root, args.split)
    if args.limit:
        rows = rows[: args.limit]
    todo = [r for r in rows if not (args.resume and is_valid_cache(root / r["pose_relpath"]))]
    if not args.resume and not pc["overwrite_valid_cache"]:
        clash = [r for r in todo if (root / r["pose_relpath"]).is_file()]
        if clash:
            raise ValueError(f"{len(clash)} caches exist; use --resume or pose.overwrite_valid_cache=true")
    print(f"{args.split}: {len(rows)} clips, {len(todo)} to extract")
    if args.dry_run:
        return 0
    est, providers, hashes = build_estimator(pc, resolve(pc["model_cache"]),
                                             Path(args.model_source_dir) if args.model_source_dir else None)
    print("providers", providers)
    index = root / f"poses_extract_index_{args.split}.jsonl"
    t0, frames_done = time.time(), 0
    for i, r in enumerate(todo):
        rec = {"sample_id": r["sample_id"], "official_split": args.split, "time": now()}
        try:
            video = raw_root / r["video_relpath"]
            xy, sc, fv, ts, mode, wh, fps, flags = extract_video(est, video, pc["confidence_threshold"])
            finite = np.isfinite(xy).all(-1)
            observed = finite & fv[:, None] & (sc >= pc["confidence_threshold"])
            meta = {"schema": "coco_wholebody_133_raw", "model_schema": SCHEMA_VERSION, "extractor": "rtmlib.Wholebody",
                    "rtmlib_revision_requested": pc["revision"], "mode": pc["mode"], "backend": pc["backend"],
                    "providers": providers, **hashes, "confidence_threshold": pc["confidence_threshold"],
                    "confidence_mode": "raw_unnormalized_extractor_score",
                    "coordinate_convention": "pixels in the native frame, origin top-left", "timestamp_mode": mode,
                    "container_fps": fps, "tracking_flags": flags, "failed_frames": int((~fv).sum()),
                    "source_video_size_bytes": video.stat().st_size, "interpolated": False, "smoothed": False}
            clip = PoseClip(xy=np.where(observed[..., None], xy, np.nan).astype(np.float32), score=sc,
                            observed=observed, timestamps_sec=ts, frame_valid=fv, image_wh=np.array(wh, np.int32),
                            source_frame_index=np.arange(len(xy), dtype=np.int32), meta=meta)
            write_clip(root / r["pose_relpath"], clip)
            frames_done += len(xy)
            rec.update(status="ok", frames=len(xy), **{f"flag_{k}": v for k, v in flags.items()}, timestamp_mode=mode)
        except Exception as e:  # noqa: BLE001
            rec.update(status="error", error=f"{type(e).__name__}: {e}")
        append_jsonl(index, rec)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(todo)} clips, {frames_done / el:.1f} frames/s", flush=True)
    el = time.time() - t0
    write_json(root / f"pose_extract_summary_{args.split}.json",
               {"time": now(), "clips": len(todo), "frames": frames_done, "seconds": el,
                "frames_per_s": frames_done / max(el, 1e-9), "providers": providers, **hashes})
    return 0


if __name__ == "__main__":
    run(main)
