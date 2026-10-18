"""Import an existing pose cache into the project's npz contract (§2.3/§6.3), with explicit schema checks.

Supported adapter:
  aslc_pkl_coco133_normxy  — one pickle per official video stem with keys `keypoints` (list over frames of
      [n_person,133,2], x/W and y/H normalised by the video size) and `scores` (list of [n_person,133], raw
      unnormalised RTMW/SimCC scores). Exactly one detected person per frame is required for a valid frame;
      0 or >1 persons -> frame_valid=False (never borrow a person from another frame).
Timestamps: index / container fps (`cfr_index_over_container_fps`), because the pickles have no PTS.
Pickles are trusted local files produced by this group's extractor; pass --source-dir explicitly.
"""
from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from asm_jepa import SCHEMA_VERSION
from asm_jepa.cli.common import run
from asm_jepa.data.manifest import SPLITS, load_split
from asm_jepa.data.pose_store import PoseClip, is_valid_cache, write_clip
from asm_jepa.utils import append_jsonl, now, read_json, resolve, sha256_file, write_json

ADAPTERS = ("aslc_pkl_coco133_normxy",)


def convert_one(task):
    row, src_dir, out_root, threshold, overwrite = task
    out = Path(out_root) / row["pose_relpath"]
    rec = {"sample_id": row["sample_id"], "official_split": row["official_split"], "pose_relpath": row["pose_relpath"]}
    if out.is_file() and not overwrite and is_valid_cache(out):
        return {**rec, "status": "exists"}
    src = Path(src_dir) / f"{row['source_stem']}.pkl"
    if not src.is_file():
        return {**rec, "status": "missing_source", "source": str(src)}
    W, H, fps, nf = row.get("width"), row.get("height"), row.get("fps"), row.get("num_frames")
    if not (W and H and fps and fps > 0):
        return {**rec, "status": "no_video_header"}
    try:
        with open(src, "rb") as f:
            d = pickle.load(f)
        kps, scs = d["keypoints"], d["scores"]
        T = len(kps)
        if T == 0 or len(scs) != T:
            return {**rec, "status": "bad_source_shape", "frames": T}
        xy = np.full((T, 133, 2), np.nan, np.float32)
        sc = np.zeros((T, 133), np.float32)
        fv = np.zeros(T, bool)
        n_persons = np.zeros(T, np.int32)
        for t, (k, s) in enumerate(zip(kps, scs)):
            k, s = np.asarray(k, np.float64), np.asarray(s, np.float64)
            if k.ndim == 2:
                k, s = k[None], s[None]
            if k.shape[-2:] != (133, 2) or s.shape[-1] != 133:
                return {**rec, "status": "schema_mismatch", "frame": t, "shape": list(k.shape)}
            n_persons[t] = k.shape[0]
            if k.shape[0] == 1:
                xy[t] = (k[0] * np.array([W, H])).astype(np.float32)
                sc[t] = s[0]
                fv[t] = True
        finite = np.isfinite(xy).all(-1) & np.isfinite(sc)
        observed = finite & fv[:, None] & (sc >= threshold)
        xy = np.where(observed[..., None], xy, np.nan).astype(np.float32)
        sc = np.where(np.isfinite(sc), sc, 0.0).astype(np.float32)
        meta = {"schema": "coco_wholebody_133_raw", "model_schema": SCHEMA_VERSION, "adapter": ADAPTERS[0],
                "source_file": str(src), "source_size_bytes": src.stat().st_size,
                "source_sha256": sha256_file(src), "coordinate_convention": "pixels, origin top-left, x right, y down",
                "source_coordinate_convention": "x/W, y/H normalised", "image_wh_from": "video header probe",
                "confidence_mode": "raw_unnormalized_extractor_score", "confidence_threshold": threshold,
                "timestamp_mode": "cfr_index_over_container_fps", "container_fps": fps,
                "video_num_frames_header": nf, "pose_num_frames": T, "frame_count_mismatch": bool(nf and nf != T),
                "frames_zero_person": int((n_persons == 0).sum()), "frames_multi_person": int((n_persons > 1).sum()),
                "tracking": "source provides at most one selected person per frame",
                "interpolated": False, "smoothed": False}
        clip = PoseClip(xy=xy, score=sc, observed=observed, timestamps_sec=np.arange(T, dtype=np.float64) / fps,
                        frame_valid=fv, image_wh=np.array([W, H], np.int32),
                        source_frame_index=np.arange(T, dtype=np.int32), meta=meta)
        write_clip(out, clip)
        return {**rec, "status": "ok", "frames": T, "frame_count_mismatch": meta["frame_count_mismatch"],
                "frames_invalid": int((~fv).sum()), "frames_multi_person": meta["frames_multi_person"]}
    except Exception as e:  # noqa: BLE001 - record and continue with independent clips
        return {**rec, "status": "error", "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-root", default="data/processed/aslc")
    ap.add_argument("--source-dir", required=True, help="directory with <official video stem>.pkl files")
    ap.add_argument("--format", default=ADAPTERS[0], choices=ADAPTERS)
    ap.add_argument("--split", default="train", choices=SPLITS + ("all",))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true", help="skip clips whose valid cache already exists")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--confidence-threshold", type=float, default=0.20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = resolve(args.processed_root)
    if not (root / "dataset_version.json").is_file():
        raise FileNotFoundError(f"{root}/dataset_version.json missing; run prepare_aslc first")
    splits = SPLITS if args.split == "all" else (args.split,)
    rows = [r for s in splits for r in load_split(root, s)]
    if args.limit:
        rows = rows[: args.limit]
    src = resolve(args.source_dir)
    missing = [r["sample_id"] for r in rows if not (src / f"{r['source_stem']}.pkl").is_file()]
    print(f"{len(rows)} clips ({','.join(splits)}), {len(missing)} without a source pickle")
    if args.dry_run:
        return 0
    if not args.resume and not args.overwrite:
        existing = sum((root / r["pose_relpath"]).is_file() for r in rows)
        if existing:
            raise ValueError(f"{existing} caches already exist; pass --resume (skip valid) or --overwrite")
    index = root / f"poses_index_{args.split}.jsonl"
    counts = {}
    tasks = [(r, str(src), str(root), args.confidence_threshold, args.overwrite) for r in rows]
    with ProcessPoolExecutor(args.workers) as ex:
        for i, rec in enumerate(ex.map(convert_one, tasks, chunksize=32)):
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            if rec["status"] != "exists":
                append_jsonl(index, {**rec, "time": now()})
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(rows)} {counts}", flush=True)
    summary = {"time": now(), "split": args.split, "n": len(rows), "counts": counts, "adapter": args.format,
               "source_dir": str(src), "confidence_threshold": args.confidence_threshold,
               "dataset_version": read_json(root / "dataset_version.json").get("expected_version")}
    write_json(root / f"pose_import_summary_{args.split}.json", summary)
    print(summary)
    bad = sum(v for k, v in counts.items() if k not in ("ok", "exists"))
    return 1 if bad and bad == len(rows) else 0


if __name__ == "__main__":
    run(main)
