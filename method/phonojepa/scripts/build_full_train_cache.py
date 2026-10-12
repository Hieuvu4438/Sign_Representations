"""Build the PhonoJEPA cache from the complete official ASL Citizen train split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pj.data.cache import load_clip
from pj.paths import ASLC_ROOT, ASLC_SPLITS, OBS_META


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_rows(split):
    with open(ASLC_SPLITS / f"{split}.csv", newline="") as stream:
        return list(csv.DictReader(stream))


def prepare(task):
    sample_id, video, known_meta = task
    if known_meta is None:
        import cv2
        cv2.setNumThreads(1)
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        known_meta = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      float(cap.get(cv2.CAP_PROP_FPS)),
                      int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
        cap.release()
    width, height, fps, _ = known_meta
    if width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video metadata: {sample_id}: {known_meta}")
    xy, conf = load_clip(sample_id, width, height)
    if len(xy) == 0:
        raise ValueError(f"Empty keypoints: {sample_id}")
    return sample_id, xy, conf.astype(np.float16), known_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = split_rows("train")
    ids = [Path(row["Video file"]).stem for row in rows]
    assert len(ids) == len(set(ids)) == 40154, "Unexpected official train split"
    for split in ("val", "test"):
        other = {Path(row["Video file"]).stem for row in split_rows(split)}
        assert not set(ids) & other, f"Train overlap with {split}"
    known = json.loads(OBS_META.read_text())
    tasks = [(sample_id, str(ASLC_ROOT / "videos" / row["Video file"]), known.get(sample_id))
             for sample_id, row in zip(ids, rows)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    xs, cs, offsets, fps, wh, meta = [], [], [0], [], [], {}
    with mp.get_context("fork").Pool(args.workers) as pool:
        for count, (sample_id, xy, conf, info) in enumerate(pool.imap(prepare, tasks, chunksize=8), 1):
            assert sample_id == ids[count - 1]
            xs.append(xy); cs.append(conf); offsets.append(offsets[-1] + len(xy))
            fps.append(info[2]); wh.append(info[:2]); meta[sample_id] = info
            if count % 1000 == 0 or count == len(ids):
                print(f"cache {count}/{len(ids)} clips, {offsets[-1]} frames, "
                      f"{time.monotonic() - started:.1f}s", flush=True)
    xy = np.concatenate(xs); del xs
    conf = np.concatenate(cs); del cs
    temporary = args.out.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez(stream, xy=xy, conf=conf, offsets=np.array(offsets),
                 fps=np.array(fps, np.float64), wh=np.array(wh), ids=np.array(ids))
    del xy, conf
    with np.load(temporary, allow_pickle=False) as cache:
        assert cache["ids"].tolist() == ids
        assert np.all(np.diff(cache["offsets"]) > 0)
        assert cache["xy"].shape == (offsets[-1], 103, 2)
        assert cache["conf"].shape == (offsets[-1], 103)
    temporary.replace(args.out)
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta) + "\n")
    corpus = args.out.with_suffix(".corpus.jsonl")
    with corpus.open("w") as stream:
        for sample_id, video, _ in tasks:
            stream.write(json.dumps({"sample_id": sample_id, "video": video}) + "\n")
    manifest = {"split": "official train only", "n_clips": len(ids), "n_frames": offsets[-1],
                "train_csv_sha256": sha256(ASLC_SPLITS / "train.csv"),
                "cache_sha256": sha256(args.out), "cache_bytes": args.out.stat().st_size,
                "preprocessing": "pj.data.cache.load_clip, unchanged from ASLC002",
                "labels_in_cache": False, "train_val_test_ids_disjoint": True,
                "seconds": round(time.monotonic() - started, 2)}
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
