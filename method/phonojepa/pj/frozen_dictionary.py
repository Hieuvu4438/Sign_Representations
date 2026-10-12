"""Frozen train-gallery/test-query retrieval for ASL Citizen and WLASL2000.

SignRep Sec. 6/7.2 retrieval: raw mean-window pooling, video L2, maximum
gallery cosine per gloss, DCG/Rec@1/Rec@5. PhonoJEPA keeps its native 64-frame
skeleton input (rather than SignRep's 16 RGB frames); sliding stride is 2.
No fitting, PCA, classifier, augmentation, activity weighting or test selection.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pj.data.cache import load_clip
from pj.data import window as W
from pj.extract import Extractor
from pj.paths import ASLC_ROOT, OBS_JEPA

sys.path.insert(0, str(OBS_JEPA))
from oj.data.clips import video_meta
from oj.wlasl_retrieval import DATA as WLASL_ROOT, protocol as locked_wlasl, read_rows, score_retrieval, sha256

READOUTS = ("_grp", "_cls", "_mean")
T = 64


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def make_manifests(dataset):
    if dataset == "wlasl":
        previous = OBS_JEPA / "runtime/wlasl2000_frozen"
        locked_wlasl(previous)  # verify the existing immutable manifest hashes
        rows = {s: read_rows(previous / f"{s}.jsonl") for s in ("train", "test")}
        source = {"annotation_sha256": sha256(WLASL_ROOT / "preprocess/nslt_2000.json"),
                  "classlist_sha256": sha256(WLASL_ROOT / "preprocess/wlasl_class_list.txt"),
                  "existing_locked_protocol": str(previous / "protocol.json")}
        kp_root, n_classes, counts = WLASL_ROOT / "keypoint", 2000, (14296, 2879)
    else:
        raw = {}
        for split in ("train", "test"):
            with open(ASLC_ROOT / "splits" / f"{split}.csv", newline="") as stream:
                raw[split] = list(csv.DictReader(stream))
        labels = {g: i for i, g in enumerate(sorted({r["Gloss"] for r in raw["train"]}))}
        rows = {split: [dict(sample_id=Path(r["Video file"]).stem,
                            video=str(ASLC_ROOT / "videos" / r["Video file"]),
                            gloss=r["Gloss"], label=labels[r["Gloss"]], signer=r["Participant ID"],
                            split=split, role="gallery" if split == "train" else "query")
                        for r in raw[split]] for split in ("train", "test")}
        source = {f"{s}_csv_sha256": sha256(ASLC_ROOT / "splits" / f"{s}.csv") for s in rows}
        kp_root, n_classes, counts = ASLC_ROOT / "keypoints", 2731, (40154, 32941)
    assert not ({r["sample_id"] for r in rows["train"]} & {r["sample_id"] for r in rows["test"]})
    for split, count in zip(("train", "test"), counts):
        assert len(rows[split]) == count
        assert len({r["sample_id"] for r in rows[split]}) == count
        assert len({r["label"] for r in rows[split]}) == n_classes
        for row in rows[split]:
            assert Path(row["video"]).is_file(), row["video"]
            assert (kp_root / f"{row['sample_id']}.pkl").is_file(), row["sample_id"]
    return rows, kp_root, n_classes, source


def prepare(row, kp_root, stride, meta=None):
    width, height, fps, nf = meta if meta is not None else video_meta(row["video"])
    if min(width, height, nf) <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video metadata: {row['sample_id']}")
    xy, conf = load_clip(row["sample_id"], width, height, kp_dir=kp_root)
    if len(xy) != nf:
        raise ValueError(f"Keypoint/video frame mismatch: {row['sample_id']}: {len(xy)} vs {nf}")
    xy, conf, _ = W.canonicalise(xy, conf)
    eff = min(fps, W.TARGET_FPS)
    nsamp = max(1, int(np.floor(nf / fps * eff)))
    starts = list(range(0, max(1, nsamp - T + 1), stride))
    arrays = {k: [] for k in ("xy", "obs", "conf", "valid")}
    for start in starts:
        i0, i1, weights, valid = W.window_positions(nf, fps, T, start / eff, 1.0)
        x, c, _ = W.resample(xy, conf, i0, i1, weights, valid)
        arrays["xy"].append(np.nan_to_num(x))
        arrays["obs"].append(np.isfinite(x).all(-1))
        arrays["conf"].append(c)
        arrays["valid"].append(valid)
    return {k: np.stack(v) for k, v in arrays.items()}


class Prepared(torch.utils.data.Dataset):
    def __init__(self, rows, kp_root, stride, meta):
        self.rows, self.kp_root, self.stride, self.meta = rows, kp_root, stride, meta

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return prepare(row, self.kp_root, self.stride, self.meta.get(row["sample_id"]))


def worker_init(_):
    import cv2
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


@torch.inference_mode()
def encode_video(extractor, prepared, window_batch):
    count = len(prepared["xy"])
    sums = {}
    for start in range(0, count, window_batch):
        part = {k: v[start:start + window_batch] for k, v in prepared.items()}
        for key, values in extractor.encode(part).items():
            values = values.astype(np.float32)
            sums[key] = sums.get(key, 0) + values.sum(0, dtype=np.float64)
    return {key: (value / count).astype(np.float32) for key, value in sums.items()}, count


def extract_split(extractor, rows, kp_root, target, args, meta, progress):
    target.mkdir()
    dataset = Prepared(rows, kp_root, args.stride, meta)
    kwargs = dict(batch_size=None, num_workers=args.workers, worker_init_fn=worker_init)
    if args.workers:
        kwargs["prefetch_factor"] = 2
    loader = torch.utils.data.DataLoader(dataset, **kwargs)
    values = {k: [] for k in READOUTS}
    counts, shards, ids = [], [], []
    start = time.monotonic()
    total_windows = 0
    for index, prepared in enumerate(loader):
        prepared = {k: v.numpy() if torch.is_tensor(v) else v for k, v in prepared.items()}
        pooled, nwin = encode_video(extractor, prepared, args.window_batch)
        for key in values:
            values[key].append(pooled[key])
        counts.append(nwin); ids.append(rows[index]["sample_id"]); total_windows += nwin
        if len(ids) == 256 or index + 1 == len(rows):
            shard = target / f"{index + 1 - len(ids):07d}.npz"
            temporary = shard.with_suffix(".partial.npz")
            np.savez(temporary, **{k: np.stack(v) for k, v in values.items()},
                     ids=np.array(ids), nwin=np.array(counts))
            temporary.replace(shard); shards.append(shard)
            values = {k: [] for k in READOUTS}; counts, ids = [], []
            write_json(progress, {"stage": target.name, "completed": index + 1, "total": len(rows),
                                  "windows": total_windows, "seconds": round(time.monotonic() - start, 1),
                                  "updated": now(), "pid": os.getpid()})
            print(target.name, index + 1, "/", len(rows), "windows", total_windows,
                  f"{time.monotonic() - start:.1f}s", flush=True)
    arrays = {key: [] for key in READOUTS}; all_ids = []
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            all_ids.extend(data["ids"].tolist())
            for key in arrays:
                arrays[key].append(data[key])
    assert all_ids == [r["sample_id"] for r in rows]
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    assert all(np.isfinite(value).all() for value in arrays.values())
    return arrays, {"n_clips": len(rows), "n_windows": total_windows,
                    "seconds": round(time.monotonic() - start, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", default="wlasl,aslc")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--window-batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    assert args.stride == 2 and args.window_batch > 0
    datasets = args.datasets.split(",")
    assert len(datasets) == len(set(datasets)) and set(datasets) <= {"aslc", "wlasl"}
    args.out.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "pid": os.getpid(), "started": now(),
             "checkpoint": str(args.ckpt), "checkpoint_sha256": sha256(args.ckpt), "datasets": datasets,
             "material_passport": {"experiment_id": "PLctx-full160-frozen-SignRep-retrieval",
                 "mode": "run", "inputs": "Frozen final checkpoint, official ASL Citizen and local WLASL2000 train/test",
                 "outputs": "Compact pooled embeddings, per-query ranks, DCG/Rec@1/Rec@5",
                 "external_data_upload": False, "test_selection": False}}
    write_json(args.out / "status.json", state)
    try:
        extractor = Extractor("model", args.ckpt, "teacher", args.device)
        extractor.enc.eval().requires_grad_(False)
        for dataset in datasets:
            rows, kp_root, n_classes, source = make_manifests(dataset)
            target = args.out / dataset; target.mkdir()
            for split in rows:
                (target / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[split]))
            protocol = {"source": "https://arxiv.org/html/2503.08529v1#S7.SS2", "paper_table": 4,
                "checkpoint_sha256": state["checkpoint_sha256"], "weights": "teacher",
                "pretraining": "ASL Citizen official train only, 40154 clips, 160 epochs, seed 0",
                "gallery": {"split": "train", "n": len(rows["train"])},
                "query": {"split": "test", "n": len(rows["test"])}, "n_classes": n_classes,
                "frozen": True, "downstream_training": False, "pca_whitening": False,
                "window_frames": T, "stride": args.stride, "effective_fps": "min(source fps,25)",
                "primary_readout": "_grp", "secondary_readouts": ["_cls", "_mean"],
                "pooling": "uniform mean of raw window embeddings, then L2 video normalization",
                "similarity": "cosine; class score = maximum over gallery clips of that class",
                "rank": "pessimistic class ties: 1 + number of other classes scoring >= correct class",
                "metrics": {"DCG": "mean(1/log2(1+rank))", "Rec@1": "mean(rank<=1)", "Rec@5": "mean(rank<=5)"},
                "adaptations_from_RGB_SignRep": ["Native 64-frame skeleton encoder versus 16-frame RGB encoder",
                    "No trained activity head: avg pooling only", "Native missing-frame validity mask for short clips"],
                "source_hashes": source,
                "manifest_sha256": {s: sha256(target / f"{s}.jsonl") for s in rows},
                "extractor_sha256": sha256(Path(__file__)),
                "native_encoder_sha256": sha256(ROOT / "pj/model.py"),
                "scorer_sha256": sha256(OBS_JEPA / "oj/wlasl_retrieval.py")}
            write_json(target / "protocol.json", protocol)
            state.update(current_dataset=dataset); write_json(args.out / "status.json", state)
            known = {}
            if dataset == "aslc":
                known = json.loads((ROOT / "runtime/cache/aslc_train_full.meta.json").read_text())
            embeddings, extraction = {}, {}
            for split in ("train", "test"):
                state.update(current_stage=f"{dataset}_{split}"); write_json(args.out / "status.json", state)
                embeddings[split], extraction[split] = extract_split(
                    extractor, rows[split], kp_root, target / split, args, known, target / "progress.json")
            result = {"dataset": dataset, "protocol": str(target / "protocol.json"),
                      "checkpoint_sha256": state["checkpoint_sha256"], "extraction": extraction, "readouts": {}}
            for key in READOUTS:
                metrics, ranks = score_retrieval(embeddings["train"][key], embeddings["test"][key],
                    [r["label"] for r in rows["train"]], [r["label"] for r in rows["test"]], n_classes)
                assert len(ranks) == len(rows["test"])
                np.save(target / f"{key}_ranks.npy", ranks)
                result["readouts"][key] = metrics
                print(dataset, key, json.dumps(metrics), flush=True)
            assert sha256(args.ckpt) == state["checkpoint_sha256"], "Checkpoint changed during inference"
            write_json(target / "results.json", result)
            del embeddings
        state.update(status="complete", finished=now()); write_json(args.out / "status.json", state)
        print(now(), "FROZEN RETRIEVAL COMPLETE", flush=True)
    except Exception as exc:
        state.update(status="failed", error=repr(exc), finished=now()); write_json(args.out / "status.json", state)
        raise


if __name__ == "__main__":
    main()
