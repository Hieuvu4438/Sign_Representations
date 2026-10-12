#!/usr/bin/env python3
"""Resumable, audited Uni-Sign feature extraction from ASL Citizen pose caches."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def destination(output, name):
    shard = hashlib.sha256(name.encode()).hexdigest()[:2]
    return Path(output) / "features" / shard / (name + ".npz")


def validate_feature(path, fingerprint, name, source=None):
    with np.load(path, allow_pickle=False) as data:
        assert str(data["run_fingerprint"]) == fingerprint, "Different pipeline fingerprint"
        assert str(data["sample_id"]) == name, "Different sample id"
        embeddings, pooled = data["pose_embeddings"], data["pooled_embedding"]
        indices, frame_ids = data["sampled_pose_indices"], data["source_frame_indices"]
        selected = data["selected_person_indices"]
        n, kept = int(data["source_frames"]), int(data["retained_frames"])
        assert embeddings.dtype == np.float32 and embeddings.shape == (len(indices), 768)
        assert pooled.dtype == np.float32 and pooled.shape == (768,)
        assert np.isfinite(embeddings).all() and np.isfinite(pooled).all()
        assert selected.shape == (n,) and 0 < kept <= n
        assert len(indices) > 0 and frame_ids.shape == indices.shape
        assert np.all(np.diff(indices) > 0) and np.all(np.diff(frame_ids) > 0)
        retained = np.flatnonzero(selected >= 0)
        assert len(retained) == kept and indices.min() >= 0 and indices.max() < kept
        assert np.array_equal(frame_ids, retained[indices])
        np.testing.assert_allclose(pooled, embeddings.mean(0), rtol=1e-5, atol=1e-6)
        if source is not None:
            stat = Path(source).stat()
            assert int(data["source_size_bytes"]) == stat.st_size
            assert int(data["source_mtime_ns"]) == stat.st_mtime_ns
            assert str(data["source_sha256"]) == sha256_file(source), "Source cache changed"
        return {
            "sample_id": name, "feature_frames": len(indices), "source_frames": n,
            "retained_frames": kept, "multi_person_frames": int(data["multi_person_frames"]),
            "bytes": Path(path).stat().st_size,
        }


def prepare(task):
    """CPU worker: read cache, choose a signer, preserve frame mapping, preprocess."""
    source, video, target, fingerprint, name, max_length = task
    try:
        if Path(target).exists():
            try:
                info = validate_feature(target, fingerprint, name, source)
                return {"existing": True, "info": info}
            except Exception:
                pass  # A corrupt/incomplete output is regenerated atomically.
        import torch
        torch.set_num_threads(1)
        from pose_preprocessing import preprocess_pose_sequence, uniform_frame_indices
        from serving.pose_runner import PrimaryPersonTracker

        raw_bytes = Path(source).read_bytes()
        raw = pickle.loads(raw_bytes)
        keypoints, scores = raw["keypoints"], raw["scores"]
        n = len(scores)
        if n == 0 or len(keypoints) != n:
            raise ValueError("Empty pose or inconsistent frame counts")
        people = []
        for k, s in zip(keypoints, scores):
            if k.ndim != 3 or k.shape[1:] != (133, 2) or s.shape != k.shape[:2]:
                raise ValueError("Unexpected whole-body array shape")
            if not np.isfinite(k).all() or not np.isfinite(s).all():
                raise ValueError("Non-finite source coordinates or scores")
            people.append(len(k))
        selected = np.full(n, -1, dtype=np.int16)
        pose = {"keypoints": [], "scores": []}
        if all(count == 1 for count in people):
            # One candidate: this is the same acceptance rule as the tracker.
            for index, (k, s) in enumerate(zip(keypoints, scores)):
                if np.count_nonzero(s[0] >= 0.3) >= 4:
                    selected[index] = 0
                    pose["keypoints"].append(np.asarray(k, dtype=np.float32))
                    pose["scores"].append(np.asarray(s, dtype=np.float32))
        else:
            import cv2
            cv2.setNumThreads(1)
            capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
            width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            capture.release()
            if width <= 0 or height <= 0:
                raise ValueError(f"Cannot read dimensions for multi-person cache: {video}")
            scale = np.array([width, height], dtype=np.float32)
            tracker = PrimaryPersonTracker()
            for index, (k, s) in enumerate(zip(keypoints, scores)):
                pixels = k * scale
                chosen_k, chosen_s, _ = tracker.select(pixels, s, width, height)
                if chosen_k is not None:
                    person = int(np.flatnonzero(np.all(pixels == chosen_k, axis=(1, 2)))[0])
                    selected[index] = person
                    pose["keypoints"].append((chosen_k.astype(np.float32) / scale)[None])
                    pose["scores"].append(chosen_s.astype(np.float32)[None])
        if not pose["scores"]:
            raise ValueError("No frame has a valid signer; no placeholder features will be written")
        indices = uniform_frame_indices(len(pose["scores"]), max_length)
        processed = preprocess_pose_sequence(pose, max_length)
        stat = Path(source).stat()
        return {
            "name": name, "target": target, "source": source,
            "parts": {key: processed[key][0].numpy() for key in ("body", "left", "right", "face_all")},
            "indices": indices, "source_frame_indices": np.flatnonzero(selected >= 0)[indices],
            "selected": selected, "source_frames": n, "retained_frames": len(pose["scores"]),
            "multi_person_frames": sum(count > 1 for count in people),
            "source_size_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
            "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }
    except Exception:
        return {"error": traceback.format_exc(), "name": name, "source": source}


def write_feature(item, embeddings, fingerprint):
    target = Path(item["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, pose_embeddings=embeddings, pooled_embedding=embeddings.mean(0),
            sampled_pose_indices=item["indices"], source_frame_indices=item["source_frame_indices"],
            selected_person_indices=item["selected"], source_frames=np.int64(item["source_frames"]),
            retained_frames=np.int64(item["retained_frames"]),
            multi_person_frames=np.int64(item["multi_person_frames"]),
            source_size_bytes=np.int64(item["source_size_bytes"]),
            source_mtime_ns=np.int64(item["source_mtime_ns"]), source_sha256=item["source_sha256"],
            sample_id=item["name"], run_fingerprint=fingerprint,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    # Every write is read back and checked, in addition to the final full audit.
    return validate_feature(target, fingerprint, item["name"])


def read_samples(dataset):
    samples = []
    names = set()
    for split in ("train", "val", "test"):
        with (dataset / "splits" / (split + ".csv")).open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                name = Path(row["Video file"]).stem
                if name in names:
                    raise ValueError(f"Duplicate sample across splits: {name}")
                names.add(name)
                source = dataset / "keypoints" / (name + ".pkl")
                video = dataset / "videos" / row["Video file"]
                if not source.is_file() or not video.is_file():
                    raise FileNotFoundError(f"Missing source for {name}")
                samples.append({"sample_id": name, "split": split, "gloss": row["Gloss"],
                                "participant_id": row["Participant ID"], "source": str(source), "video": str(video)})
    cached = {path.stem for path in (dataset / "keypoints").glob("*.pkl")}
    if cached != names:
        raise ValueError(f"Splits/cache mismatch: {len(cached - names)} extra, {len(names - cached)} missing")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/dongvk/datasets/ASL_Citizen"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "pretrained_weight/unisign/wlasl_pose_only_islr.pth")
    parser.add_argument("--mt5-path", type=Path, default=ROOT / "pretrained_weight/mt5-base")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--writers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Pilot only; zero covers all splits")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if min(args.max_length, args.batch_size, args.workers, args.writers) < 1:
        parser.error("Lengths and worker counts must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "job.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    samples = read_samples(args.dataset_root.resolve())
    if args.limit:
        # Spread pilots over all splits rather than testing only the first signer.
        chosen = np.linspace(0, len(samples) - 1, num=min(args.limit, len(samples)), dtype=int)
        samples = [samples[index] for index in chosen]
    config = {
        "schema": "unisign-pose-features-v1", "dataset_root": str(args.dataset_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
        "mt5_path": str(args.mt5_path.resolve()), "max_length": args.max_length,
        "dtype": "float32", "feature_stage": "pose_proj_before_mt5_without_prompt",
        "cuda_math": "TF32 disabled; deterministic cuDNN; no autocast",
        "selection": "serving.PrimaryPersonTracker; confidence>=0.3; >=4 valid keypoints",
        "expected_samples": len(samples), "limit": args.limit,
        "code_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in [
            Path(__file__), ROOT / "models.py", ROOT / "pose_preprocessing.py", ROOT / "serving/pose_runner.py",
            ROOT / "stgcn_layers/stgcn_block.py", ROOT / "stgcn_layers/gcn_utils.py",
        ]},
        "splits_sha256": {split: sha256_file(args.dataset_root / "splits" / (split + ".csv")) for split in ("train", "val", "test")},
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["fingerprint"] = fingerprint
    config_path = output / "run.json"
    if config_path.exists() and json.loads(config_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Output belongs to a different pipeline. Use a new directory.")
    atomic_json(config_path, config)
    with (output / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for sample in samples:
            sample["feature_path"] = str(destination(output, sample["sample_id"]).relative_to(output))
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    started = time.time()
    completed = 0
    output_bytes = 0
    failures = []
    last_status = 0.0

    def status(stage, force=False, **extra):
        nonlocal last_status
        now = time.time()
        if not force and now - last_status < 5:
            return
        elapsed = now - started
        state = {"stage": stage, "pid": os.getpid(), "updated_at_unix": now, "expected": len(samples),
                 "completed": completed, "failures": len(failures), "output_bytes": output_bytes,
                 "elapsed_seconds": round(elapsed, 1), "clips_per_second": round(completed / max(elapsed, 0.001), 3),
                 "run_fingerprint": fingerprint, **extra}
        if completed:
            state["eta_seconds"] = round((len(samples) - completed) * elapsed / completed)
        atomic_json(output / "status.json", state)
        print(json.dumps(state), flush=True)
        last_status = now

    def record(info):
        nonlocal completed, output_bytes
        completed += 1
        output_bytes += info["bytes"]
        with (output / "completed.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(info, ensure_ascii=False) + "\n")
        status("extracting")

    if not args.verify_only:
        import torch
        import importlib.metadata as metadata
        from models import Uni_Sign
        if not torch.cuda.is_available():
            raise RuntimeError("GPU inference requires CUDA")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model_args = SimpleNamespace(hidden_dim=256, dataset="WLASL", task="ISLR", rgb_support=False,
                                     label_smoothing=0.0, mt5_path=str(args.mt5_path.resolve()))
        model = Uni_Sign(model_args)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
        del checkpoint
        model.eval().cuda()
        atomic_json(output / "runtime.json", {
            "python": sys.version, "executable": sys.executable, "gpu": torch.cuda.get_device_name(0),
            "packages": {p: metadata.version(p) for p in ("torch", "torchvision", "numpy", "transformers", "einops")},
            "batch_size": args.batch_size, "workers": args.workers, "writers": args.writers,
        })
        tasks = [(sample["source"], sample["video"], str(destination(output, sample["sample_id"])),
                  fingerprint, sample["sample_id"], args.max_length) for sample in samples]
        buckets = defaultdict(list)
        writes = []
        status("extracting", True)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool, \
                ThreadPoolExecutor(max_workers=args.writers) as writers:

            def finish_writes(all_pending=False):
                nonlocal writes
                remaining = []
                for future in writes:
                    if all_pending or future.done():
                        record(future.result())
                    else:
                        remaining.append(future)
                writes = remaining
                if len(writes) > args.batch_size * 2:
                    record(writes.pop(0).result())

            def encode(items):
                lengths = len(items[0]["indices"])
                source = {part: torch.from_numpy(np.stack([item["parts"][part] for item in items])).cuda()
                          for part in ("body", "left", "right", "face_all")}
                source["attention_mask"] = torch.ones((len(items), lengths), dtype=torch.long, device="cuda")
                with torch.inference_mode():
                    try:
                        encoded = model.encode(source)["inputs_embeds"][:, -lengths:].float().cpu().numpy()
                    except torch.cuda.OutOfMemoryError:
                        del source
                        torch.cuda.empty_cache()
                        if len(items) == 1:
                            raise
                        middle = len(items) // 2
                        encode(items[:middle])
                        encode(items[middle:])
                        return
                if not np.isfinite(encoded).all():
                    raise RuntimeError("Encoder returned non-finite values")
                for item, features in zip(items, encoded):
                    writes.append(writers.submit(write_feature, item, features, fingerprint))
                finish_writes()

            # Keep a bounded number of CPU tasks in flight; retries are per file.
            iterator = iter(tasks)
            pending = {}
            for _ in range(args.workers * 4):
                task = next(iterator, None)
                if task is not None:
                    pending[pool.submit(prepare, task)] = (task, 0)
            while pending:
                future = next(as_completed(pending))
                task, attempt = pending.pop(future)
                item = future.result()
                if "error" in item:
                    with (output / "errors.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({**item, "attempt": attempt + 1}) + "\n")
                    if attempt < 2:
                        pending[pool.submit(prepare, task)] = (task, attempt + 1)
                    else:
                        failures.append(item)
                elif item.get("existing"):
                    record(item["info"])
                else:
                    length = len(item["indices"])
                    buckets[length].append(item)
                    if len(buckets[length]) >= args.batch_size:
                        encode(buckets.pop(length))
                following = next(iterator, None)
                if following is not None:
                    pending[pool.submit(prepare, following)] = (following, 0)
                finish_writes()
                status("extracting")
            for items in buckets.values():
                if items:
                    encode(items)
            finish_writes(True)
        del model
        torch.cuda.empty_cache()

    # Completion is based on the expected source set, never on the log counter.
    status("verifying", True, validated=0)
    verified = 0
    verification_errors = []
    split_counts = Counter()
    measured_bytes = 0
    total_feature_frames = 0

    def verify(sample):
        try:
            info = validate_feature(destination(output, sample["sample_id"]), fingerprint, sample["sample_id"], sample["source"])
            return sample, info, None
        except Exception:
            return sample, None, traceback.format_exc()

    with ThreadPoolExecutor(max_workers=args.workers) as verifier:
        for sample, info, error in verifier.map(verify, samples):
            if error:
                verification_errors.append({"sample_id": sample["sample_id"], "error": error})
            else:
                verified += 1
                split_counts[sample["split"]] += 1
                measured_bytes += info["bytes"]
                total_feature_frames += info["feature_frames"]
            status("verifying", validated=verified, verification_failures=len(verification_errors))
    actual_names = {path.stem for path in (output / "features").glob("*/*.npz")}
    expected_names = {sample["sample_id"] for sample in samples}
    report = {
        "success": verified == len(samples) and not verification_errors and actual_names == expected_names,
        "expected": len(samples), "validated": verified, "split_counts": dict(split_counts),
        "bytes": measured_bytes, "GB": measured_bytes / 1e9, "feature_frames": total_feature_frames,
        "extra_outputs": sorted(actual_names - expected_names), "missing_outputs": sorted(expected_names - actual_names),
        "verification_errors": verification_errors, "processing_failures": failures,
        "elapsed_seconds": time.time() - started, "fingerprint": fingerprint,
    }
    atomic_json(output / "verification.json", report)
    completed, output_bytes = verified, measured_bytes
    status("complete" if report["success"] else "failed", True, validated=verified,
           verification_failures=len(verification_errors))
    if not report["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
