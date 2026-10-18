"""Atomic, resumable parallel pose extraction for an immutable full-corpus job."""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from signrep_baseline.job_support import (atomic_json, read_json, progress,
    install_stop_handlers, stop_requested, terminate_pool)
from signrep_baseline.cache import sha256

TEACHER = None


def initialize(weights):
    global TEACHER
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sys.path.insert(0, str(ROOT / "scripts"))
    from pose_worker import Teacher
    TEACHER = Teacher(weights)


def inspect_cache(raw, row, teacher_weights):
    import numpy as np
    with np.load(raw, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        flags = data["body_detected"].copy()
        assert meta["schema"] == "signrep_teacher_v1"
        assert meta["body_root_mode"] == "canonical"
        assert meta["video_sha256"] == sha256(row["video"])
        assert meta["weights"] == teacher_weights
        assert np.array_equal(data["frame_indices"], np.arange(meta["decoded_frames"]))
    valid = int((np.convolve(flags.astype(int), np.ones(16, int), mode="valid") == 16).sum()) if len(flags) >= 16 else 0
    return {"video_id": row["video_id"], "decoded_frames": len(flags),
            "valid_body_windows": valid, "raw": str(raw),
            "status": "PASS" if valid else "NO_VALID_WINDOW"}


def one(task):
    index, row, destination = task
    from pose_worker import extract
    raw = Path(destination) / f"{index:08d}.npz"
    temporary = raw.with_name(raw.stem + ".partial.npz")
    started = time.monotonic()
    if not raw.exists():
        temporary.unlink(missing_ok=True)
        try:
            extract(TEACHER, row["video"], temporary)
            with temporary.open("rb") as file:
                os.fsync(file.fileno())
            os.replace(temporary, raw)
        finally:
            temporary.unlink(missing_ok=True)
    result = inspect_cache(raw, row, TEACHER.metadata()["weights"])
    result.update(index=index, seconds=time.monotonic() - started)
    return result


def run(job):
    install_stop_handlers()
    opts = read_json(job / "options.json")
    source = [json.loads(s) for s in (job / "source.jsonl").read_text().splitlines() if s]
    output = job / "corpus/raw"
    receipts = job / "pose_receipts"
    output.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    expected = {p.name: sha256(p) for p in Path(opts["teacher_weights"]).glob("*.ckpt")}
    completed, results = set(), {}
    for receipt in receipts.glob("*.json"):
        value = read_json(receipt)
        index = value["index"]
        if not 0 <= index < len(source) or value["video_id"] != source[index]["video_id"]:
            raise ValueError("Pose receipt differs from frozen corpus membership")
        raw = output / f"{index:08d}.npz"
        if raw.is_file() and sha256(raw) == value["raw_sha256"]:
            completed.add(index)
            results[index] = value
    progress(job, "POSE", completed=len(completed), total=len(source), workers=opts["pose_workers"])
    pending = iter((i, row, str(output)) for i, row in enumerate(source) if i not in completed)
    pool = ProcessPoolExecutor(max_workers=opts["pose_workers"],
           mp_context=multiprocessing.get_context("spawn"), initializer=initialize,
           initargs=(opts["teacher_weights"],))
    futures = {}
    started = time.monotonic()
    last_report = started
    initial = len(completed)
    errors = []
    try:
        while True:
            while len(futures) < opts["pose_workers"] * 2 and not stop_requested():
                try:
                    task = next(pending)
                except StopIteration:
                    break
                futures[pool.submit(one, task)] = task
            if stop_requested():
                progress(job, "POSE_STOPPED", completed=len(completed), total=len(source),
                         detail="Completed atomic caches will be reused on restart")
                return 75
            if not futures:
                break
            done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                try:
                    value = future.result()
                    value["raw_sha256"] = sha256(value["raw"])
                    atomic_json(receipts / f"{value['index']:08d}.json", value)
                    completed.add(value["index"])
                    results[value["index"]] = value
                except Exception as error:
                    value = {"index": task[0], "video_id": task[1]["video_id"],
                             "status": "ERROR", "error": repr(error)}
                    errors.append(value)
                    atomic_json(job / "pose_errors.json", errors)
                    progress(job, "POSE_ERROR", **value)
            now = time.monotonic()
            if done or now - last_report >= 30:
                fresh = len(completed) - initial
                rate = fresh / max(now - started, 1)
                progress(job, "POSE", completed=len(completed), total=len(source),
                         fresh_this_launch=fresh, elapsed_hours=round((now - started) / 3600, 3),
                         eta_hours=round((len(source) - len(completed)) / max(rate, 1e-9) / 3600, 2) if fresh else "estimating",
                         errors=len(errors))
                last_report = now
        ordered = [results[i] for i in sorted(results)]
        invalid = [v for v in ordered if v["status"] != "PASS"]
        atomic_json(job / "pose_report.json", {"results": ordered,
                    "all_usable": not invalid and not errors and len(ordered) == len(source),
                    "invalid_videos": invalid, "errors": errors,
                    "seconds_this_launch": time.monotonic() - started})
        if invalid or errors or len(ordered) != len(source):
            progress(job, "POSE_REVIEW_REQUIRED", invalid=len(invalid), errors=len(errors),
                     detail="Full official membership preserved; inspect pose_report.json before training")
            return 78
        return 0
    finally:
        terminate_pool(pool)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    sys.exit(run(args.job.resolve()))
