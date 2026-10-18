"""Operational helpers; model code and the scientific cache schema stay unchanged."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

from signpath.cache.hashing import file_hash
from signpath.cache.index import index_lock, sample_directory
from signpath.cli import write_json
from signpath.data.manifest import read_manifest
from signpath.training.interruptions import raise_if_requested
from write_progress_txt import write_snapshot


@contextmanager
def progress_reporter(jobs, interval=60):
    """Log both jobs in this process; no third Python monitor is needed."""
    stopped = threading.Event()
    jobs = tuple(dict.fromkeys(Path(job).resolve() for job in jobs))

    def publish():
        for job in jobs:
            try:
                write_snapshot(job)
            except Exception as error:
                print(f"TXT logging advisory for {job}: {error}", flush=True)

    def loop():
        while not stopped.is_set():
            publish()
            stopped.wait(interval)

    thread = threading.Thread(target=loop, name="progress-txt", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()
        publish()


def wait_for_gpu_dependency(job, *, interval=30):
    """Wait through interruptions; release GPU training after the other job ends."""
    job = Path(job)
    begin = time.monotonic()
    while True:
        raise_if_requested()
        state = json.loads((job / "status.json").read_text())
        active = True
        with (job / ".job.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                active = False
                fcntl.flock(lock, fcntl.LOCK_UN)
        if not active and state["status"] in {"completed", "failed", "stopped_by_user"}:
            report = {"dependency_job": str(job.resolve()), "dependency_status": state["status"],
                "elapsed_seconds": time.monotonic() - begin, "gpu_training_released": True}
            print(f"GPU dependency released: {report}", flush=True)
            return report
        print(f"Waiting for GPU training: dependency {state['status']}/{state['stage']}", flush=True)
        # Short slices ensure SIGTERM is handled while waiting, without invoking
        # subprocesses or a second GPU model.
        until = time.monotonic() + interval
        while time.monotonic() < until:
            raise_if_requested()
            time.sleep(min(1, max(0, until - time.monotonic())))


def reuse_completed_cache(resolved, split, source_roots):
    """Adopt immutable, checksum-verified caches only for identical official records."""
    root = resolved.path(resolved.values["cache"]["root"])
    manifest = resolved.path(resolved.values["data"][f"{split}_manifest"])
    records = read_manifest(manifest)
    added, restored, linked_bytes = 0, 0, 0
    with index_lock(root):
        index = json.loads((root / "index.json").read_text())
        if index.get("fixture_only"):
            raise ValueError("Full job refuses fixture cache")
        identity = index["identity"]
        sources = []
        for source_root in source_roots:
            source_root = Path(source_root).resolve()
            source_index = json.loads((source_root / "index.json").read_text())
            if source_index["identity"] != identity or source_index.get("fixture_only"):
                raise ValueError("Cache reuse source has incompatible extraction identity")
            sources.append((source_root, source_index))
        for record in records:
            raise_if_requested()
            if record.sample_id in index["samples"]:
                continue
            destination = sample_directory(root, identity["extractor_hash"], record.sample_id)
            source = destination if destination.is_dir() else None
            for source_root, source_index in sources:
                if source is not None:
                    break
                entry = source_index["samples"].get(record.sample_id)
                if entry is None:
                    continue
                expected = (record.source_sha256, record.start_s, record.end_s, split)
                if (entry["source_sha256"], entry["start_s"], entry["end_s"], entry["split"]) != expected:
                    raise ValueError("Cache reuse record source/interval/split mismatch")
                candidate = (source_root / entry["path"]).resolve()
                if not candidate.is_relative_to(source_root) or file_hash(candidate / "meta.json") != entry["meta_sha256"]:
                    raise ValueError("Cache reuse source metadata/path integrity failure")
                source = candidate
            if source is None:
                continue
            meta = json.loads((source / "meta.json").read_text())
            if any(meta.get(key) != value for key, value in identity.items()):
                raise ValueError("Cache reuse bundle identity mismatch")
            if (meta["sample_id"], meta["source_sha256"], meta["split"], meta["source_interval_s"]) != (
                record.sample_id, record.source_sha256, split, [record.start_s, record.end_s]):
                raise ValueError("Cache reuse bundle source/interval/split mismatch")
            if meta.get("selection_target_fps") != resolved.values["data"]["target_fps"] or meta["backbone"].get("cache_execution_batch_size", 16) != 16:
                raise ValueError("Cache reuse sampling/batch recipe mismatch")
            if meta.get("storage_format") != "npz_compressed_v1" or set(meta["array_sha256"]) != {"features.npz", "raw_supports.npz"}:
                raise ValueError("Cache reuse storage/checksum schema mismatch")
            for name, digest in meta["array_sha256"].items():
                if file_hash(source / name) != digest:
                    raise ValueError("Cache reuse array checksum failure")
            if source != destination:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(tempfile.mkdtemp(prefix=".reuse-", dir=destination.parent))
                try:
                    for name in ("meta.json", "features.npz", "raw_supports.npz"):
                        os.link(source / name, temporary / name)
                        linked_bytes += (source / name).stat().st_size
                    temporary.rename(destination)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
                added += 1
            else:
                restored += 1
            index["samples"][record.sample_id] = {"path": str(destination.relative_to(root)),
                "source_sha256": record.source_sha256, "meta_sha256": file_hash(destination / "meta.json"),
                "split": split, "start_s": record.start_s, "end_s": record.end_s}
        write_json(root / "index.json", index)
    return {"status": "completed", "split": split, "hardlinked_samples": added,
        "restored_unindexed_samples": restored, "reused_bytes_without_copy": linked_bytes,
        "sources": [str(Path(p).resolve()) for p in source_roots],
        "checks": "source SHA256/interval/split/extractor/FPS/batch/content checksums; no feature regeneration"}
