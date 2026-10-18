"""Protect operational cache reuse and GPU scheduling without spawning processes."""
import fcntl
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import full_job_service_support as support
from signpath.cache.hashing import file_hash
from signpath.cache.index import sample_directory
from signpath.training.interruptions import JobInterrupted


def cache_fixture(tmp_path):
    root, source = tmp_path / "destination", tmp_path / "source"
    root.mkdir()
    source.mkdir()
    identity = {"extractor_hash": "extractor", "feature_dim": 384}
    record = SimpleNamespace(sample_id="clip", source_sha256="video-sha", start_s=0., end_s=None)
    directory = sample_directory(source, "extractor", "clip")
    directory.mkdir(parents=True)
    (directory / "features.npz").write_bytes(b"already validated feature payload")
    (directory / "raw_supports.npz").write_bytes(b"already validated support payload")
    meta = {**identity, "sample_id": "clip", "source_sha256": "video-sha", "split": "train",
        "source_interval_s": [0., None], "selection_target_fps": 25., "backbone": {},
        "storage_format": "npz_compressed_v1", "array_sha256": {
            name: file_hash(directory / name) for name in ("features.npz", "raw_supports.npz")}}
    (directory / "meta.json").write_text(json.dumps(meta))
    entry = {"path": str(directory.relative_to(source)), "meta_sha256": file_hash(directory / "meta.json"),
        "source_sha256": "video-sha", "start_s": 0., "end_s": None, "split": "train"}
    for location, samples in ((root, {}), (source, {"clip": entry})):
        (location / "index.json").write_text(json.dumps({"schema_version": 1, "identity": identity, "samples": samples}))
    resolved = SimpleNamespace(values={"cache": {"root": str(root)}, "data": {
        "train_manifest": str(tmp_path / "manifest.jsonl"), "target_fps": 25.}}, path=lambda p: Path(p))
    return resolved, root, source, directory, record


def test_reuse_hardlinks_unchanged_content_and_recovers_unpublished_index(tmp_path, monkeypatch):
    resolved, root, source, directory, record = cache_fixture(tmp_path)
    monkeypatch.setattr(support, "read_manifest", lambda _: [record])
    report = support.reuse_completed_cache(resolved, "train", [source])
    assert report["hardlinked_samples"] == 1
    destination = sample_directory(root, "extractor", "clip")
    for name in ("features.npz", "raw_supports.npz", "meta.json"):
        assert (directory / name).stat().st_ino == (destination / name).stat().st_ino
    index = json.loads((root / "index.json").read_text())
    index["samples"] = {}
    (root / "index.json").write_text(json.dumps(index))
    assert support.reuse_completed_cache(resolved, "train", [source])["restored_unindexed_samples"] == 1


@pytest.mark.parametrize("bad", ["payload", "source", "split", "identity"])
def test_reuse_rejects_corruption_and_wrong_membership(tmp_path, monkeypatch, bad):
    resolved, root, source, directory, record = cache_fixture(tmp_path)
    monkeypatch.setattr(support, "read_manifest", lambda _: [record])
    if bad == "payload":
        (directory / "features.npz").write_bytes(b"corrupted")
    else:
        index = json.loads((source / "index.json").read_text())
        if bad == "identity":
            index["identity"]["feature_dim"] = 999
        else:
            index["samples"]["clip"]["source_sha256" if bad == "source" else "split"] = "wrong"
        (source / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        support.reuse_completed_cache(resolved, "train", [source])
    assert not json.loads((root / "index.json").read_text())["samples"]


def test_gpu_wait_checks_lock_and_terminal_state_without_another_process(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"status": "completed", "stage": "done"}))
    with (tmp_path / ".job.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        released = threading.Event()

        def release():
            released.wait(.05)
            fcntl.flock(held, fcntl.LOCK_UN)

        thread = threading.Thread(target=release)
        thread.start()
        try:
            report = support.wait_for_gpu_dependency(tmp_path, interval=.01)
            assert report["gpu_training_released"] and report["elapsed_seconds"] >= .04
        finally:
            released.set()
            thread.join()


def test_gpu_wait_honors_safe_interruption(tmp_path, monkeypatch):
    def interrupted():
        raise JobInterrupted("test interruption")
    monkeypatch.setattr(support, "raise_if_requested", interrupted)
    with pytest.raises(JobInterrupted):
        support.wait_for_gpu_dependency(tmp_path)


def test_reporter_writes_both_jobs_and_preserves_final_status(tmp_path):
    jobs = [tmp_path / "small", tmp_path / "full"]
    for job in jobs:
        job.mkdir()
        (job / "status.json").write_text(json.dumps({"status": "running", "stage": "cache_train"}))
    with support.progress_reporter(jobs, interval=.01):
        (jobs[1] / "status.json").write_text(json.dumps({"status": "completed", "stage": "done"}))
    assert "completed" in (jobs[1] / "status_latest.txt").read_text()
    assert "running" in (jobs[0] / "progress.txt").read_text()
