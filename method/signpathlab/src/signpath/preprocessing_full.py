"""Bounded CPU extraction queue, lossless disk cache and batched index publication."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time
import threading
import numpy as np

from .backbones.dino import DINOAdapter
from .cache.hashing import canonical_hash, file_hash
from .cache.index import index_lock, sample_directory
from .cache.reader import read_bundle
from .cache.schema import ARRAY_DTYPES, IDENTITY_FIELDS
from .cache.writer import write_bundle
from .cli import write_json
from .data.decode import decode_video, nearest_observations
from .pose.context import ExtractionContext
from .pose.extract import RawLandmarks
from .preprocessing import asset_plan, manifest_records, raw_identity, raw_directory, read_raw, _close_detector, _preprocess_one

_THREAD_LOCAL = threading.local()


def _initialize(plan, minimum, instances):
    from .pose.extract import LandmarkExtractor
    detector = LandmarkExtractor(*(plan["detectors"][k]["path"] for k in ("pose", "hand", "face")), min_body_visibility=minimum)
    _THREAD_LOCAL.detector = detector
    instances.append(detector)


@contextmanager
def _cpu_executor(plan, minimum, workers=1):
    instances = []
    try:
        with ThreadPoolExecutor(max_workers=workers, initializer=_initialize, initargs=(plan, minimum, instances)) as pool:
            yield pool
    finally:
        for detector in instances:
            detector.close()
        _close_detector()


def _prepare(resolved, record, identity):
    directory = raw_directory(resolved, record, identity)
    # A coordinate NPZ is published before its metadata marker. Power loss in
    # between leaves an uncommitted sidecar; retain it then regenerate from RGB.
    for sidecar in directory.glob("coordinates_*.npz"):
        if not sidecar.with_suffix(".json").exists():
            archive = directory / "recovery"
            archive.mkdir(exist_ok=True)
            sidecar.rename(archive / f"{time.time_ns()}_{sidecar.name}")
    result = _preprocess_one(resolved, record, identity, detector=_THREAD_LOCAL.detector)
    return record, result


def fingerprint(frames):
    import cv2
    bits = []
    for i in np.rint(np.linspace(0, len(frames) - 1, 8)).astype(int):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        low = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].flatten()[1:]
        bits.extend(low > np.median(low))
    return np.packbits(bits).tobytes().hex()


def cache_adapter(plan, device, dino_batch_size=16):
    """Batch execution is versioned when it differs from the original batch16 cache."""
    if dino_batch_size not in (16, 32, 64):
        raise ValueError("Supported measured DINO batch sizes:16,32,64")
    adapter = DINOAdapter().load(plan["dino"], device)
    if dino_batch_size != 16:
        original = adapter.encode_crops
        adapter.encode_crops = lambda images: original(images, batch_size=dino_batch_size)
        adapter._provenance["cache_execution_batch_size"] = dino_batch_size
    return adapter


def build_full_cache(resolved, split, assets_file, device, *, workers=1, dino_batch_size=16,
                     reserve_gib=6., progress_path=None, adapter=None):
    if workers not in (1, 2):
        raise ValueError("User process limit: one Python process with at most2CPU worker threads")
    import torch
    torch.set_num_threads(1)
    started = time.monotonic()
    plan = asset_plan(resolved, assets_file, include_dino=True)
    raw_hash = raw_identity(resolved, plan)
    root = resolved.path(resolved.values["cache"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    records = manifest_records(resolved, split)
    if any(r.availability != "available" for r in records):
        raise ValueError("Full-corpus run refuses missing official records; repair availability first")
    index_file = root / "index.json"
    index = json.loads(index_file.read_text()) if index_file.exists() else None
    pending = [r for r in records if index is None or r.sample_id not in index["samples"]]
    adapter = cache_adapter(plan, device, dino_batch_size) if adapter is None else adapter
    expected_batch = adapter.provenance().get("cache_execution_batch_size", 16)
    if expected_batch != dino_batch_size:
        raise ValueError("Preloaded adapter batch provenance differs from requested execution")
    if index is not None and index["samples"]:
        first = next(iter(index["samples"].values()))
        cached_meta = json.loads((root / first["path"] / "meta.json").read_text())
        if cached_meta["backbone"].get("cache_execution_batch_size", 16) != dino_batch_size:
            raise ValueError("DINO batch recipe differs from existing cache; use a separately versioned cache root")
    done = len(records) - len(pending)
    def publish():
        if index is not None:
            with index_lock(root):
                write_json(index_file, index)
    queue = deque()
    iterator = iter(pending)
    progress_path = Path(progress_path) if progress_path else root / f"progress.{split}.json"
    timings = []
    with _cpu_executor(plan, resolved.values["pose"]["min_body_visibility"], workers) as pool:
        def submit():
            try:
                record = next(iterator)
            except StopIteration:
                return False
            queue.append(pool.submit(_prepare, resolved, record, raw_hash))
            return True
        try:
            for _ in range(workers * 2):
                if not submit():
                    break
            while queue:
                from .training.interruptions import raise_if_requested
                raise_if_requested()
                if shutil.disk_usage(root).free < reserve_gib * 2 ** 30:
                    raise OSError(f"Disk reserve {reserve_gib}GiB reached; cache preserved, no automatic retry")
                record, prepared = queue.popleft().result()
                clip_started = time.monotonic()
                raw_path = raw_directory(resolved, record, raw_hash)
                raw, times, sources, raw_meta = read_raw(raw_path, record, raw_hash)
                video = decode_video(record.video_path, record.start_s, record.end_s)
                if not np.array_equal(times, video.timestamps) or not np.array_equal(sources, video.source_frame_index):
                    raise ValueError("Raw decode changed before frozen visual extraction")
                selected = nearest_observations(times, sources, resolved.values["data"]["target_fps"])
                unique, inverse = np.unique(selected.indices, return_inverse=True)
                raw_subset = RawLandmarks(**{name: getattr(raw, name)[unique] for name in
                    ("xy", "observed", "quality", "body_raw", "hand_candidates", "face_raw", "handedness_scores", "assignment")}, provenance=raw.provenance)
                context = ExtractionContext(record.sample_id, record.source_sha256, sources[unique], raw_subset,
                    {"source_mirrored": record.metadata.get("source_mirrored"), "hand_assignment": raw.provenance["hand_assignment"]},
                    min_body_visibility=resolved.values["pose"]["min_body_visibility"])
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
                visual_started = time.monotonic()
                bundle = adapter.extract(video.frames[unique], times[unique], context)
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
                visual_seconds = time.monotonic() - visual_started
                bundle = replace(bundle, **{name: getattr(bundle, name)[inverse] for name in ARRAY_DTYPES},
                    raw_supports=tuple(s for i in inverse for s in bundle.raw_supports[i * 7:(i + 1) * 7]))
                bundle.meta.update(split=split, source_interval_s=[record.start_s, record.end_s], raw_preprocessing_identity=raw_hash,
                    selection_target_fps=resolved.values["data"]["target_fps"], selected_original_array_indices=selected.indices.tolist(),
                    duplicate_selection=selected.duplicate_selection.tolist(), roi_metadata_source_frame_index=sources[unique].tolist(),
                    fingerprint_hex=fingerprint(video.frames), fingerprint_recipe="8_uniform_frames_gray32_dct8_drop_dc_median_bits",
                    extraction_entrypoint_sha256=file_hash(__file__), raw_intermediate_retention="metadata/checksums kept; source videos retained; temporary raw arrays regenerated if needed")
                bundle.meta["cache_execution"] = {"cpu_threads": workers, "python_processes": 1, "dino_batch_size": dino_batch_size}
                identity = {key: bundle.meta[key] for key in IDENTITY_FIELDS}
                directory = sample_directory(root, bundle.meta["extractor_hash"], record.sample_id)
                if not directory.exists():
                    write_bundle(directory, bundle, compressed=True)
                verified = read_bundle(directory, identity)
                if verified.meta["source_sha256"] != record.source_sha256 or verified.meta["sample_id"] != record.sample_id:
                    raise ValueError("Existing cache belongs to another source")
                if index is None:
                    index = {"schema_version": 1, "identity": identity, "samples": {}, "fixture_only": False}
                if index["identity"] != identity:
                    raise ValueError("Full cache cannot mix extractor identities")
                index["samples"][record.sample_id] = {"path": str(directory.relative_to(root)), "source_sha256": record.source_sha256,
                    "meta_sha256": file_hash(directory / "meta.json"), "split": split, "start_s": record.start_s, "end_s": record.end_s}
                receipt = root.parent / "raw_receipts" / split / f"{canonical_hash(record.sample_id)[:24]}.json"
                write_json(receipt, {"raw_metadata": raw_meta, "coordinate_sidecars": [json.loads(p.read_text()) for p in raw_path.glob("coordinates_*.json")],
                    "cache_meta_sha256": file_hash(directory / "meta.json"), "source_video_unchanged": True, "raw_arrays_retained": False})
                # Only this run's generated intermediate is removed after validated
                # feature publication; original dataset and other caches are untouched.
                if not raw_path.resolve().is_relative_to((root.parent / "pose_roi").resolve()):
                    raise ValueError("Refuse intermediate cleanup outside this cache parent")
                shutil.rmtree(raw_path)
                timings.append({"sample_id": record.sample_id, "pose_decode_seconds": prepared["seconds"],
                    "visual_roi_seconds": visual_seconds, "main_cache_seconds": time.monotonic() - clip_started})
                done += 1
                elapsed = time.monotonic() - started
                if done % 64 == 0 or done == len(records):
                    publish()
                if done % 10 == 0 or done == len(records):
                    write_json(progress_path, {"status": "running", "split": split, "completed": done, "total": len(records),
                        "elapsed_seconds": elapsed, "new_samples": done - (len(records) - len(pending)), "free_gib": shutil.disk_usage(root).free / 2 ** 30})
                    print(f"Cache {split}: {done}/{len(records)} ({elapsed:.1f}s)", flush=True)
                del bundle, verified, raw, raw_subset, video
                submit()
        finally:
            publish()
    report = {"status": "completed", "split": split, "completed": done, "total": len(records), "elapsed_seconds": time.monotonic() - started,
        "workers": workers, "worker_kind": "independent_detector_threads", "python_process_count": 1,
        "dino_batch_size": dino_batch_size, "clip_timings": timings,
        "backbone": adapter.provenance(), "lossless_compression": True, "raw_intermediates_retained": False}
    write_json(progress_path, report)
    from .training.ledger import stage_record
    stage_record(root, "full_fused_cache", split, report, elapsed_seconds=report["elapsed_seconds"], device=device)
    return report
