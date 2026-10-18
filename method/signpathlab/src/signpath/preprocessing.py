"""Manifest-driven IMAGE preprocessing and per-frame DINO cache orchestration."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
import yaml

from .cache.hashing import canonical_hash, file_hash, is_sha256
from .cache.index import record_cache, sample_directory
from .cache.schema import IDENTITY_FIELDS
from .cache.writer import write_bundle
from .cli import write_json
from .config import UniqueLoader
from .data.decode import decode_video, nearest_observations
from .data.manifest import read_manifest


def manifest_records(resolved, split):
    path = resolved.path(resolved.values["data"][f"{split}_manifest"])
    records = read_manifest(path)
    if not records or any(r.split != split for r in records) or len({r.sample_id for r in records}) != len(records):
        raise ValueError(f"Expected nonempty unique {split} manifest")
    return records


def asset_plan(resolved, assets_file=None, *, include_dino=False):
    path = Path(assets_file).resolve() if assets_file else resolved.root / "registries/extraction.yaml"
    value = yaml.load(path.read_text(), Loader=UniqueLoader)
    if value.get("schema_version") != 1 or set(value) != {"schema_version", "dino", "detectors"}:
        raise ValueError("Extraction assets require schema_version/dino/detectors")
    detectors = {}
    for kind in ("pose", "hand", "face"):
        entry = value["detectors"][kind]
        file = resolved.path(entry["path"])
        if not is_sha256(entry["sha256"]) or file_hash(file) != entry["sha256"]:
            raise ValueError(f"Unverified {kind} detector asset: {file}")
        detectors[kind] = {"path": str(file), "sha256": entry["sha256"]}
    plan = {"detectors": detectors, "assets_file_sha256": file_hash(path)}
    if include_dino:
        registry_path = resolved.path(value["dino"]["registry"])
        registry = yaml.load(registry_path.read_text(), Loader=UniqueLoader)
        key = resolved.values["cache"]["extractor_registry_key"]
        entry = dict(registry["models"][key])
        weights = resolved.path(value["dino"]["weights_path"])
        checksum = value["dino"]["weights_sha256"]
        if not is_sha256(checksum) or file_hash(weights) != checksum:
            raise ValueError("DINO extraction needs explicit verified weight bytes")
        if entry.get("sha256") is not None and entry["sha256"] != checksum:
            raise ValueError("DINO extraction asset conflicts with model registry checksum")
        entry.update(repo_dir=str(resolved.path(value["dino"]["repo_dir"])), local_path=str(weights), sha256=checksum, registry_key=key)
        plan["dino"] = entry
    return plan


def raw_identity(resolved, plan):
    from .pose import extract, topology
    return canonical_hash({"detectors": {k: v["sha256"] for k, v in plan["detectors"].items()},
        "mediapipe": importlib.metadata.version("mediapipe"), "detector_mode": "IMAGE",
        "min_body_visibility": resolved.values["pose"]["min_body_visibility"],
        "extract_code": file_hash(extract.__file__), "topology_code": file_hash(topology.__file__)})


def raw_directory(resolved, record, identity):
    key = canonical_hash({"sample_id": record.sample_id, "source_sha256": record.source_sha256,
                          "start_s": record.start_s, "end_s": record.end_s})[:24]
    return resolved.path(resolved.values["cache"]["root"]).parent / "pose_roi" / identity / key


def check_source(record):
    if record.availability != "available" or not is_sha256(record.source_sha256):
        raise ValueError(f"Available video with actual source SHA-256 required: {record.sample_id}")
    if file_hash(record.video_path) != record.source_sha256:
        raise ValueError(f"Video changed since manifest import: {record.sample_id}")


def read_raw(directory, record, identity):
    from .pose.extract import RawLandmarks
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    if (meta["schema_version"], meta["sample_id"], meta["source_sha256"], meta["raw_identity"], meta["start_s"], meta["end_s"]) != (
            1, record.sample_id, record.source_sha256, identity, record.start_s, record.end_s):
        raise ValueError("Raw preprocessing source/interval/detector identity mismatch")
    if file_hash(directory / "landmarks.npz") != meta["landmarks_sha256"]:
        raise ValueError("Raw landmark content checksum mismatch")
    with np.load(directory / "landmarks.npz", allow_pickle=False) as arrays:
        fields = {name: arrays[name] for name in ("xy", "observed", "quality", "body_raw", "hand_candidates", "face_raw", "handedness_scores", "assignment")}
        raw = RawLandmarks(**fields, provenance=meta["detector_provenance"])
        timestamps, sources = arrays["timestamps"], arrays["source_frame_index"]
    return raw, timestamps, sources, meta


def ensure_raw_coordinates(directory, record, resolved, raw=None, video=None):
    """Persist image-space pixels and per-frame center/scale without changing detector data."""
    from .pose import coordinates
    directory = Path(directory)
    recipe = {"coordinate_code_sha256": file_hash(coordinates.__file__), "epsilon_px": 1.0,
              "min_shoulder_quality": resolved.values["pose"]["min_body_visibility"],
              "coordinate_system": "per_frame_shoulder_center_isotropic_scale"}
    key = canonical_hash(recipe)[:16]
    output, metadata_path = directory / f"coordinates_{key}.npz", directory / f"coordinates_{key}.json"
    landmarks_sha = file_hash(directory / "landmarks.npz")
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        if meta["source_sha256"] != record.source_sha256 or meta["landmarks_sha256"] != landmarks_sha or meta["recipe"] != recipe or file_hash(output) != meta["arrays_sha256"]:
            raise ValueError("Raw coordinate sidecar identity/content mismatch")
        return output
    if video is None:
        video = decode_video(record.video_path, record.start_s, record.end_s)
    if raw is None:
        with np.load(directory / "landmarks.npz", allow_pickle=False) as values:
            from .pose.extract import RawLandmarks
            raw_meta = json.loads((directory / "meta.json").read_text())
            raw = RawLandmarks(**{name: values[name] for name in ("xy", "observed", "quality", "body_raw", "hand_candidates", "face_raw", "handedness_scores", "assignment")}, provenance=raw_meta["detector_provenance"])
            if not np.array_equal(values["timestamps"], video.timestamps) or not np.array_equal(values["source_frame_index"], video.source_frame_index):
                raise ValueError("Raw coordinate sidecar decode no longer matches original observations")
    sizes = np.array([[frame.shape[1], frame.shape[0]] for frame in video.frames], dtype=np.float64)
    body = coordinates.body_coordinates(raw.xy, raw.observed, raw.quality, sizes, min_shoulder_quality=recipe["min_shoulder_quality"])
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez(stream, frame_sizes_wh=sizes, xy_image_px=raw.xy.astype(np.float64) * sizes[:, None],
            shoulder_center_px=body.center_px, shoulder_scale_px=body.scale_px,
            pose_coordinate_valid=body.coordinate_valid, timestamps=video.timestamps, source_frame_index=video.source_frame_index)
    try:
        import os
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(metadata_path, {"schema_version": 1, "source_sha256": record.source_sha256, "landmarks_sha256": landmarks_sha,
        "recipe": recipe, "arrays_sha256": file_hash(output), "missing_raw_coordinates": "NaN; no fabricated zero observations"})
    return output


_DETECTOR = None


def _close_detector():
    global _DETECTOR
    if _DETECTOR is not None:
        _DETECTOR.close()
        _DETECTOR = None


def _init_detector(plan, min_visibility):
    from .pose.extract import LandmarkExtractor
    global _DETECTOR
    _DETECTOR = LandmarkExtractor(*(plan["detectors"][k]["path"] for k in ("pose", "hand", "face")), min_body_visibility=min_visibility)


def _preprocess_one(resolved, record, identity, *, detector=None):
    started = time.monotonic()
    check_source(record)
    destination = raw_directory(resolved, record, identity)
    if destination.exists():
        _, times, _, _ = read_raw(destination, record, identity)
        ensure_raw_coordinates(destination, record, resolved)
        return {"sample_id": record.sample_id, "status": "verified_reuse", "frames": len(times), "seconds": time.monotonic() - started}
    video = decode_video(record.video_path, record.start_s, record.end_s)
    raw = (detector if detector is not None else _DETECTOR).extract(video.frames)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".raw-", dir=destination.parent))
    try:
        np.savez(temporary / "landmarks.npz", **{name: getattr(raw, name) for name in
            ("xy", "observed", "quality", "body_raw", "hand_candidates", "face_raw", "handedness_scores", "assignment")},
            timestamps=video.timestamps, source_frame_index=video.source_frame_index)
        write_json(temporary / "meta.json", {"schema_version": 1, "sample_id": record.sample_id, "source_sha256": record.source_sha256,
            "raw_identity": identity, "start_s": record.start_s, "end_s": record.end_s,
            "detector_provenance": raw.provenance, "landmarks_sha256": file_hash(temporary / "landmarks.npz"),
            "timestamp_fallback": video.timestamp_fallback, "nominal_fps": video.nominal_fps})
        temporary.rename(destination)
        ensure_raw_coordinates(destination, record, resolved, raw, video)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"sample_id": record.sample_id, "status": "completed", "frames": len(video.frames), "seconds": time.monotonic() - started}


def _preprocess_shard(resolved, records, identity, plan):
    _init_detector(plan, resolved.values["pose"]["min_body_visibility"])
    results, failures = [], []
    try:
        for record in records:
            try:
                results.append(_preprocess_one(resolved, record, identity))
            except (ValueError, OSError, RuntimeError) as error:
                failures.append({"sample_id": record.sample_id, "reason": str(error)})
    finally:
        # Close before Python's global ThreadPool shutdown; MediaPipe close dispatches callbacks.
        _close_detector()
    return results, failures


def preprocess(resolved, split, assets_file=None, shards=1):
    stage_started = time.monotonic()
    if shards < 1:
        raise ValueError("Preprocessing shards must be positive")
    plan = asset_plan(resolved, assets_file)
    identity = raw_identity(resolved, plan)
    records = manifest_records(resolved, split)
    available = [r for r in records if r.availability == "available"]
    if not available:
        raise ValueError("Preprocessing has no actually available video")
    results, failures = [], []
    def completed(record, result=None, error=None):
        if error:
            failures.append({"sample_id": record.sample_id, "reason": str(error)})
        else:
            results.append(result)
    if shards == 1:
        _init_detector(plan, resolved.values["pose"]["min_body_visibility"])
        try:
            for record in available:
                try:
                    completed(record, _preprocess_one(resolved, record, identity))
                except (ValueError, OSError, RuntimeError) as error:
                    completed(record, error=error)
        finally:
            _close_detector()
    else:
        partitions = [available[i::shards] for i in range(min(shards, len(available)))]
        with ProcessPoolExecutor(max_workers=shards, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_preprocess_shard, resolved, part, identity, plan): part for part in partitions}
            for future in as_completed(futures):
                try:
                    done, failed = future.result()
                    results.extend(done)
                    failures.extend(failed)
                except Exception as error:
                    failures.extend({"sample_id": record.sample_id, "reason": str(error)} for record in futures[future])
    report = {"status": "failed" if failures else "completed", "split": split, "raw_identity": identity,
              "shards": shards, "results": sorted(results, key=lambda r: r["sample_id"]), "failures": failures,
              "unavailable_records": len(records) - len(available)}
    path = resolved.path(resolved.values["cache"]["root"]).parent / "pose_roi" / identity / f"{split}_report.json"
    write_json(path, report)
    from .training.ledger import stage_record
    stage_record(path.parent, "pose_roi", split, report, elapsed_seconds=time.monotonic() - stage_started, device="CPU IMAGE delegates")
    if failures:
        raise ValueError(f"Preprocessing failures; inspect {path}")
    return {"status": report["status"], "completed": len(results), "report": str(path)}


def build_cache(resolved, split, assets_file=None, device="cpu"):
    stage_started = time.monotonic()
    from .backbones.dino import DINOAdapter
    from .pose.context import ExtractionContext
    from .cache.reader import read_bundle
    plan = asset_plan(resolved, assets_file, include_dino=True)
    raw_hash = raw_identity(resolved, plan)
    adapter = DINOAdapter().load(plan["dino"], device)
    root = resolved.path(resolved.values["cache"]["root"])
    results, failures = [], []
    for record in manifest_records(resolved, split):
        if record.availability != "available":
            continue
        try:
            check_source(record)
            index_file = root / "index.json"
            index = json.loads(index_file.read_text()) if index_file.exists() else None
            entry = index["samples"].get(record.sample_id) if index else None
            if entry is not None:
                directory = (root / entry["path"]).resolve()
                if not directory.is_relative_to(root.resolve()) or file_hash(directory / "meta.json") != entry["meta_sha256"]:
                    raise ValueError("Indexed cache path or metadata checksum mismatch")
                cached = read_bundle(directory, index["identity"])
                cached_code = cached.meta["preprocessing"]["implementation_sha256"]
                source_root = Path(__file__).parent
                if (cached.meta.get("raw_preprocessing_identity") != raw_hash
                    or cached.meta.get("selection_target_fps") != resolved.values["data"]["target_fps"]
                    or cached.meta["weights_sha256"] != plan["dino"]["sha256"]
                    or any(file_hash(source_root / name) != digest for name, digest in cached_code.items())):
                    raise ValueError("Existing cache uses a different extraction/sampling implementation; choose a new cache root")
                record_cache(root, directory, record)
                results.append({"sample_id": record.sample_id, "status": "verified_reuse", "samples": len(cached.timestamps)})
                continue
            raw, times, source_ids, meta = read_raw(raw_directory(resolved, record, raw_hash), record, raw_hash)
            video = decode_video(record.video_path, record.start_s, record.end_s)
            if not np.array_equal(video.timestamps, times) or not np.array_equal(video.source_frame_index, source_ids):
                raise ValueError("Decoded timestamps/indices differ from raw landmark observations")
            context = ExtractionContext(record.sample_id, record.source_sha256, source_ids, raw,
                {"source_mirrored": record.metadata.get("source_mirrored"), "hand_assignment": raw.provenance["hand_assignment"]},
                min_body_visibility=resolved.values["pose"]["min_body_visibility"])
            bundle = adapter.extract(video.frames, times, context)
            selected = nearest_observations(times, source_ids, resolved.values["data"]["target_fps"])
            names = ("timestamps", "source_frame_index", "visual", "visual_valid", "pose_xy_body", "pose_observed", "pose_quality", "pose_coordinate_valid")
            bundle = replace(bundle, **{name: getattr(bundle, name)[selected.indices] for name in names},
                raw_supports=tuple(s for t in selected.indices for s in bundle.raw_supports[t * 7:(t + 1) * 7]))
            bundle.meta.update(selected_original_array_indices=selected.indices.tolist(), timestamp_fallback=meta["timestamp_fallback"],
                source_interval_s=[record.start_s, record.end_s], raw_preprocessing_identity=raw_hash, split=split,
                selection_target_fps=resolved.values["data"]["target_fps"],
                duplicate_selection=selected.duplicate_selection.tolist())
            directory = sample_directory(root, bundle.meta["extractor_hash"], record.sample_id)
            if directory.exists():
                read_bundle(directory, {k: bundle.meta[k] for k in IDENTITY_FIELDS})
                status = "verified_reuse"
            else:
                write_bundle(directory, bundle)
                status = "completed"
            record_cache(root, directory, record)
            results.append({"sample_id": record.sample_id, "status": status, "samples": len(bundle.timestamps)})
        except (ValueError, OSError, RuntimeError) as error:
            failures.append({"sample_id": record.sample_id, "reason": str(error)})
    report_path = root / f"{split}_build_report.json"
    report = {"status": "failed" if failures else "completed", "split": split, "results": results, "failures": failures,
              "backbone": adapter.provenance(), "assets_file_sha256": plan["assets_file_sha256"]}
    write_json(report_path, report)
    from .training.ledger import stage_record
    stage_record(root, "visual_cache", split, report, elapsed_seconds=time.monotonic() - stage_started, device=device)
    if failures:
        raise ValueError(f"Cache build failures; inspect {report_path}")
    return {"status": "completed", "samples": len(results), "report": str(report_path)}
