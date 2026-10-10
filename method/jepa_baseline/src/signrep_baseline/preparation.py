"""Audited corpus preparation without temporal frame dropping or per-video bone scales."""

import json
from pathlib import Path
import subprocess
import numpy as np
from .cache import sha256
from .data import read_manifest, deduplicate_manifest, deduplicate_local_manifest, source_identity
from .bones import corpus_bone_statistics

ROOT = Path(__file__).resolve().parents[2]


def pose_command(python, command, weights, **paths):
    arguments = [str(python), str(ROOT / "scripts/pose_worker.py"), command, "--weights", str(weights)]
    for name, value in paths.items():
        arguments.extend(["--" + name.replace("_", "-"), str(value)])
    return subprocess.run(arguments, check=True, text=True, capture_output=True).stdout


def infer_signer_bbox(raw_cache, width, height, margin=.1):
    """Named engineering crop: union of observed upper-body/face/hand coordinates."""
    with np.load(raw_cache, allow_pickle=False) as file:
        points = np.concatenate([file[key] for key in ["body2d", "right2d", "left2d"]], axis=1)
        valid = file["confidence"] > .5
    values = points[valid]
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Cannot infer signer crop from absent/nonfinite pose observations")
    low, high = values.min(0), values.max(0)
    extent = high - low
    low, high = np.floor(low - margin * extent), np.ceil(high + margin * extent)
    return [max(0, int(low[0])), max(0, int(low[1])), min(width, int(high[0]) + 1), min(height, int(high[1]) + 1)]


def prepare_corpus(source_manifest, wlasl_annotations, output_directory, pose_python, weights,
                   held_out_manifest=None, training_reference_manifest=None):
    import cv2
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    filtered = output / "filtered_source.jsonl"
    if (wlasl_annotations is None) == (held_out_manifest is None):
        raise ValueError("Choose exactly one exclusion protocol: WLASL or local held-out manifest")
    if held_out_manifest and not training_reference_manifest:
        raise ValueError("Local corpus requires an official training reference manifest")
    audit = (deduplicate_manifest(source_manifest, wlasl_annotations, filtered) if wlasl_annotations
             else deduplicate_local_manifest(source_manifest, held_out_manifest, filtered, training_reference_manifest))
    rows = read_manifest(filtered, "pretrain")
    cache_paths = []
    for index, row in enumerate(rows):
        video = Path(row["video"])
        raw = output / "raw" / f"{index:08d}.npz"
        if not raw.exists():
            pose_command(pose_python, "extract", weights, video=video, output=raw)
        with np.load(raw, allow_pickle=False) as file:
            metadata = json.loads(str(file["metadata_json"]))
        if metadata["video_sha256"] != sha256(video) or metadata["schema"] != "signrep_teacher_v1" or metadata.get("body_root_mode") != "canonical":
            raise ValueError("Raw cache video/source mismatch; use fresh output directory")
        cap = cv2.VideoCapture(str(video))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        bbox = row.get("signer_bbox_xyxy") or infer_signer_bbox(raw, width, height)
        row.update(video=str(video.resolve()), video_sha256=metadata["video_sha256"],
                   raw_cache=str(raw), raw_cache_sha256=sha256(raw), signer_bbox_xyxy=bbox,
                   crop_policy="supplied bbox or observed upper-body/face/hand union +10% (U06)")
        cache_paths.append(raw)
    # Freeze corpus membership before derived bone/cache hashes exist (no circular pin).
    filtered.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    corpus_sha = sha256(filtered)
    stats = output / "corpus_bones.npz"
    corpus_bone_statistics(cache_paths, stats, corpus_sha)
    bones_sha = sha256(stats)
    tasks = [{"raw": str(cache_paths[index]), "stats": str(stats),
              "output": str(output / "normalized" / f"{index:08d}.npz")}
             for index in range(len(rows))]
    task_manifest = output / "normalization_tasks.json"
    task_manifest.write_text(json.dumps(tasks))
    pose_command(pose_python, "normalize-batch", weights, manifest=task_manifest)
    for index, row in enumerate(rows):
        normalized = output / "normalized" / f"{index:08d}.npz"
        row.update(pose_cache=str(normalized), pose_cache_sha256=sha256(normalized), bone_statistics_sha256=bones_sha,
                   corpus_manifest_sha256=corpus_sha)
    manifest = output / "pretrain.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    audit.update(filtered_source_sha256=corpus_sha, prepared_manifest_sha256=sha256(manifest),
                 bone_statistics_sha256=bones_sha, filtered_source=str(filtered), bone_statistics=str(stats),
                 prepared_manifest=str(manifest))
    manifest.with_suffix(".jsonl.audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def validate_pretrain_audit(manifest):
    manifest = Path(manifest)
    audit = json.loads(manifest.with_suffix(manifest.suffix + ".audit.json").read_text())
    rows = read_manifest(manifest, "pretrain")
    if audit["prepared_manifest_sha256"] != sha256(manifest) or audit["filtered_source_sha256"] != sha256(audit["filtered_source"]):
        raise ValueError("Audited pretraining manifest changed")
    if audit["bone_statistics_sha256"] != sha256(audit["bone_statistics"]):
        raise ValueError("Audited corpus bone table changed")
    from .bones import load_bone_statistics
    load_bone_statistics(audit["bone_statistics"], audit["filtered_source_sha256"])
    from .data import youtube_id
    local = audit.get("audit_kind") == "local_train_only"
    if local:
        if sha256(audit["training_reference_manifest"]) != audit["training_reference_sha256"]:
            raise ValueError("Official training reference manifest changed")
        reference = {source_identity(row): row for row in read_manifest(audit["training_reference_manifest"], "train")}
        if sha256(audit["held_out_manifest"]) != audit["held_out_sha256"]:
            raise ValueError("Official held-out exclusion manifest changed")
        excluded = set(audit["excluded_source_ids"])
        held_out = read_manifest(audit["held_out_manifest"])
        blocked_paths = {row["video"] for row in held_out if row.get("video")}
        blocked_hashes = {row["video_sha256"] for row in held_out if row.get("video_sha256")}
    for row in rows:
        if local:
            original = reference.get(source_identity(row))
            if original is None or row["video"] != original["video"]:
                raise ValueError("Pretraining membership differs from official training reference")
            if row.get("source_split") != "train" or source_identity(row) in excluded or row["video"] in blocked_paths or row["video_sha256"] in blocked_hashes:
                raise ValueError("Held-out video leaked into local pretraining")
        elif youtube_id(row["youtube_id"]) in set(audit["excluded_ids"]):
            raise ValueError("WLASL video leaked into pretraining")
        if row["bone_statistics_sha256"] != audit["bone_statistics_sha256"] or row["corpus_manifest_sha256"] != audit["filtered_source_sha256"]:
            raise ValueError("Prepared cache provenance differs across corpus")
    return audit
