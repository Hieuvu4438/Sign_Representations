"""Explicit, audited exceptions to official full-train membership, before training."""

from pathlib import Path
import json
from .cache import sha256
from .data import read_manifest


def training_source(job, options):
    return Path(options.get("training_source_manifest", Path(job) / "source.jsonl"))


def full_gallery(job, options):
    return Path(options.get("full_gallery_manifest", Path(job) / "manifests/official_train.jsonl"))


def validate_pose_policy(job, options):
    """Accept only user-approved short clips; never hide other pose failures."""
    job = Path(job)
    original = read_manifest(job / "source.jsonl", "pretrain")
    report = json.loads((job / "pose_report.json").read_text())
    results = report.get("results", [])
    if report.get("errors") or len(results) != len(original):
        raise ValueError("Pose extraction is incomplete or has unapproved errors")
    indexed = {r["video_id"]: r for r in results}
    if len(indexed) != len(results) or set(indexed) != {r["video_id"] for r in original}:
        raise ValueError("Pose report membership differs from frozen official sources")
    invalid = {r["video_id"]: r for r in results if r["status"] != "PASS"}
    policy_path = options.get("approved_exclusions")
    entries = []
    if policy_path:
        policy = json.loads(Path(policy_path).read_text())
        if not policy.get("approved_by_user"):
            raise ValueError("Corpus exclusions have no recorded user authorization")
        entries = policy["entries"]
    approved = {r["video_id"]: r for r in entries}
    if len(approved) != len(entries) or set(invalid) != set(approved):
        raise ValueError("Invalid pose videos differ from the explicit approved exclusions")
    for video_id, item in approved.items():
        actual = invalid[video_id]
        if (item["reason"] != "shorter_than_16_decoded_frames"
                or actual["status"] != "NO_VALID_WINDOW"
                or not 0 < actual["decoded_frames"] < 16
                or actual["decoded_frames"] != item["decoded_frames"]
                or actual["valid_body_windows"] != 0):
            raise ValueError(f"Exclusion does not describe an observed short clip: {video_id}")
    expected = [r for r in original if r["video_id"] not in approved]
    effective = read_manifest(training_source(job, options), "pretrain")
    gallery = read_manifest(full_gallery(job, options), "train")
    expected_ids = [r["video_id"] for r in expected]
    if [r["video_id"] for r in effective] != expected_ids or [r["video_id"] for r in gallery] != expected_ids:
        raise ValueError("Training and gallery must exclude exactly the approved videos, in source order")
    for row in effective:
        receipt = indexed[row["video_id"]]
        if row.get("raw_cache"):
            if (Path(row["raw_cache"]).resolve() != Path(receipt["raw"]).resolve()
                    or row.get("raw_cache_sha256") != receipt["raw_sha256"]):
                raise ValueError("Explicit raw cache is not bound to its original pose receipt")
    if {r["label"] for r in effective} != {r["label"] for r in original}:
        raise ValueError("The exclusion removed an entire class")
    return {"status": "PASS", "official_train_videos": len(original),
            "effective_train_videos": len(effective), "gallery_videos": len(gallery),
            "classes": len({r["label"] for r in effective}), "excluded_video_ids": sorted(approved),
            "effective_source_sha256": sha256(training_source(job, options)),
            "gallery_sha256": sha256(full_gallery(job, options)),
            "official_pose_report_modified": False}
