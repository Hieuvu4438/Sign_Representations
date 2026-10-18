"""Apply the user's single short-clip exclusion without reindexing raw pose caches."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from signrep_baseline.cache import sha256
from signrep_baseline.corpus_policy import validate_pose_policy
from signrep_baseline.data import read_manifest
from signrep_baseline.job_support import atomic_json, progress
from signrep_baseline.preparation import infer_signer_bbox
from setup_full_aslc_job import snapshot, write_rows, SERVICE

BAD_ID = "asl_citizen:011101656879706212-YAWN 1"


def run(job, start):
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    if list((job / "train").glob("*.pt")):
        raise ValueError("This migration is only allowed before production training starts")
    subprocess.run(["systemctl", "--user", "stop", SERVICE], check=True)
    marker = job / "short_clip_repair.json"
    if marker.exists():
        opts = json.loads((job / "options.json").read_text())
        validate_pose_policy(job, opts)
        if start:
            subprocess.run(["systemctl", "--user", "reset-failed", SERVICE], check=True)
            subprocess.run(["systemctl", "--user", "start", SERVICE], check=True)
        print(marker.read_text())
        return
    source = read_manifest(job / "source.jsonl", "pretrain")
    report = json.loads((job / "pose_report.json").read_text())
    bad = [r for r in report["results"] if r["status"] != "PASS"]
    if (report["errors"] or len(report["results"]) != len(source) or len(bad) != 1
            or bad[0]["video_id"] != BAD_ID or bad[0]["decoded_frames"] != 11
            or bad[0]["valid_body_windows"] != 0):
        raise ValueError("Pose report does not match the single user-approved 11-frame exception")
    receipts = {r["video_id"]: r for r in report["results"]}
    if len(receipts) != len(source) or set(receipts) != {r["video_id"] for r in source}:
        raise ValueError("Pose report has duplicated/missing source identities")
    archive = job / "history" / ("before_short_clip_exclusion_" + time.strftime("%Y%m%dT%H%M%S"))
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("options.json", "inventory.json", "input_pins.json", "code_manifest.json", "status.json"):
        shutil.copy2(job / name, archive / name)
    opts = json.loads((job / "options.json").read_text())
    teacher_weights = {p.name: sha256(p) for p in Path(opts["teacher_weights"]).glob("*.ckpt")}
    effective = []
    started = time.monotonic()
    for index, row in enumerate(source):
        if row["video_id"] == BAD_ID:
            continue
        receipt = receipts[row["video_id"]]
        raw = Path(receipt["raw"])
        if sha256(raw) != receipt["raw_sha256"]:
            raise ValueError(f"Completed raw cache changed: {row['video_id']}")
        with np.load(raw, allow_pickle=False) as cache:
            meta = json.loads(str(cache["metadata_json"]))
            if (meta["video_sha256"] != sha256(row["video"])
                    or meta["schema"] != "signrep_teacher_v1"
                    or meta["body_root_mode"] != "canonical" or meta["weights"] != teacher_weights
                    or meta["decoded_frames"] < 16
                    or not np.array_equal(cache["frame_indices"], np.arange(meta["decoded_frames"]))):
                raise ValueError(f"Pose/video/teacher alignment mismatch: {row['video_id']}")
        # Header-only single-thread capture avoids extra decoder thread pools.
        cap = cv2.VideoCapture(str(row["video"]), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {row['video_id']}")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        bbox = infer_signer_bbox(raw, width, height)
        if not 0 <= bbox[0] < bbox[2] <= width or not 0 <= bbox[1] < bbox[3] <= height:
            raise ValueError(f"Invalid signer crop: {row['video_id']}")
        effective.append({**row, "raw_cache": str(raw), "raw_cache_sha256": receipt["raw_sha256"],
                          "original_pose_index": receipt["index"], "video_sha256": meta["video_sha256"],
                          "signer_bbox_xyxy": bbox})
        if len(effective) % 1000 == 0 or index + 1 == len(source):
            progress(job, "REPAIR_PREFLIGHT", completed=len(effective), total=len(source) - 1,
                     elapsed_minutes=round((time.monotonic() - started) / 60, 2))
    if len(effective) != 40153 or len({r["label"] for r in effective}) != 2731:
        raise ValueError("Repaired membership does not preserve the expected count/classes")
    unchanged = {name: sha256(job / "manifests" / name) for name in
                 ("official_val.jsonl", "official_test.jsonl", "val_proxy.jsonl", "gallery_proxy.jsonl")}
    for name in ("gallery_proxy.jsonl", "val_proxy.jsonl"):
        if any(r["video_id"] == BAD_ID for r in read_manifest(job / "manifests" / name)):
            raise ValueError("Proxy contains the bad video; requires a separately audited replacement")
    approval = {"schema": "signrep_corpus_exclusions_v1", "approved_by_user": True,
                "approval": "User requested the recommended single-clip exclusion on 2026-10-14",
                "original_files_deleted": False, "entries": [{"video_id": BAD_ID,
                "decoded_frames": 11, "reason": "shorter_than_16_decoded_frames",
                "original_pose_index": bad[0]["index"], "raw_cache": bad[0]["raw"],
                "raw_sha256": bad[0]["raw_sha256"]}],
                "original_source_sha256": sha256(job / "source.jsonl"),
                "pose_report_sha256": sha256(job / "pose_report.json")}
    atomic_json(job / "approved_exclusions.json", approval)
    write_rows(job / "source_training.jsonl", effective)
    write_rows(job / "manifests/gallery_train.jsonl", [{**r, "split": "train"} for r in effective])
    opts.update(training_source_manifest=str(job / "source_training.jsonl"),
                full_gallery_manifest=str(job / "manifests/gallery_train.jsonl"),
                approved_exclusions=str(job / "approved_exclusions.json"))
    atomic_json(job / "options.json", opts)
    policy = validate_pose_policy(job, opts)
    counts = Counter(r["label"] for r in effective)
    excluded_label = next(r["label"] for r in source if r["video_id"] == BAD_ID)
    audit = {**policy, "original_source_sha256": sha256(job / "source.jsonl"),
             "pose_report_sha256": sha256(job / "pose_report.json"),
             "approval_sha256": sha256(job / "approved_exclusions.json"),
             "remaining_YAWN1_train_videos": counts[excluded_label],
             "raw_caches_reused": len(effective), "raw_cache_files_reindexed": False,
             "held_out_and_proxy_pins_unchanged": unchanged,
             "original_files_deleted": False, "training_started_before_repair": False,
             "archive": str(archive), "timestamp": time.time()}
    atomic_json(job / "membership_exclusion_audit.json", audit)
    inventory = json.loads((job / "inventory.json").read_text())
    inventory.update(effective_train_membership=40153, effective_gallery_membership=40153,
                     approved_excluded_video_ids=[BAD_ID], invalid_pose_policy="explicit_user_approved_exclusions_only")
    atomic_json(job / "inventory.json", inventory)
    snapshot(job)
    pins = json.loads((job / "input_pins.json").read_text())
    for path in (job / "options.json", job / "inventory.json", job / "approved_exclusions.json",
                 job / "source_training.jsonl", job / "manifests/gallery_train.jsonl",
                 job / "membership_exclusion_audit.json", job / "pose_report.json"):
        pins[str(path)] = sha256(path)
    atomic_json(job / "input_pins.json", pins)
    if any(sha256(job / "manifests" / name) != expected for name, expected in unchanged.items()):
        raise ValueError("Held-out/proxy manifests changed unexpectedly")
    atomic_json(marker, audit)
    progress(job, "REPAIRED_READY", videos=40153, classes=2731, excluded=1,
             raw_caches_reused=40153, stage_limit=50000)
    subprocess.run(["systemctl", "--user", "reset-failed", SERVICE], check=True)
    if start:
        subprocess.run(["systemctl", "--user", "start", SERVICE], check=True)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, default=ROOT / "runtime/full_aslc_kinetics")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    run(args.job.resolve(), args.start)
