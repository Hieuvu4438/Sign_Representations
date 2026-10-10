"""ASL Citizen manifest construction and per-video eligibility audit."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .pose_reader import read_raw_pose
from .video import probe_video, valid_clip_starts

SPLITS = ("train", "val", "test")


def stable_int(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def read_split_rows(dataset_root: Path, label_map: Dict[str, int]) -> List[dict]:
    rows, seen = [], set()
    for split in SPLITS:
        with open(dataset_root / "splits" / f"{split}.csv", newline="") as f:
            for r in csv.DictReader(f):
                vid = r["Video file"]
                sid = Path(vid).stem
                if sid in seen:
                    raise ValueError(f"duplicate sample id {sid}")
                seen.add(sid)
                rows.append({
                    "sample_id": sid,
                    "dataset": "ASL_Citizen",
                    "official_split": split,
                    "source_root": str(dataset_root / "videos"),
                    "relative_path": vid,
                    "pose_path": str(dataset_root / "keypoints" / f"{sid}.pkl"),
                    "parent_video_id": sid,
                    "signer_id": r["Participant ID"],
                    "gloss": r["Gloss"],
                    "label_id": int(label_map[r["Gloss"]]),
                })
    return rows


def audit_record(rec: dict, clip_length: int, max_rel_dev: float, score_threshold: float) -> dict:
    out = dict(rec)
    reasons = []
    vpath = str(Path(rec["source_root"]) / rec["relative_path"])
    try:
        pv = probe_video(vpath)
    except Exception as e:  # decode failure keeps the record with a reason
        out.update(eligible=False, reasons=[f"decode_error:{type(e).__name__}"])
        return out
    F = pv["num_frames"]
    out.update(num_frames=F, width=pv["width"], height=pv["height"], average_rate=pv["average_rate"])
    pts = pv["pts_sec"]
    out["pts_first"] = float(pts[0]) if F else None
    out["median_dt"] = float(np.median(np.diff(pts))) if F > 1 else None
    if F < clip_length:
        reasons.append("short_video")
    starts = valid_clip_starts(pts, clip_length, max_rel_dev) if F >= clip_length else np.zeros(0, np.int64)
    out["num_valid_starts"] = int(len(starts))
    out["num_possible_starts"] = max(0, F - clip_length + 1)
    if F >= clip_length and len(starts) == 0:
        reasons.append("clock_policy_no_valid_start")
    try:
        raw = read_raw_pose(rec["pose_path"], score_threshold)
        out["pose_frames"] = raw["num_frames"]
        if raw["num_frames"] != F:
            reasons.append(f"pose_frame_count_mismatch:{raw['num_frames']}!={F}")
        out["frac_single_person"] = float((raw["person_count"] == 1).mean()) if F else 0.0
        out["frac_left_present"] = float(raw["hand_present"][:, 0].mean()) if F else 0.0
        out["frac_right_present"] = float(raw["hand_present"][:, 1].mean()) if F else 0.0
    except Exception as e:
        reasons.append(f"pose_error:{type(e).__name__}:{e}")
    out["valid_starts"] = starts.tolist() if len(starts) != out["num_possible_starts"] else "all"
    out["eligible"] = not reasons
    out["reasons"] = reasons
    return out


def record_valid_starts(rec: dict, clip_length: int) -> np.ndarray:
    vs = rec["valid_starts"]
    if vs == "all":
        return np.arange(0, rec["num_frames"] - clip_length + 1, dtype=np.int64)
    return np.asarray(vs, dtype=np.int64)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "x") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
