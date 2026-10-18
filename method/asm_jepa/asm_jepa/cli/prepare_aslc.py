"""Official ASL-Citizen CSVs -> canonical manifests, train-only vocab, split audit, version/diff, hashes.

Idempotent: re-running rewrites identical files; video header probes are reused when the CSV hashes match
(pass --reprobe to force). No random validation split is ever created; unknown val/test glosses fail.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from asm_jepa.cli.common import add_config_args, config_from_args, run
from asm_jepa.data.manifest import (SPLITS, ProtocolError, audit_splits, build_split_rows, build_vocab,
                                    read_official_csv)
from asm_jepa.utils import now, read_json, read_jsonl, resolve, sha256_file, write_json, write_jsonl


def find_root(raw_root: Path, splits_dir: str, train_file: str) -> Path:
    if (raw_root / splits_dir / train_file).is_file():
        return raw_root
    for cand in sorted(raw_root.glob(f"**/{splits_dir}/{train_file}")):
        if len(cand.relative_to(raw_root).parts) <= 4:
            return cand.parent.parent
    raise FileNotFoundError(f"no {splits_dir}/{train_file} under {raw_root}")


def probe_video(path: str):
    """Header-only probe (container fps / frame count / size); full decode checks happen in extract_pose."""
    import cv2

    cv2.setNumThreads(1)
    p = Path(path)
    if not p.is_file():
        return {"decode_status": "missing"}
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            return {"decode_status": "unreadable_header"}
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, n = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if w <= 0 or h <= 0 or n <= 0 or not fps > 0:
        return {"decode_status": "unreadable_header", "width": w, "height": h, "fps": fps, "num_frames": n}
    return {"decode_status": "header_ok", "width": w, "height": h, "fps": fps, "num_frames": n,
            "duration_sec": n / fps, "video_size_bytes": p.stat().st_size}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default="configs/data/aslc.yaml")
    ap.add_argument("--raw-root")
    ap.add_argument("--output-root")
    ap.add_argument("--expected-version")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--no-probe", action="store_true", help="skip video header probing")
    ap.add_argument("--reprobe", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="read/audit only, write nothing")
    args = ap.parse_args()
    cfg = config_from_args(args, "data")
    raw_root = find_root(resolve(args.raw_root or cfg["raw_root"]), cfg["splits_dir"], cfg["split_files"]["train"])
    out = resolve(args.output_root or cfg["processed_root"])
    version = args.expected_version or cfg["expected_version"]
    workers = args.workers or cfg["probe_workers"]

    raw, csv_info, csv_sha = {}, {}, {}
    for split in SPLITS:
        path = raw_root / cfg["splits_dir"] / cfg["split_files"][split]
        if not path.is_file():
            raise FileNotFoundError(f"official {split} CSV missing: {path} (a random validation split is never created)")
        raw[split], csv_info[split] = read_official_csv(path, cfg["csv_headers"], cfg["csv_positions"])
        csv_sha[split] = sha256_file(path)
    vocab = build_vocab(raw["train"])
    try:
        rows = {s: build_split_rows(s, raw[s], vocab, cfg["videos_dir"]) for s in SPLITS}
    except ProtocolError as e:
        raise ValueError(f"protocol check failed: {e}")
    for s, rs in rows.items():
        ids = [r["sample_id"] for r in rs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"sample_id collision in {s}")
    all_ids = [r["sample_id"] for rs in rows.values() for r in rs]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("sample_id collision across splits")
    audit = audit_splits(rows)
    if not audit["disjoint"]:
        raise ValueError(f"official splits overlap: {audit['overlaps']}")
    exp = cfg["expected_counts"]
    observed = {"train": len(rows["train"]), "val": len(rows["val"]), "test": len(rows["test"]),
                "glosses": len(vocab), "train_signers": audit["num_signers"]["train"],
                "val_signers": audit["num_signers"]["val"], "test_signers": audit["num_signers"]["test"]}
    diff = {k: {"expected": exp[k], "observed": observed[k]} for k in exp if exp[k] != observed[k]}
    print(f"raw root      {raw_root}")
    print(f"counts        {observed}")
    print(f"matches {version}: {not diff}" + ("" if not diff else f"  diff={diff}"))
    print(f"split audit   disjoint={audit['disjoint']}")
    if args.dry_run:
        return 0

    # ---- reuse header probes when the official CSVs did not change
    previous = {}
    vfile = out / "dataset_version.json"
    if vfile.is_file() and not args.reprobe:
        old = read_json(vfile)
        if old.get("csv_sha256") == csv_sha:
            for s in SPLITS:
                pth = out / f"metadata_{s}.jsonl"
                if pth.is_file():
                    for r in read_jsonl(pth):
                        if r.get("decode_status") not in (None, "not_checked"):
                            previous[r["sample_id"]] = r
    keys = ("width", "height", "fps", "num_frames", "duration_sec", "video_size_bytes", "decode_status")
    todo = [r for s in SPLITS for r in rows[s] if r["sample_id"] not in previous]
    if not args.no_probe and todo:
        print(f"probing {len(todo)} video headers with {workers} workers ...", flush=True)
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(probe_video, [str(raw_root / r["video_relpath"]) for r in todo], chunksize=64))
        for r, res in zip(todo, results):
            previous[r["sample_id"]] = res
    for s in SPLITS:
        for r in rows[s]:
            if r["sample_id"] in previous:
                for k in keys:
                    if k in previous[r["sample_id"]]:
                        r[k] = previous[r["sample_id"]][k]
    unreadable = [{"sample_id": r["sample_id"], "official_split": s, "video_relpath": r["video_relpath"],
                   "decode_status": r["decode_status"]}
                  for s in SPLITS for r in rows[s] if r["decode_status"] in ("missing", "unreadable_header")]

    out.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        write_jsonl(out / f"metadata_{s}.jsonl", rows[s])
    ssl_rows = [{"sample_id": r["sample_id"], "pose_relpath": r["pose_relpath"], "duration_sec": r["duration_sec"],
                 "fps": r["fps"], "num_frames": r["num_frames"]} for r in rows["train"]]
    write_jsonl(out / "ssl_train.jsonl", ssl_rows)
    write_json(out / "vocab.json", {"source": "official train split only", "rule": "sorted(unique(gloss.strip()))",
                                    "num_classes": len(vocab), "gloss_to_class_id": vocab})
    write_json(out / "split_audit.json", audit)
    write_jsonl(out / "unreadable_videos.jsonl", unreadable)
    hashes = {f"metadata_{s}.jsonl": sha256_file(out / f"metadata_{s}.jsonl") for s in SPLITS}
    hashes.update({"ssl_train.jsonl": sha256_file(out / "ssl_train.jsonl"), "vocab.json": sha256_file(out / "vocab.json")})
    write_json(out / "manifest_hashes.json", hashes)
    write_json(vfile, {"dataset": cfg["dataset"], "expected_version": version, "matches_expected": not diff,
                       "dataset_diff": diff, "observed_counts": observed, "csv_sha256": csv_sha,
                       "csv_columns": csv_info, "raw_root": str(raw_root), "videos_dir": cfg["videos_dir"],
                       "sample_id_rule": "sha256(videos/<official filename>)[:24]", "created": now(),
                       "unreadable_videos": len(unreadable)})
    if diff:
        write_json(out / "dataset_diff.json", diff)
    print(f"wrote manifests to {out} (unreadable/missing videos: {len(unreadable)})")
    return 0


if __name__ == "__main__":
    run(main)
