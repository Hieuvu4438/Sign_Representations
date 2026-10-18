"""Measured cache extrapolation, excluding pre-existing videos/weights/environment."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import zipfile
from signpath.cli import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/full_aslc/cache")
    parser.add_argument("--output", default="docs/evidence/full_aslc_storage.json")
    args = parser.parse_args()
    cache = Path(args.cache)
    measurements = []
    for path in sorted(cache.rglob("meta.json")):
        arrays, supports = path.parent / "features.npz", path.parent / "raw_supports.npz"
        if not arrays.exists() or not supports.exists():
            continue
        metadata = json.loads(path.read_text())
        sizes = {p.name: p.stat().st_size for p in (arrays, supports, path)}
        unpacked = 0
        for archive in (arrays, supports):
            with zipfile.ZipFile(archive) as handle:
                unpacked += sum(info.file_size for info in handle.infolist())
        measurements.append({"bytes": sum(sizes.values()), "allocated_bytes": sum(p.stat().st_blocks * 512 for p in (arrays, supports, path)),
            "uncompressed_payload_bytes": unpacked + sizes["meta.json"], "sizes": sizes,
            "selected_samples": len(metadata["selected_original_array_indices"])})
    if not measurements:
        raise ValueError("No completed compressed feature cache available")
    mean = statistics.mean(row["bytes"] for row in measurements)
    allocated = statistics.mean(row["allocated_bytes"] for row in measurements)
    plain = statistics.mean(row["uncompressed_payload_bytes"] for row in measurements)
    gib = 2 ** 30
    counts = {"train": 40154, "val": 10304, "test": 32941, "train_val": 50458, "all": 83399, "ASLC3000_train_val": 4500}
    projection = {name: {"videos": count, "cache_logical_gib": mean * count / gib,
        "cache_allocated_gib": allocated * count / gib, "uncompressed_cache_payload_gib": plain * count / gib,
        "planning_gib_including_20_to_30_percent_cache_margin_and_2_to_3_gib_artifacts": [allocated * count / gib * 1.2 + 2, allocated * count / gib * 1.3 + 3]}
        for name, count in counts.items()}
    root = Path.cwd()
    report = {"status": "measured_sample_extrapolation", "sample_videos": len(measurements),
        "sample_selection": "completed prefix of official train; not randomly representative of full corpus",
        "mean_selected_samples": statistics.mean(r["selected_samples"] for r in measurements),
        "mean_cache_bytes_per_video": mean, "mean_cache_allocated_bytes_per_video": allocated,
        "bytes_by_component_mean": {k: statistics.mean(r["sizes"][k] for r in measurements) for k in ("features.npz", "raw_supports.npz", "meta.json")},
        "measured_cache_sample_gib": sum(r["bytes"] for r in measurements) / gib,
        "free_disk_gib": shutil.disk_usage(root).free / gib,
        "projected": projection, "existing_locked_venv_gib": sum(p.stat().st_size for p in (root / ".venv-locked").rglob("*") if p.is_file()) / gib,
        "checkpoint_bytes_measured": next((p.stat().st_size for p in (root / "artifacts/full_aslc/preflight/training").glob("*/checkpoints/last.pt")), None),
        "excluded": "original ASL Citizen videos and pre-existing shared weights; venv listed separately; temporary raw queue bounded",
        "limitations": "prefix estimate; clip length/availability affect size; full-corpus exact totals unmeasured; free space shared with other tasks"}
    original = {}
    for split in ("train", "val", "test"):
        manifest = root / f"data/full_aslc/manifests/asl_citizen.{split}.jsonl"
        if not manifest.exists():
            continue
        total, count = 0, 0
        with manifest.open() as stream:
            for line in stream:
                record = json.loads(line)
                total += Path(record["video_path"]).stat().st_size
                count += 1
        original[split] = {"videos": count, "bytes": total, "gib": total / gib}
    report["original_video_sizes_measured"] = original
    if len(original) == 3:
        report["original_all_video_gib"] = sum(row["gib"] for row in original.values())
        report["from_empty_machine_estimate_gib_all_splits"] = (report["original_all_video_gib"] + report["existing_locked_venv_gib"]
            + projection["all"]["planning_gib_including_20_to_30_percent_cache_margin_and_2_to_3_gib_artifacts"][1])
    write_json(args.output, report)
    print(json.dumps({"sample_videos": len(measurements), "free_disk_gib": report["free_disk_gib"], "projected": projection}, indent=2))


if __name__ == "__main__":
    main()
