"""Real-data calibration and D384 retrieval-only training gate; no main-run scores."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import time
import numpy as np
import yaml
from signpath.cli import write_json
from signpath.config import load_config
from signpath.data.manifest import read_manifest, write_manifest
from signpath.preprocessing_full import build_full_cache
from signpath.targets.fitting import fit_targets
from signpath.evaluation.protocols import verify_pipeline
from signpath.training.cached_runner import run_cached_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/full_aslc/preflight")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    official = Path("data/full_aslc/manifests")
    pools = {split: read_manifest(official / f"asl_citizen.{split}.jsonl") for split in ("train", "val")}
    by_class = {split: {} for split in pools}
    for split, records in pools.items():
        for record in records:
            by_class[split].setdefault(record.label_id, []).append(record)
    labels = sorted(set(by_class["train"]) & set(by_class["val"]))
    chosen = np.random.default_rng(137).choice(labels, 8, replace=False).tolist()
    selected = {}
    for split, count in (("train", 2), ("val", 1)):
        selected[split] = [r for label in chosen for r in by_class[split][label][:count]]
        write_manifest(root / f"{split}.jsonl", selected[split])
    value = {"extends": str(Path("configs/full_aslc/E04_seed17.yaml").resolve()),
        "experiment": {"id": "E04_full_setup_preflight", "output_dir": str(root / "training")},
        "data": {"train_manifest": str(root / "train.jsonl"), "val_manifest": str(root / "val.jsonl")},
        "cache": {"root": str(root / "cache")},
        "geometry": {"projection": {"statistics_path": str(root / "targets/stats.npz"), "matrices_path": str(root / "targets/matrices.npz")}},
        "training": {"max_successful_steps": 3, "validation_interval": 3, "checkpoint_interval": 3, "diagnostic_interval": 3},
        "runtime": {"max_runtime_hours": 1.0}}
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    resolved = load_config(path)
    timings = {}
    for split in ("train", "val"):
        started = time.monotonic()
        build_full_cache(resolved, split, "registries/local_extraction.yaml", "cuda:0", workers=1)
        timings[f"cache_{split}"] = time.monotonic() - started
    started = time.monotonic()
    fit_targets(resolved, "geometry", "train")
    timings["targets"] = time.monotonic() - started
    started = time.monotonic()
    verify_pipeline(resolved, root / "pipeline_checks.json")
    timings["gate"] = time.monotonic() - started
    started = time.monotonic()
    training = run_cached_training(resolved, "cuda:0")
    timings["training_and_val"] = time.monotonic() - started
    logs = [json.loads(line) for line in (Path(training["run_dir"]) / "logs.jsonl").read_text().splitlines()]
    cache_bytes = sum(p.stat().st_size for p in (root / "cache").rglob("*") if p.is_file())
    report = {"status": "passed" if training["status"] == "completed" else training["status"],
        "scope": "24-real-video correctness/calibration only, not final E04 retrieval result",
        "selected_train": len(selected["train"]), "selected_val": len(selected["val"]), "timings_seconds": timings,
        "cache_bytes": cache_bytes, "cache_bytes_per_video": cache_bytes / 24, "training": training,
        "updates": [{k: row[k] for k in ("successful_step", "succeeded", "elapsed_seconds") if k in row} for row in logs]}
    write_json(root / "report.json", report)
    print({"status": report["status"], "report": str(root / "report.json"), "timings": timings}, flush=True)


if __name__ == "__main__":
    main()
