"""Paired real-video CPU-thread/DINO-batch timings including extraction and disk writes."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from signpath.cache.schema import ARRAY_DTYPES
from signpath.cache.reader import read_bundle
from signpath.cli import write_json
from signpath.config import ResolvedConfig, load_config
from signpath.cache.hashing import canonical_hash
from signpath.data.manifest import read_manifest, write_manifest
from signpath.preprocessing import asset_plan
from signpath.preprocessing_full import cache_adapter, build_full_cache


def compare_caches(first, second):
    a = json.loads((first / "index.json").read_text())
    b = json.loads((second / "index.json").read_text())
    assert a["samples"].keys() == b["samples"].keys()
    max_error, exact_visual, other_exact, supports_equal = 0., True, True, True
    changed = total = 0
    for name in a["samples"]:
        x = read_bundle(first / a["samples"][name]["path"], a["identity"])
        y = read_bundle(second / b["samples"][name]["path"], b["identity"])
        for field in ARRAY_DTYPES:
            p, q = getattr(x, field), getattr(y, field)
            if field == "visual":
                error = float(np.abs(p.astype(np.float32) - q.astype(np.float32)).max())
                max_error = max(max_error, error)
                exact_visual &= np.array_equal(p, q)
                changed += int(np.count_nonzero(p != q)); total += p.size
            else:
                other_exact &= np.array_equal(p, q)
        supports_equal &= x.raw_supports == y.raw_supports
        for field in ("roi_affines", "roi_boxes", "roi_padding_fractions", "global_full_frame_fallback"):
            other_exact &= x.meta[field] == y.meta[field]
    return {"visual_exact": bool(exact_visual), "visual_max_absolute_error_fp16": max_error,
        "visual_changed_fraction": changed / total, "nonvisual_arrays_and_rois_exact": bool(other_exact),
        "supports_exact": bool(supports_equal)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/aslc3000/cache_benchmark")
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.clips < 4 or args.repeats < 1:
        raise ValueError("Paired benchmark needs at least4clips and one repeat")
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=False)
    original = load_config("configs/aslc3000/E04_seed17.yaml")
    rows = read_manifest(original.path(original.values["data"]["train_manifest"]))
    # Declared source-uniform sample chosen before candidate timings; avoid the
    # first10same-gloss clips that dominated initial startup extrapolation.
    rng = np.random.default_rng(2718)
    chosen = [rows[i] for i in sorted(rng.choice(len(rows), args.clips, replace=False).tolist())]
    write_manifest(root / "train.jsonl", chosen)
    write_json(root / "plan.json", {"seed": 2718, "sources": [r.sample_id for r in chosen],
        "options": [[1,16],[2,16],[1,32],[1,64],[2,32],[2,64]], "repeats": args.repeats,
        "scope": "cache throughput/correctness only; no model score or training selection"})
    options = [(1,16),(2,16),(1,32),(1,64),(2,32),(2,64)]
    adapters = {batch: cache_adapter(asset_plan(original, "registries/local_extraction.yaml", include_dino=True), "cuda:0", batch)
                for batch in (16,32,64)}
    # Warm each numerical batch recipe before throughput comparison.
    for batch, adapter in adapters.items():
        adapter.encode_crops(np.zeros((batch,224,224,3), dtype=np.uint8))
    torch.cuda.synchronize()
    reports = []
    for repeat in range(args.repeats):
        order = list(options) if repeat == 0 else list(reversed(options))
        for workers, batch in order:
            name = f"repeat{repeat}_threads{workers}_batch{batch}"
            directory = root / name
            config = json.loads(json.dumps(original.values))
            config["data"]["train_manifest"] = str(root / "train.jsonl")
            config["cache"]["root"] = str(directory / "cache")
            resolved = ResolvedConfig(config, original.root, (), canonical_hash(config))
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            result = build_full_cache(resolved, "train", "registries/local_extraction.yaml", "cuda:0",
                workers=workers, dino_batch_size=batch, adapter=adapters[batch])
            elapsed = time.monotonic() - started
            baseline = root / f"repeat{repeat}_threads1_batch16/cache"
            # On the reverse-order repeat, reference is available after all candidates.
            report = {"candidate": name, "repeat": repeat, "threads": workers, "dino_batch_size": batch,
                "real_clips": len(chosen), "elapsed_seconds": elapsed, "seconds_per_clip": elapsed / len(chosen),
                "gpu_peak_bytes": torch.cuda.max_memory_allocated(), "stage": result,
                "cache": str(directory / "cache")}
            reports.append(report)
            write_json(root / "progress.json", {"status": "running", "completed_options": len(reports), "reports": reports})
            print({k: report[k] for k in ("candidate", "elapsed_seconds", "seconds_per_clip", "gpu_peak_bytes")}, flush=True)
        for report in [r for r in reports if r["repeat"] == repeat]:
            report["parity"] = compare_caches(root / f"repeat{repeat}_threads1_batch16/cache", Path(report["cache"]))
    aggregates = []
    baseline_time = statistics.mean(r["elapsed_seconds"] for r in reports if r["threads"]==1 and r["dino_batch_size"]==16)
    for workers, batch in options:
        group = [r for r in reports if r["threads"]==workers and r["dino_batch_size"]==batch]
        mean = statistics.mean(r["elapsed_seconds"] for r in group)
        aggregates.append({"threads":workers,"dino_batch_size":batch,"seconds_mean":mean,
            "speedup_over_baseline":baseline_time/mean,"repeat_seconds":[r["elapsed_seconds"] for r in group],
            "nonvisual_and_supports_exact":all(r["parity"]["nonvisual_arrays_and_rois_exact"] and r["parity"]["supports_exact"] for r in group),
            "visual_exact":all(r["parity"]["visual_exact"] for r in group),
            "visual_max_error":max(r["parity"]["visual_max_absolute_error_fp16"] for r in group)})
    report = {"status":"completed", "scope":"paired real-video cache benchmark; not training/retrieval metrics",
        "sample_count":len(chosen), "repeats":args.repeats,"python_processes":1,
        "aggregates":aggregates,"reports":reports}
    write_json(root / "result.json", report)
    print(json.dumps({"status":"completed","aggregates":aggregates},indent=2),flush=True)


if __name__ == "__main__":
    main()
