"""Fair-budget runs and source-linked summaries; never fill missing metrics with zero."""
import csv
import json
from pathlib import Path
import statistics

from .cache.hashing import canonical_hash, file_hash
from .cli import write_json
from .config import load_config


def paired_prediction_intervals(first_run, second_run):
    from .evaluation.bootstrap import paired_classification_bootstrap
    records = []
    for directory in (first_run, second_run):
        path = Path(directory) / "predictions.val.jsonl"
        if not path.exists():
            return {"status": "not_run", "reason": "versioned predictions.val.jsonl required for paired uncertainty"}
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mapping = {r["sample_id"]: r for r in rows}
        if len(mapping) != len(rows):
            raise ValueError("Duplicate prediction IDs in paired comparison")
        records.append(mapping)
    first, second = records
    if not first or first.keys() != second.keys():
        raise ValueError("Paired prediction sample pools differ")
    ids = sorted(first)
    for name in ids:
        if any(first[name][key] != second[name][key] for key in ("reference_label_id", "source_group_id", "signer_id")):
            raise ValueError("Paired prediction reference/source/signer alignment differs")
    signer_groups = [first[name]["signer_id"] for name in ids]
    use_signers = all(signer_groups) and len(set(signer_groups)) > 1
    group_key = "signer_id" if use_signers else "source_group_id"
    outcomes = [[int(pool[name]["prediction_label_id"] == pool[name]["reference_label_id"]) for name in ids] for pool in (first, second)]
    return paired_classification_bootstrap(*outcomes, [first[name]["reference_label_id"] for name in ids],
        [first[name][group_key] for name in ids], resampling_unit=group_key)


def run_experiments(configs, device="cpu"):
    from .evaluation.evaluate import evaluate_run
    from .evaluation.islr import downstream_fit
    from .training.cached_runner import run_cached_training
    from .training.runner import runtime_implementation_hash
    reports = []
    for path in configs:
        resolved = load_config(path)
        c = resolved.values
        try:
            if c["downstream"]["representation"] == "frozen_visual":
                probe = downstream_fit(resolved, device=device)
            else:
                training = run_cached_training(resolved, device=device)
                if training["status"] != "completed":
                    reports.append({"config": str(path), "status": training["status"], "run": training["run_dir"]})
                    continue
                checkpoint = Path(training["run_dir"]) / "checkpoints/best.pt"
                probe = downstream_fit(resolved, checkpoint, device=device)
            evaluation = evaluate_run(probe["output"], "val", True, device)
            reports.append({"config": str(path), "status": "completed", "head_run": probe["output"], "evaluation": evaluation})
        except Exception as error:
            reports.append({"config": str(path), "status": "failed", "reason": str(error)})
            # Preserve negative/failed trials, then continue independent declared experiments.
    return reports


def summarize(root, output):
    root, output = Path(root), Path(output)
    rows = []
    for metrics in sorted(root.rglob("metrics.val.json")):
        run = metrics.parent
        report = json.loads(metrics.read_text())
        config_file = run / "resolved_config.json"
        if not config_file.exists():
            continue
        resolved = json.loads(config_file.read_text())
        c = resolved["config"]
        if "downstream" not in c:
            continue
        evaluation = run / "evaluation.val.json"
        evaluated = json.loads(evaluation.read_text()) if evaluation.exists() else {}
        training = None
        import torch
        head = torch.load(run / "head.pt", map_location="cpu", weights_only=False)
        if head["encoder_path"]:
            path = Path(head["encoder_path"]).parent.parent / "summary.json"
            if path.exists():
                training = json.loads(path.read_text())
        row = {"run_id": str(run), "experiment": c["experiment"]["id"], "seed": c["experiment"]["seed"],
            "status": report["status"], "task": "islr", "split": "val", "head": c["downstream"]["head"],
            "representation": c["downstream"]["representation"], "descriptor": c["geometry"]["descriptor"] if c["geometry"]["enabled"] else None,
            "protocol": c["experiment"]["evaluation_protocol"], "train_corpus": report["identity"]["train"], "val_corpus": report["identity"]["val"],
            "encoder_hash": head["encoder_hash"], "head_sha256": file_hash(run / "head.pt"),
            "per_instance_top1": report["metrics"]["per_instance_top1"], "per_class_top1": report["metrics"]["per_class_top1"],
            "top5": report["metrics"]["top5"], "recall_at_1": evaluated.get("retrieval", {}).get("recall_at_1"),
            "mrr": evaluated.get("retrieval", {}).get("mrr"), "successful_steps": training["successful_steps"] if training else None,
            "train_seconds": training["total_training_seconds"] if training else None, "test_used_for_selection": False,
            "missing_metric_reason": "retrieval_not_evaluated" if not evaluated else None}
        ds_budget = {k: v for k, v in c["downstream"].items() if k != "output_dir"}
        row["comparison_signature"] = canonical_hash({"protocol": row["protocol"], "train": row["train_corpus"],
            "implementation_hash": report["identity"].get("implementation_hash"),
            "world_size": training.get("world_size", 1) if training else None,
            "val": row["val_corpus"], "head_budget": ds_budget, "model": c["model"], "masking": c["masking"],
            "training_budget": {key: c["training"][key] for key in ("max_successful_steps", "microbatch_per_gpu", "gradient_accumulation", "learning_rate", "betas", "weight_decay", "warmup_fraction", "precision")},
            "geometry_schedule": {key: c["geometry"][key] for key in ("role", "window_seconds", "center_stride_samples", "min_distinct_observations")}})
        row["mask_schedule_digest"] = None
        if training:
            logs_path = Path(head["encoder_path"]).parent.parent / "logs.jsonl"
            logs = [json.loads(line) for line in logs_path.read_text().splitlines()]
            row["mask_schedule_digest"] = canonical_hash([r["mask_schedule_hash"] for r in logs])
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["run_id", "status", "missing_metric_reason"]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {}
    for experiment, signature in sorted({(r["experiment"], r["comparison_signature"]) for r in rows}):
        group = [r for r in rows if r["experiment"] == experiment and r["comparison_signature"] == signature]
        scores = [r["per_class_top1"] for r in group]
        aggregate[f"{experiment}:{signature[:8]}"] = {"seeds": [r["seed"] for r in group], "pc_top1_mean": statistics.mean(scores),
            "pc_top1_seed_std": statistics.stdev(scores) if len(scores) > 1 else None, "seed_std_is_not_confidence_interval": True}
    paired = []
    for motion in [r for r in rows if r["experiment"] == "E03"]:
        for signature in [r for r in rows if r["experiment"] == "E04" and r["seed"] == motion["seed"] and r["comparison_signature"] == motion["comparison_signature"]]:
            matched = motion["successful_steps"] == signature["successful_steps"] and motion["mask_schedule_digest"] == signature["mask_schedule_digest"]
            paired.append({"seed": motion["seed"], "matched_successful_steps_and_masks": matched,
                "pc_top1_difference_E04_minus_E03": signature["per_class_top1"] - motion["per_class_top1"] if matched else None,
                "paired_sampling_intervals": paired_prediction_intervals(motion["run_id"], signature["run_id"]) if matched
                    else {"status": "not_run", "reason": "unmatched update or masking budgets"}})
    report = {"rows": rows, "aggregate": aggregate, "primary_E03_E04_pairs": paired,
              "scope": "validation results only; separate groups for incompatible data/model/head/compute protocols"}
    write_json(output.with_suffix(".json"), report)
    return {"status": "completed", "rows": len(rows), "csv": str(output), "aggregate": aggregate}
