"""Lock a small official-split/availability pilot before examining model scores."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import random
import yaml

from signpath.cache.hashing import file_hash
from signpath.cli import write_json
from signpath.data.audit import audit_records
from signpath.data.manifest import ManifestRecord, write_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--train-per-class", type=int, default=4)
    parser.add_argument("--val-per-class", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=137)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = Path(args.dataset).resolve()
    tables, hashes, grouped = {}, {}, {}
    for split in ("train", "val"):
        path = dataset / "splits" / f"{split}.csv"
        hashes[split] = file_hash(path)
        with path.open(newline="") as stream:
            tables[split] = list(csv.DictReader(stream))
        grouped[split] = defaultdict(list)
        for row in tables[split]:
            if (dataset / "videos" / row["Video file"]).is_file():
                grouped[split][row["Gloss"]].append(row)
    taxonomy = {label: i for i, label in enumerate(sorted({r["Gloss"] for r in tables["train"]}))}
    eligible = sorted(label for label in taxonomy if len(grouped["train"][label]) >= args.train_per_class and len(grouped["val"][label]) >= args.val_per_class)
    rng = random.Random(args.selection_seed)
    selected = sorted(rng.sample(eligible, args.classes))
    manifests = root / "data/pilot8/manifests"
    manifests.mkdir(parents=True, exist_ok=False)
    all_records, counts = [], {}
    for split, per_class in [("train", args.train_per_class), ("val", args.val_per_class)]:
        records = []
        for label in selected:
            rows = grouped[split][label].copy()
            rng.shuffle(rows)
            for row in rows[:per_class]:
                video = dataset / "videos" / row["Video file"]
                name = "asl_citizen:" + video.stem
                records.append(ManifestRecord(1, name, "asl_citizen", "ase", split, name, str(video), file_hash(video),
                    signer_id=row["Participant ID"], label_id=taxonomy[label], gloss=label,
                    official_split_version="csv_sha256:" + hashes[split], availability="available"))
        write_manifest(manifests / f"{split}.jsonl", records)
        all_records.extend(records)
        counts[split] = len(records)
    audit = audit_records(all_records, verify_files=True)
    if not audit["passed"]:
        raise ValueError(f"Selected official pilot is not split-safe: {audit}")
    plan = {"scope": "label-curated small official-split development pilot; not full ASL Citizen benchmark",
        "selection_seed": args.selection_seed, "selected_classes": selected, "class_ids": [taxonomy[c] for c in selected],
        "counts": counts, "csv_sha256": hashes, "manifest_sha256": {s: file_hash(manifests / f"{s}.jsonl") for s in counts},
        "selection_uses": "official train/val labels and actual video availability; no model scores; disclosed class-curated pool",
        "SSL_sampler": "uniform source then contiguous segment; labels absent from SSL losses and sampling",
        "test_accessed": False, "seeds": [17, 29, 43], "successful_steps": 100, "microbatch": 4,
        "primary_metric": "validation per-class Top1", "primary_comparison": "E03 vs E04",
        "minimum_useful_effect_percentage_points": .5,
        "head_budget": {"learning_rates": [.001, .003, .01], "max_epochs": 40, "patience": 10}, "audit": audit}
    write_json(root / "data/pilot8/plan.json", plan)
    configs = root / "configs/pilot"
    configs.mkdir(exist_ok=True)
    base = {"extends": "../base.yaml", "experiment": {"id": "pilot8", "evaluation_protocol": "asl_citizen_official_signer_disjoint_pilot8_v1"},
        "data": {"train_manifest": "data/pilot8/manifests/train.jsonl", "val_manifest": "data/pilot8/manifests/val.jsonl",
                 "test_manifest": "data/pilot8/manifests/test.jsonl", "num_workers": 4},
        "cache": {"root": "data/pilot8/cache"}, "geometry": {"projection": {"statistics_path": "data/pilot8/targets/logsig_stats.npz", "matrices_path": "data/pilot8/targets/logsig_matrices.npz"}},
        "model": {"embed_dim": 96, "projection_hidden_dim": 96, "encoder": {"layers": 2, "heads": 4, "return_layers": [1, 2]}, "predictor": {"embed_dim": 96, "heads": 4, "layers": 1}},
        "training": {"max_successful_steps": 100, "microbatch_per_gpu": 4, "gradient_accumulation": 1, "validation_interval": 100, "checkpoint_interval": 50, "diagnostic_interval": 50},
        "downstream": {"max_epochs": 40, "patience": 10, "batch_size": 32}, "runtime": {"max_runtime_hours": 1.0}}
    (configs / "base.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    for experiment in ["E00", "E01", "E02", "E03", "E04"]:
        for seed in plan["seeds"]:
            c = {"extends": "base.yaml", "experiment": {"id": experiment, "seed": seed, "output_dir": f"artifacts/pilot8/{experiment}/seed{seed}"},
                 "downstream": {"output_dir": f"artifacts/pilot8/heads/{experiment}/seed{seed}"}}
            if experiment in {"E00", "E01", "E02"}:
                c.update(geometry={"enabled": False}, loss={"geometry_weight": 0.0})
            if experiment == "E00":
                c["downstream"]["representation"] = "frozen_visual"
            if experiment == "E01":
                c["loss"].update(semantic_weight=0., cluster_weight=1.)
            if experiment == "E03":
                c["geometry"] = {"descriptor": "ordinary_motion_anchor_timing", "projection": {
                    "statistics_path": "data/pilot8/targets/motion_stats.npz", "matrices_path": "data/pilot8/targets/motion_matrices.npz"}}
            (configs / f"{experiment}_seed{seed}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    print(f"Locked {counts} on {len(selected)} classes; configs/pilot; no test access")


if __name__ == "__main__":
    main()
