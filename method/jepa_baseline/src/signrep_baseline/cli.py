"""Auditable entry points for the reconstructed SignRep baseline."""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import yaml
from .cache import sha256
from .data import read_manifest

ROOT = Path(__file__).resolve().parents[2]
PATH_KEYS = {"manifest", "output", "checkpoint", "validation_gallery", "validation_queries",
             "train_manifest", "validation_manifest", "class_distribution"}


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    for key in PATH_KEYS:
        if config.get(key) is not None:
            candidate = Path(config[key])
            config[key] = str((path.parent / candidate).resolve() if not candidate.is_absolute() else candidate)
    return config


def environment_record():
    packages = {}
    for name in ["torch", "torchvision", "timm", "numpy", "albumentations", "opencv-python", "PyYAML"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    return {"python": sys.version, "executable": sys.executable, "packages": packages,
            "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available()}


def load_model(checkpoint, device):
    from .model import SignRep
    model = SignRep()
    model.load_released(checkpoint)
    return model.to(device).eval()


def held_out_contract(gallery, queries, split="val", require_wlasl=False):
    left, right = read_manifest(gallery, "train"), read_manifest(queries, split)
    if {r["video_id"] for r in left} & {r["video_id"] for r in right}:
        raise ValueError("Training dictionary overlaps query split")
    for key in ["video", "video_sha256", "youtube_id"]:
        left_keys = {row[key] for row in left if row.get(key)}
        right_keys = {row[key] for row in right if row.get(key)}
        if left_keys & right_keys:
            raise ValueError(f"Gallery/query overlap in {key}")
    from .data import source_identity
    local_left = {source_identity(row) for row in left if row.get("source_id") and row.get("dataset")}
    local_right = {source_identity(row) for row in right if row.get("source_id") and row.get("dataset")}
    if local_left & local_right:
        raise ValueError("Gallery/query overlap in source_id")
    if require_wlasl and any(row.get("dataset") != "wlasl" for row in left + right):
        raise ValueError("Paper pretraining selection requires WLASL validation retrieval")
    return left, right


def evaluate_live(model, gallery, queries, weighted=True, activity_mode="raw", split="val", require_wlasl=False):
    from .retrieval import extract_windows, pool_windows, dictionary_retrieval
    left, right = held_out_contract(gallery, queries, split, require_wlasl)
    features = []
    for rows in [left, right]:
        pooled = []
        for row in rows:
            result = extract_windows(model, row["video"], bbox=row.get("signer_bbox_xyxy"))
            pooled.append(pool_windows(result["features"], result["activity"] if weighted else None, activity_mode))
        features.append(torch.stack(pooled))
    return dictionary_retrieval(features[0], [r["label"] for r in left], features[1], [r["label"] for r in right],
                                [r["video_id"] for r in left], [r["video_id"] for r in right], split)[0]


def cached_feature_rows(manifest, required_split):
    rows = read_manifest(manifest, required_split)
    pins = set()
    for row in rows:
        with np.load(row["features"], allow_pickle=False) as file:
            metadata = json.loads(str(file["metadata_json"]))
        pin = metadata.get("encoder_checkpoint_sha256")
        if not pin:
            raise ValueError("Feature cache missing encoder checkpoint pin")
        if row.get("features_sha256") and sha256(row["features"]) != row["features_sha256"]:
            raise ValueError("Frozen feature cache changed")
        if row.get("video_sha256") and metadata["video_sha256"] != row["video_sha256"]:
            raise ValueError("Feature cache belongs to a different video")
        pins.add(pin)
    if len(pins) != 1:
        raise ValueError("Feature caches use different encoder checkpoints")
    return rows, next(iter(pins))


def run_pretrain(args):
    from .schema import FidelityChoices
    from .augmentation import VideoTransform, AugmentationConfig
    from .data import PairedVideoDataset, StatefulPairSampler, collate_pairs
    from .model import SignRep
    from .training import Pretrainer, seed_everything
    from .preparation import validate_pretrain_audit
    config = load_config(args.config)
    choices = FidelityChoices(**config.get("choices", {}))
    paper = config.get("scope") == "paper_recipe"
    if paper:
        for key, expected in [("total_steps", 500000), ("warmup_steps", 50000), ("max_lr", 1e-4), ("batch_size", 20), ("validation_interval", 25000)]:
            if config.get(key) != expected:
                raise ValueError(f"Paper recipe changed: {key}; label scope technical/ablation explicitly")
    audit = validate_pretrain_audit(config["manifest"])
    if paper and audit.get("audit_kind") == "local_train_only":
        raise ValueError("Local-corpus adaptation must not be labeled paper_recipe")
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_sources.py")], check=True, capture_output=True)
    held_out_contract(config["validation_gallery"], config["validation_queries"], "val", paper)
    seed_everything(config["seed"])
    dataset = PairedVideoDataset(config["manifest"], VideoTransform(True, AugmentationConfig(**config.get("augmentation", {}))), choices)
    views = config["batch_size"]
    if choices.pretrain_batch_counts == "total_views" and views % 2:
        raise ValueError("Total-view batch must be even")
    pairs = views // 2 if choices.pretrain_batch_counts == "total_views" else views
    sampler = StatefulPairSampler(len(dataset), pairs, config["seed"])
    model = SignRep(mask_ratio=.8)
    if not args.resume:
        if config["initialization"] == "kinetics_mae":
            model.initialize_backbone(config["checkpoint"])
        elif config["initialization"] == "released_signrep" and not paper:
            model.load_released(config["checkpoint"])
        else:
            raise ValueError("Paper pretraining initializes from Kinetics MAE")
    trainer = Pretrainer(model, choices, config["device"], config["total_steps"], config["warmup_steps"], config["max_lr"], config.get("amp", False), config.get("amp_dtype", "float16"))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    contract = {"config": config, "manifest_sha256": sha256(config["manifest"]),
                "audit_sha256": sha256(Path(config["manifest"]).with_suffix(".jsonl.audit.json")),
                "validation_gallery_sha256": sha256(config["validation_gallery"]),
                "validation_queries_sha256": sha256(config["validation_queries"]),
                "initial_checkpoint_sha256": sha256(config["checkpoint"]),
                "source_verification_sha256": sha256(ROOT / "evidence/source_verification.json"),
                "environment": environment_record()}
    if args.resume:
        trainer.resume(args.resume, sampler, contract)
    (output / "run_contract.json").write_text(json.dumps(contract, indent=2))
    limit = min(config["total_steps"], args.stop_after or config["total_steps"])
    if limit <= trainer.step:
        raise ValueError("Stop-after must exceed resumed completed step")
    while trainer.step < limit:
        batch = collate_pairs([dataset[i] for i in sampler.next_indices()])
        metrics = trainer.update(batch)
        with (output / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(metrics) + "\n")
        if trainer.step % config["validation_interval"] == 0:
            result = evaluate_live(trainer.model, config["validation_gallery"], config["validation_queries"],
                                   config.get("weighted_retrieval", True), choices.activity_weighting,
                                   require_wlasl=paper)
            trainer.select_validation(result, output / "best.pt", sampler, contract, config["validation_interval"])
            with (output / "validation.jsonl").open("a") as file:
                file.write(json.dumps({"step": trainer.step, **result}) + "\n")
        if trainer.step % config["save_interval"] == 0 or trainer.step == limit:
            trainer.save(output / "last.pt", sampler, contract)
        if trainer.step == 1 or trainer.step % 100 == 0 or trainer.step == limit:
            print(json.dumps(metrics), flush=True)
    return {"completed_steps": trainer.step, "requested_recipe_steps": config["total_steps"],
            "checkpoint": str(output / "last.pt"), "score_reproduction": "not established"}


def run_recognition(args):
    from .model import SignRecognizer
    from .recognition import RecognitionDataset, load_class_distribution
    from .recognition_training import RecognitionTrainer
    from .data import StatefulPairSampler
    from .training import seed_everything
    config = load_config(args.config)
    if config.get("scope") == "paper_recipe" and (config["epochs"] != 100 or config["batch_size"] != 8 or config["warmup_steps"] != 1000):
        raise ValueError("Recognition paper recipe changed; label ablation/technical scope explicitly")
    train = RecognitionDataset(config["train_manifest"], True, config["short_policy"])
    valid = RecognitionDataset(config["validation_manifest"], False, config["short_policy"])
    held_out_contract(config["train_manifest"], config["validation_manifest"])
    distribution, _ = load_class_distribution(config["class_distribution"], config["train_manifest"], sha256(config["checkpoint"]))
    if distribution.shape != (config["classes"], config["classes"]):
        raise ValueError("Configured recognition classes differ from train-derived table")
    seed_everything(config["seed"])
    model = SignRecognizer(load_model(config["checkpoint"], "cpu").inflate_for_recognition(), config["classes"])
    batches_per_epoch = len(train) // config["batch_size"]
    total = batches_per_epoch * config["epochs"]
    sampler = StatefulPairSampler(len(train), config["batch_size"], config["seed"])
    trainer = RecognitionTrainer(model, distribution, total, config["warmup_steps"], config["device"], config.get("amp", False))
    contract = {"config": config, "train_manifest_sha256": sha256(config["train_manifest"]),
                "validation_manifest_sha256": sha256(config["validation_manifest"]),
                "class_table_sha256": sha256(config["class_distribution"]),
                "initial_checkpoint_sha256": sha256(config["checkpoint"]), "environment": environment_record()}
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        trainer.resume(args.resume, sampler, contract)
    (output / "run_contract.json").write_text(json.dumps(contract, indent=2))
    limit = min(total, args.stop_after or total)
    while trainer.step < limit:
        items = [train[i] for i in sampler.next_indices()]
        metrics = trainer.update(torch.stack([v[0] for v in items]), torch.stack([v[1] for v in items]))
        with (output / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(metrics) + "\n")
        if trainer.step % batches_per_epoch == 0:
            result = trainer.evaluate(valid)
            trainer.select_validation(result, output / "best.pt", sampler, contract)
            with (output / "validation.jsonl").open("a") as file:
                file.write(json.dumps({"step": trainer.step, **result}) + "\n")
        if trainer.step % batches_per_epoch == 0 or trainer.step == limit:
            trainer.save(output / "last.pt", sampler, contract)
        if trainer.step == 1 or trainer.step % 100 == 0 or trainer.step == limit:
            print(json.dumps(metrics), flush=True)
    return {"completed_steps": trainer.step, "configured_total_steps": total, "checkpoint": str(output / "last.pt")}


def main():
    parser = argparse.ArgumentParser(description="SignRep paper-grounded module reconstruction (historical unknowns documented)")
    parser.add_argument("--threads", type=int, default=2)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("environment")
    commands.add_parser("verify-sources")
    assets = commands.add_parser("download-assets")
    assets.add_argument("--kind", choices=["all", "released", "kinetics", "teacher"], default="all")
    assets.add_argument("--destination")
    pose = commands.add_parser("pose")
    pose.add_argument("operation", choices=["inspect", "extract", "normalize"])
    pose.add_argument("--python", required=True)
    pose.add_argument("--weights", required=True)
    pose.add_argument("--body-root-mode", choices=["canonical", "camera"], default="canonical")
    for name in ["video", "raw", "stats", "output"]:
        pose.add_argument("--" + name)
    prepare = commands.add_parser("prepare-corpus")
    for name in ["manifest", "output", "pose-python", "weights"]:
        prepare.add_argument("--" + name, required=True)
    protocol = prepare.add_mutually_exclusive_group(required=True)
    protocol.add_argument("--wlasl")
    protocol.add_argument("--held-out-manifest")
    prepare.add_argument("--training-reference-manifest")
    bones = commands.add_parser("bone-stats")
    bones.add_argument("--manifest", required=True, help="deduplicated corpus manifest with raw_cache paths")
    bones.add_argument("--output", required=True)
    for command in ["pretrain", "recognition"]:
        train = commands.add_parser(command)
        train.add_argument("--config", required=True)
        train.add_argument("--resume")
        train.add_argument("--stop-after", type=int, help="absolute completed-step limit for technical smoke/interruption")
    recognition_eval = commands.add_parser("recognition-eval")
    for name in ["checkpoint", "manifest", "output"]:
        recognition_eval.add_argument("--" + name, required=True)
    recognition_eval.add_argument("--split", choices=["val", "test"], default="test")
    recognition_eval.add_argument("--device", default="cpu")
    recognition_eval.add_argument("--batch-size", type=int, default=8)
    recognition_eval.add_argument("--short-policy", choices=["reject", "repeat_last"], default="repeat_last")
    translation_run = commands.add_parser("sign2gpt-train")
    translation_run.add_argument("--config", required=True, help="JSON described in docs/TRANSLATION.md")
    translation_run.add_argument("--python", required=True, help="isolated Sign2GPT environment interpreter")
    translation_run.add_argument("--inspect", action="store_true")
    features = commands.add_parser("features")
    for name in ["manifest", "checkpoint", "output"]:
        features.add_argument("--" + name, required=True)
    features.add_argument("--device", default="cpu")
    features.add_argument("--batch-size", type=int, default=2)
    features.add_argument("--stride", type=int, default=2)
    features.add_argument("--short-policy", choices=["reject", "repeat_last"], default="reject")
    retrieval = commands.add_parser("retrieval")
    for name in ["gallery", "queries", "output"]:
        retrieval.add_argument("--" + name, required=True)
    retrieval.add_argument("--split", choices=["val", "test"], default="test")
    retrieval.add_argument("--unweighted", action="store_true")
    retrieval.add_argument("--activity-mode", choices=["raw", "clamp_0_1"], default="raw")
    table = commands.add_parser("class-distribution")
    table.add_argument("--train-manifest", required=True)
    table.add_argument("--classes", type=int, required=True)
    table.add_argument("--output", required=True)
    table.add_argument("--unweighted", action="store_true")
    table.add_argument("--exclude-self", action="store_true")
    translation = commands.add_parser("how2sign-export")
    for name in ["manifest", "output-tsv", "feature-directory"]:
        translation.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    result = None
    if args.command == "environment":
        result = environment_record()
    elif args.command == "verify-sources":
        subprocess.run([sys.executable, str(ROOT / "scripts/verify_sources.py")], check=True)
    elif args.command == "download-assets":
        from .assets import download_assets
        result = download_assets(args.kind, args.destination)
    elif args.command == "pose":
        from .preparation import pose_command
        print(pose_command(args.python, args.operation, args.weights,
                           **{k: getattr(args, k) for k in ["video", "raw", "stats", "output", "body_root_mode"] if getattr(args, k)}))
    elif args.command == "prepare-corpus":
        from .preparation import prepare_corpus
        result = prepare_corpus(args.manifest, args.wlasl, args.output, args.pose_python, args.weights,
                                args.held_out_manifest, args.training_reference_manifest)
    elif args.command == "bone-stats":
        from .bones import corpus_bone_statistics
        rows = read_manifest(args.manifest, "pretrain")
        result = corpus_bone_statistics([r["raw_cache"] for r in rows], args.output, sha256(args.manifest))
    elif args.command == "pretrain":
        result = run_pretrain(args)
    elif args.command == "recognition":
        result = run_recognition(args)
    elif args.command == "recognition-eval":
        from .model import SignRep, SignRecognizer
        from .recognition import RecognitionDataset
        from .recognition_training import RecognitionTrainer
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if state.get("schema") != "signrep_recognition_training_v1":
            raise ValueError("Expected complete reconstructed recognition checkpoint")
        classes = state["model"]["classifier.weight"].shape[0]
        model = SignRecognizer(SignRep().inflate_for_recognition(), classes)
        model.load_state_dict(state["model"], strict=True)
        trainer = RecognitionTrainer(model, state["distribution"], total_steps=1001,
                                     warmup_steps=1000, device=args.device)
        dataset = RecognitionDataset(args.manifest, False, args.short_policy)
        result = trainer.evaluate(dataset, args.split, args.batch_size)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
    elif args.command == "sign2gpt-train":
        command = [args.python, str(ROOT / "scripts/run_sign2gpt.py"), "--config", str(Path(args.config).resolve())]
        if args.inspect:
            command.append("--inspect")
        subprocess.run(command, check=True)
    elif args.command == "features":
        from .retrieval import extract_windows
        model = load_model(args.checkpoint, args.device)
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        rows = read_manifest(args.manifest)
        checkpoint_pin = sha256(args.checkpoint)
        for index, row in enumerate(rows):
            values = extract_windows(model, row["video"], args.stride, args.batch_size,
                                     row.get("signer_bbox_xyxy"), args.short_policy)
            metadata = json.loads(str(values["metadata_json"]))
            metadata.update(encoder_checkpoint_sha256=checkpoint_pin, checkpoint=str(Path(args.checkpoint).resolve()),
                            inference_preprocessing="immutable public author Transformation")
            values["metadata_json"] = np.array(json.dumps(metadata))
            path = output / f"{index:08d}.npz"
            np.savez_compressed(path, **values)
            row["features"] = str(path)
            row["features_sha256"] = sha256(path)
        manifest = output / "features.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        result = {"manifest": str(manifest), "videos": len(rows), "checkpoint_sha256": checkpoint_pin}
    elif args.command == "retrieval":
        from .retrieval import dictionary_retrieval, feature_record
        left, lpin = cached_feature_rows(args.gallery, "train")
        right, rpin = cached_feature_rows(args.queries, args.split)
        if lpin != rpin:
            raise ValueError("Dictionary and query features use different encoders")
        gallery = torch.stack([feature_record(r, not args.unweighted, args.activity_mode) for r in left])
        queries = torch.stack([feature_record(r, not args.unweighted, args.activity_mode) for r in right])
        result = dictionary_retrieval(gallery, [r["label"] for r in left], queries, [r["label"] for r in right],
                                     [r["video_id"] for r in left], [r["video_id"] for r in right], args.split)[0]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
    elif args.command == "class-distribution":
        from .retrieval import feature_record
        from .recognition import save_class_distribution
        rows, pin = cached_feature_rows(args.train_manifest, "train")
        features = torch.stack([feature_record(row, not args.unweighted) for row in rows])
        result = save_class_distribution(args.output, features, [r["label"] for r in rows], args.classes,
                                         args.train_manifest, include_self=not args.exclude_self,
                                         encoder_checkpoint_sha256=pin)
    elif args.command == "how2sign-export":
        from .translation import export_how2sign
        result = export_how2sign(args.manifest, args.output_tsv, args.feature_directory)
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
