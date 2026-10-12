"""Console and module entry points share one command implementation."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from .config import load_config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def doctor():
    import torch
    packages = {}
    for name in ("signpathlab", "torch", "torchvision", "numpy", "PyYAML", "scikit-learn", "pytest", "av", "opencv-python", "opencv-contrib-python", "mediapipe", "sacrebleu", "jiwer"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    def command(args):
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=10)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    cuda = torch.cuda.is_available()
    return {"python": platform.python_version(), "os": platform.platform(), "packages": packages,
            "cuda_available": cuda, "cuda_runtime": torch.version.cuda,
            "bf16_available": cuda and torch.cuda.is_bf16_supported(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
            "driver": command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
            "cuda_toolkit": command(["nvcc", "--version"]), "git_commit": command(["git", "rev-parse", "HEAD"])}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="signpath", description="SignPathLab research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("doctor", help="Record environment and available accelerators")
    inspect.add_argument("--output", required=True)
    config = commands.add_parser("config", help="Resolve and validate configuration")
    validate = config.add_subparsers(dest="operation", required=True).add_parser("validate")
    validate.add_argument("--config", required=True)
    validate.add_argument("--output")
    data = commands.add_parser("data", help="Import official splits and audit manifests")
    operations = data.add_subparsers(dest="operation", required=True)
    ingest = operations.add_parser("import")
    ingest.add_argument("--adapter", choices=["asl_citizen", "local_jsonl"], required=True)
    ingest.add_argument("--root", required=True)
    ingest.add_argument("--output", required=True)
    audit = operations.add_parser("audit")
    audit.add_argument("--manifest-dir", required=True)
    audit.add_argument("--output", required=True)
    assets = commands.add_parser("assets", help="Fetch explicit pretrained assets and save checksum receipts")
    fetch = assets.add_subparsers(dest="operation", required=True).add_parser("fetch")
    fetch.add_argument("--registry", required=True)
    fetch.add_argument("--model", required=True)
    smoke = commands.add_parser("smoke", help="No-download CPU/single-device/DDP correctness training smoke")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--resume")
    prep = commands.add_parser("preprocess", help="Decode/IMAGE landmarks with source-verified resumable shards")
    prep.add_argument("--config", required=True)
    prep.add_argument("--split", choices=["train", "val", "test"], required=True)
    prep.add_argument("--stage", choices=["pose_roi"], default="pose_roi")
    prep.add_argument("--shards", type=int, default=1)
    prep.add_argument("--assets-file")
    cache = commands.add_parser("cache", help="Build verified per-frame feature corpus")
    build = cache.add_subparsers(dest="operation", required=True).add_parser("build")
    build.add_argument("--config", required=True)
    build.add_argument("--split", choices=["train", "val", "test"], required=True)
    build.add_argument("--device", default="cpu")
    build.add_argument("--assets-file")
    targets = commands.add_parser("targets", help="Fit train-only geometry transforms or hidden-unit codebooks")
    fit = targets.add_subparsers(dest="operation", required=True).add_parser("fit")
    fit.add_argument("--config", required=True)
    fit.add_argument("--kind", choices=["geometry", "kmeans"], required=True)
    fit.add_argument("--split", default="train")
    verify = commands.add_parser("verify", help="Validate corpus/split/closure gates before experiments")
    pipeline = verify.add_subparsers(dest="operation", required=True).add_parser("pipeline")
    pipeline.add_argument("--config", required=True)
    pipeline.add_argument("--output")
    train = commands.add_parser("train", help="Train a real cached corpus with registered validation probe selection")
    train.add_argument("--config", required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--resume")
    downstream = commands.add_parser("downstream", help="Fit a supervised downstream head using train labels")
    head = downstream.add_subparsers(dest="operation", required=True).add_parser("fit")
    head.add_argument("--config", required=True)
    head.add_argument("--encoder")
    head.add_argument("--device", default="cpu")
    evaluate = commands.add_parser("evaluate", help="Evaluate a validation-selected locked head")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="val")
    evaluate.add_argument("--export-predictions", action="store_true")
    evaluate.add_argument("--device", default="cpu")
    profiling = commands.add_parser("profile", help="Measure actual cached training steps including teacher/optimizer")
    profiling.add_argument("--config", required=True)
    profiling.add_argument("--steps", type=int, default=20)
    profiling.add_argument("--output", required=True)
    profiling.add_argument("--device", default="cpu")
    experiments = commands.add_parser("experiments", help="Run declared trials or summarize actual result artifacts")
    experiment_ops = experiments.add_subparsers(dest="operation", required=True)
    summary = experiment_ops.add_parser("summarize")
    summary.add_argument("--root", required=True)
    summary.add_argument("--output", required=True)
    run_experiments = experiment_ops.add_parser("run")
    run_experiments.add_argument("--configs", nargs="+", required=True)
    run_experiments.add_argument("--device", default="cpu")
    run_experiments.add_argument("--output", required=True)
    embed = commands.add_parser("embed", help="Actual raw-video to timestamped offline embeddings")
    embed.add_argument("--run", required=True)
    embed.add_argument("--input", required=True)
    embed.add_argument("--output", required=True)
    embed.add_argument("--assets-file")
    embed.add_argument("--device", default="cpu")
    export = commands.add_parser("export", help="Export selected unmasked encoder/head with dynamic lengths and parity")
    export.add_argument("--run", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--format", default="torch_export")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor()
            write_json(args.output, result)
        elif args.command == "config":
            result = load_config(args.config).provenance()
            if args.output:
                write_json(args.output, result)
        elif args.command == "assets":
            from .assets import fetch_asset
            result = fetch_asset(args.registry, args.model)
        elif args.command == "smoke":
            from .training.runner import run_smoke
            result = run_smoke(load_config(args.config), args.device, args.resume)
        elif args.command == "preprocess":
            from .preprocessing import preprocess
            result = preprocess(load_config(args.config), args.split, args.assets_file, args.shards)
        elif args.command == "cache":
            from .preprocessing import build_cache
            result = build_cache(load_config(args.config), args.split, args.assets_file, args.device)
        elif args.command == "targets":
            from .targets.fitting import fit_targets
            result = fit_targets(load_config(args.config), args.kind, args.split)
        elif args.command == "verify":
            from .evaluation.protocols import verify_pipeline
            result = verify_pipeline(load_config(args.config), args.output)
        elif args.command == "train":
            from .training.cached_runner import run_cached_training
            result = run_cached_training(load_config(args.config), args.device, args.resume)
        elif args.command == "downstream":
            from .evaluation.islr import downstream_fit
            result = downstream_fit(load_config(args.config), args.encoder, args.device)
        elif args.command == "evaluate":
            from .evaluation.evaluate import evaluate_run
            result = evaluate_run(args.run, args.split, args.export_predictions, args.device)
        elif args.command == "profile":
            from .training.profile import profile
            result = profile(load_config(args.config), args.steps, args.output, args.device)
        elif args.command == "embed":
            from .inference.embed import embed_video
            result = embed_video(args.run, args.input, args.output, assets_file=args.assets_file, device=args.device)
        elif args.command == "export":
            from .inference.export import export_run
            result = export_run(args.run, args.output, args.format)
        elif args.command == "experiments":
            from .experiments import run_experiments, summarize
            if args.operation == "summarize":
                result = summarize(args.root, args.output)
            else:
                result = run_experiments(args.configs, args.device)
                write_json(args.output, result)
        elif args.operation == "import":
            from .data.adapters import asl_citizen, local_jsonl
            result = (asl_citizen if args.adapter == "asl_citizen" else local_jsonl).ingest(args.root, args.output)
        else:
            from .data.audit import audit_directory
            result = audit_directory(args.manifest_dir)
            write_json(Path(args.output) / "audit.json", result)
            if not result["passed"]:
                raise ValueError(f"Split audit failed; inspect {Path(args.output) / 'audit.json'}")
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f"signpath: {error}\n")


if __name__ == "__main__":
    main()
