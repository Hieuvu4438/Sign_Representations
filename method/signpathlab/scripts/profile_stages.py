"""Instrument actual cached updates; forward hooks include synchronization overhead."""
import argparse
from pathlib import Path
import time
import torch
from signpath.cli import write_json
from signpath.config import load_config
from signpath.data.dataset import BatchProvider
from signpath.evaluation.protocols import development_corpora
from signpath.models.teacher_student import ResearchModel
from signpath.targets.fitting import load_targets
from signpath.training.engine import Trainer
from signpath.training.runner import seed_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("Positive profiling steps required")
    resolved = load_config(args.config)
    c = resolved.values
    torch.set_num_threads(1)
    seed_runtime(c["experiment"]["seed"])
    train, _ = development_corpora(resolved)
    geo, cluster = load_targets(resolved, train)
    loader = train.pretrain_loader(c["data"]["num_workers"], c["data"]["persistent_workers"])
    provider = BatchProvider(c, loader, train.pretrain_sources(), geo, cluster_targets=cluster)
    trainer = Trainer(ResearchModel(c), c, args.device)
    device = torch.device(args.device)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    timings, starts, handles = {}, {}, []
    for name in ("online", "teacher", "predictor"):
        timings[name] = []
        def before(module, inputs, name=name):
            sync()
            starts[name] = time.perf_counter()
        def after(module, inputs, output, name=name):
            sync()
            timings[name].append(time.perf_counter() - starts[name])
        handles.extend((getattr(trainer.core, name).register_forward_pre_hook(before),
                        getattr(trainer.core, name).register_forward_hook(after)))
    records = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for _ in range(min(args.steps, c["training"]["max_successful_steps"])):
            start = time.perf_counter()
            batches = [provider.next_batch() for _ in range(c["training"]["gradient_accumulation"])]
            loading = time.perf_counter() - start
            sync()
            start = time.perf_counter()
            counts = {k: len(v) for k, v in timings.items()}
            result = trainer.update(batches)
            sync()
            update = time.perf_counter() - start
            stages = {k: sum(v[counts[k]:]) for k, v in timings.items()}
            records.append({"successful": result.succeeded, "batch_preparation_seconds": loading,
                "update_seconds": update, "forward_seconds": stages,
                "loss_backward_optimizer_ema_and_transfer_seconds": max(0., update - sum(stages.values())),
                "visible_raw_tokens": sum(int(b.visible.sum()) for b in batches),
                "teacher_raw_tokens": sum(int(b.valid.sum()) for b in batches) if c["loss"]["semantic_weight"] else 0,
                "eligible_geometry_targets": sum(int(b.geometry_targets.sum()) for b in batches)})
    finally:
        for handle in handles:
            handle.remove()
        loader.close()
    prepare = sum(r["batch_preparation_seconds"] for r in records)
    updates = sum(r["update_seconds"] for r in records)
    storage = sum(p.stat().st_size for p in resolved.path(c["cache"]["root"]).rglob("*") if p.is_file())
    report = {"status": "completed_profile_only", "config_hash": resolved.hash, "records": records,
        "loading_fraction": prepare / (prepare + updates), "cache_bytes_on_disk": storage,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "module_parameters": {name: sum(p.numel() for p in getattr(trainer.core, name).parameters()) for name in ("online", "teacher", "predictor")},
        "total_profile_cuda_device_hours": (prepare + updates) / 3600 if device.type == "cuda" else None,
        "scope": "instrumented cached updates including hooks/synchronization; not exclusive utilization or production latency",
        "preprocessing_gpu_hours": None, "preprocessing_gpu_hours_reason": "historical extraction not instrumented for utilization; raw embed records actual whole-pipeline wall seconds",
        "disk_throughput_bytes_per_second": None, "disk_throughput_reason": "OS page cache and bounded verified in-process cache; no physical disk benchmark was performed"}
    write_json(args.output, report)
    print({"status": report["status"], "successful_updates": sum(r["successful"] for r in records), "output": args.output})


if __name__ == "__main__":
    main()
