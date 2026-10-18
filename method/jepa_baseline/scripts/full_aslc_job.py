"""Detached full ASL-Citizen stages; durable resume, validation, and a 50k review gate."""

import argparse
from contextlib import nullcontext
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from signrep_baseline.job_support import (atomic_json, read_json, progress,
    install_stop_handlers, stop_requested, OrderedPairLoader, rotate_previous,
    append_metrics, trim_metrics, chunked_retrieval)
from signrep_baseline.cache import sha256
from signrep_baseline.corpus_policy import training_source, full_gallery, validate_pose_policy


def runtime(job):
    return read_json(job / "config.json"), read_json(job / "options.json")


def assert_source(job):
    manifest = read_json(job / "code_manifest.json")
    for name, expected in manifest.items():
        if sha256(job / "code" / name) != expected:
            raise ValueError(f"Frozen runtime source changed: {name}")
    if (job / "input_pins.json").exists():
        for name, expected in read_json(job / "input_pins.json").items():
            if sha256(name) != expected:
                raise ValueError(f"Frozen job configuration/input changed: {name}")


def disk_check(job, minimum_gib=3):
    import shutil
    free = shutil.disk_usage(job).free / 2**30
    if free < minimum_gib:
        raise OSError(f"Only {free:.2f} GiB disk free; need {minimum_gib} GiB to checkpoint safely")


def compute_features(job, checkpoint, split, rows, phase):
    import numpy as np
    import torch
    from signrep_baseline.model import SignRep
    from signrep_baseline.retrieval import extract_windows, pool_windows
    config, opts = runtime(job)
    model = SignRep().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    model.to(config["device"])
    model_pin = sha256(checkpoint)
    folder = job / "evaluation" / phase / split
    folder.mkdir(parents=True, exist_ok=True)
    unweighted, weighted = [], []
    invalid_activity = 0
    for index, row in enumerate(rows):
        if stop_requested():
            return None
        target = folder / f"{index:08d}.npz"
        video_pin = sha256(row["video"])
        if target.exists():
            with np.load(target, allow_pickle=False) as cache:
                meta = json.loads(str(cache["metadata_json"]))
                if meta["checkpoint_sha256"] != model_pin or meta["video_sha256"] != video_pin:
                    raise ValueError("Evaluation cache checkpoint/video identity changed")
                mean = cache["mean"].copy()
                active = cache["weighted"].copy()
                active_ok = bool(cache["weighted_valid"])
        else:
            context = torch.autocast("cuda", dtype=torch.bfloat16) if config["device"] == "cuda" else nullcontext()
            with context:
                values = extract_windows(model, row["video"], stride=2,
                         batch_size=opts["inference_batch_size"], short_policy="reject")
            mean = pool_windows(values["features"]).numpy()
            try:
                active = pool_windows(values["features"], values["activity"]).numpy()
                active_ok = True
            except ValueError:
                # Mean protocol is authoritative; never silently substitute it in weighted scores.
                active = np.zeros(768, np.float32)
                active_ok = False
            meta = {"schema": "signrep_pooled_job_v1", "video_id": row["video_id"],
                    "video_sha256": video_pin, "checkpoint_sha256": model_pin,
                    "stride": 2, "windows": len(values["features"]),
                    "compute": "bf16" if config["device"] == "cuda" else "float32"}
            tmp = target.with_name(target.name + ".tmp")
            with tmp.open("wb") as file:
                np.savez_compressed(file, mean=mean, weighted=active,
                                    weighted_valid=np.array(active_ok), metadata_json=np.array(json.dumps(meta)))
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp, target)
        unweighted.append(mean)
        weighted.append(active)
        invalid_activity += int(not active_ok)
        if index % 32 == 0 or index + 1 == len(rows):
            progress(job, "EVALUATION", phase=phase, split=split,
                     completed=index + 1, total=len(rows), invalid_activity=invalid_activity)
    del model, state
    gc.collect()
    if config["device"] == "cuda":
        torch.cuda.empty_cache()
    return {"mean": np.stack(unweighted), "weighted": np.stack(weighted),
            "weighted_invalid": invalid_activity}


def evaluate(job, checkpoint, gallery_path, queries_path, phase, split="val"):
    import torch
    from signrep_baseline.cli import held_out_contract
    result_path = job / "evaluation" / phase / "metrics.json"
    model_pin = sha256(checkpoint)
    if result_path.exists():
        result = read_json(result_path)
        if result["checkpoint_sha256"] != model_pin:
            raise ValueError("Evaluation phase already binds to a different checkpoint")
        return result
    gallery, queries = held_out_contract(gallery_path, queries_path, split)
    config, opts = runtime(job)
    if config["device"] == "cuda":
        torch.cuda.set_per_process_memory_fraction(opts["gpu_memory_fraction"])
    left = compute_features(job, checkpoint, "gallery", gallery, phase)
    if left is None:
        return None
    right = compute_features(job, checkpoint, "queries", queries, phase)
    if right is None:
        return None
    metrics = {}
    for mode in ("mean", "weighted"):
        if mode == "weighted" and (left["weighted_invalid"] or right["weighted_invalid"]):
            metrics[mode] = {"status": "UNAVAILABLE", "reason": "raw activity denominator/pooled vector invalid",
                             "gallery_invalid": left["weighted_invalid"], "query_invalid": right["weighted_invalid"]}
            continue
        score, _ = chunked_retrieval(left[mode], [r["label"] for r in gallery],
                right[mode], [r["label"] for r in queries], opts["retrieval_query_chunk"])
        score["split"] = split
        metrics[mode] = score
    result = {"checkpoint_sha256": model_pin, "phase": phase, "split": split,
              "gallery_sha256": sha256(gallery_path), "queries_sha256": sha256(queries_path),
              "metrics": metrics, "selected_primary": "mean"}
    atomic_json(result_path, result)
    progress(job, "EVALUATION_DONE", phase=phase, **metrics["mean"])
    return result


def training_contract(job, dataset, config, opts):
    from signrep_baseline.cli import environment_record
    return {"schema": "full_aslc_job_training_v1", "config": config,
            "manifest_sha256": dataset.manifest_sha256,
            "audit_sha256": sha256(Path(config["manifest"]).with_suffix(".jsonl.audit.json")),
            "checkpoint_sha256": sha256(config["checkpoint"]),
            "proxy_gallery_sha256": sha256(config["validation_gallery"]),
            "proxy_queries_sha256": sha256(config["validation_queries"]),
            "code_manifest_sha256": sha256(job / "code_manifest.json"),
            "data_rng_scheme": "signrep_pair_data_v1",
            "execution_options": opts, "environment": environment_record()}


def train(job):
    import torch
    from signrep_baseline.augmentation import VideoTransform, AugmentationConfig
    from signrep_baseline.data import PairedVideoDataset, StatefulPairSampler
    from signrep_baseline.model import SignRep
    from signrep_baseline.schema import FidelityChoices
    from signrep_baseline.training import Pretrainer, seed_everything, atomic_torch_save
    from signrep_baseline.preparation import validate_pretrain_audit
    install_stop_handlers()
    config, opts = runtime(job)
    stage_limit = read_json(job / "control.json")["stage_limit"]
    torch.set_num_threads(opts["cpu_threads"])
    if config["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_per_process_memory_fraction(opts["gpu_memory_fraction"])
    validate_pretrain_audit(config["manifest"])
    seed_everything(config["seed"])
    choices = FidelityChoices(**config["choices"])
    dataset = PairedVideoDataset(config["manifest"], VideoTransform(True,
                    AugmentationConfig(**config["augmentation"])), choices, cache_capacity=8)
    sampler = StatefulPairSampler(len(dataset), config["batch_size"] // 2, config["seed"])
    model = SignRep()
    # Always only Kinetics encoder initialization; no SignRep checkpoint code path.
    if config["initialization"] != "kinetics_mae":
        raise ValueError("This full job permits only Kinetics encoder initialization")
    model.initialize_backbone(config["checkpoint"])
    trainer = Pretrainer(model, choices, config["device"], config["total_steps"],
                config["warmup_steps"], config["max_lr"], config["amp"], config["amp_dtype"])
    contract = training_contract(job, dataset, config, opts)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    last = output / "last.pt"
    history = output / "metrics.jsonl"
    resumed = False
    if last.exists():
        try:
            trainer.resume(last, sampler, contract)
        except (EOFError, OSError, RuntimeError):
            previous = output / "last.previous.pt"
            if not previous.exists():
                raise
            trainer.resume(previous, sampler, contract)
            progress(job, "RESUMED_PREVIOUS_CHECKPOINT", checkpoint=str(previous), step=trainer.step)
        resumed = True
    trim_metrics(history, trainer.step)
    atomic_json(output / "run_contract.json", contract)
    if not last.exists():
        trainer.save(last, sampler, contract)
        atomic_torch_save({"model": model.state_dict(), "step": 0,
                           "initialization": "kinetics_mae", "run_contract": contract}, output / "initial.model.pt")
    progress(job, "TRAINING", step=trainer.step, stage_limit=stage_limit,
             schedule_steps=config["total_steps"], resumed=resumed, views=config["batch_size"])
    # A forced interruption can occur after publishing the full checkpoint but
    # before publishing its model export/pending-validation marker. Reconstruct
    # the export and run this boundary's validation before any further update.
    if trainer.step and trainer.step % config["validation_interval"] == 0:
        phase = f"proxy_step{trainer.step}"
        if not (job / "evaluation" / phase / "metrics.json").exists():
            trainer.save(last, sampler, contract)
            atomic_json(job / "pending_validation.json", {"step": trainer.step, "phase": phase})
            return 76
    loader = OrderedPairLoader(dataset, sampler, config["seed"], trainer.step,
            opts["data_workers"], opts["prefetch_factor"], opts["worker_threads"],
            pin_memory=config["device"] == "cuda")
    last_save = time.monotonic()
    started, first_step = time.monotonic(), trainer.step
    validation_log = output / "validation.jsonl"
    try:
        while trainer.step < min(stage_limit, config["total_steps"]):
            if stop_requested():
                rotate_previous(last)
                trainer.save(last, sampler, contract)
                progress(job, "TRAINING_STOPPED", step=trainer.step, checkpoint=str(last))
                return 75
            before = time.monotonic()
            batch = loader.next_batch()
            if stop_requested():
                rotate_previous(last)
                trainer.save(last, sampler, contract)
                progress(job, "TRAINING_STOPPED", step=trainer.step, checkpoint=str(last))
                return 75
            metrics = trainer.update(batch)
            loader.commit(trainer.step)
            metrics["wall_seconds"] = time.monotonic() - before
            metrics["supervision_valid_elements"] = {k: int(v.sum()) for k, v in batch["masks"].items()}
            metrics["supervision_total_elements"] = {k: v.numel() for k, v in batch["masks"].items()}
            append_metrics(history, metrics)
            elapsed = time.monotonic() - started
            rate = (trainer.step - first_step) / max(elapsed, 1e-6)
            if trainer.step == first_step + 1 or trainer.step % 25 == 0:
                progress(job, "TRAINING", step=trainer.step, stage_limit=stage_limit,
                         loss=round(metrics["loss"], 5), reconstruction=round(metrics["reconstruction"], 5),
                         adversarial=metrics["adversarial"], lr=metrics["lr"],
                         eta_hours=round((stage_limit - trainer.step) / rate / 3600, 2),
                         last_checkpoint=str(last), cuda_reserved_MiB=round(torch.cuda.memory_reserved() / 2**20, 1) if config["device"] == "cuda" else 0)
            now = time.monotonic()
            if trainer.step % config["save_interval"] == 0 or now - last_save >= opts["checkpoint_seconds"] or stop_requested():
                disk_check(job)
                rotate_previous(last)
                trainer.save(last, sampler, contract)
                last_save = now
            if trainer.step % config["validation_interval"] == 0:
                # Restart this subprocess at validation boundaries. Optimizer and
                # worker VRAM/RAM are then released for deterministic frozen eval.
                rotate_previous(last)
                trainer.save(last, sampler, contract)
                phase = f"proxy_step{trainer.step}"
                if not (job / "evaluation" / phase / "metrics.json").exists():
                    atomic_json(job / "pending_validation.json", {"step": trainer.step, "phase": phase})
                    return 76
        rotate_previous(last)
        trainer.save(last, sampler, contract)
        atomic_torch_save({"model": model.state_dict(), "step": trainer.step,
                           "run_contract": contract}, output / f"step{trainer.step}.model.pt")
        atomic_json(job / "training_stage_complete.json", {"step": trainer.step,
                    "stage_limit": stage_limit, "checkpoint": str(last)})
        return 0
    finally:
        loader.close()


def validation(job):
    import torch
    from signrep_baseline.training import atomic_torch_save
    install_stop_handlers()
    config, opts = runtime(job)
    torch.set_num_threads(opts["cpu_threads"])
    task = read_json(job / "pending_validation.json")
    last = Path(config["output"]) / "last.model.pt"
    result = evaluate(job, last, config["validation_gallery"], config["validation_queries"], task["phase"])
    if result is None:
        return 75
    score = result["metrics"]["mean"]["per_instance_top1"]
    best_note = job / "best_proxy.json"
    best = read_json(best_note)["score"] if best_note.exists() else -float("inf")
    append_metrics(Path(config["output"]) / "validation.jsonl", {"step": task["step"], **result})
    if score > best:
        import shutil
        best_path = Path(config["output"]) / "best.model.pt"
        tmp = best_path.with_name(best_path.name + ".tmp")
        shutil.copyfile(last, tmp)
        with tmp.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(tmp, best_path)
        atomic_json(best_note, {"score": score, "step": task["step"], "checkpoint": str(best_path)})
    (job / "pending_validation.json").unlink()
    return 0


def corpus_audit(job):
    import numpy as np
    import torch
    from signrep_baseline.preparation import validate_pretrain_audit
    from signrep_baseline.cache import load_standardized, clip_targets
    config, opts = runtime(job)
    torch.set_num_threads(opts["cpu_threads"])
    validate_pretrain_audit(config["manifest"])
    rows = [json.loads(s) for s in Path(config["manifest"]).read_text().splitlines() if s]
    policy = validate_pose_policy(job, opts)
    expected = policy["effective_train_videos"]
    if len(rows) != expected:
        raise ValueError("Prepared corpus does not contain exactly the approved training membership")
    valid, total = {}, {}
    for index, row in enumerate(rows):
        if stop_requested():
            return 75
        data, _ = load_standardized(row["pose_cache"], row["bone_statistics_sha256"], row["video_sha256"])
        flags = data["body_detected"]
        starts = np.flatnonzero(np.convolve(flags.astype(int), np.ones(16, int), mode="valid") == 16)
        if not len(starts):
            raise ValueError(f"No signer-present training clip: {row['video_id']}")
        _, masks = clip_targets(data, np.arange(starts[0], starts[0] + 16))
        for key, value in masks.items():
            valid[key] = valid.get(key, 0) + int(value.sum())
            total[key] = total.get(key, 0) + value.numel()
        if index % 256 == 0:
            progress(job, "CORPUS_AUDIT", completed=index + 1, total=len(rows))
    atomic_json(job / "corpus_check.json", {"status": "PASS", "videos": len(rows),
                "membership_policy": policy,
                "coverage_first_valid_clip": {k: valid[k] / total[k] for k in valid},
                "activity_limitation": "Coverage is reported; missing targets never become inactive labels"})
    return 0


def child(job, command, stage):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUNBUFFERED="1",
               SIGNREP_JOB_DIRECTORY=str(job))
    progress(job, stage, command=" ".join(map(str, command)))
    process = subprocess.Popen(list(map(str, command)), env=env)
    last_report = time.monotonic()
    while process.poll() is None:
        if stop_requested():
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return 75
        if time.monotonic() - last_report >= 30:
            # Child stages write more specific progress; heartbeat does not overwrite it.
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | HEARTBEAT | {stage} | child_pid={process.pid}"
            with (job / "progress.txt").open("a", buffering=1) as file:
                file.write(line + "\n")
            print(line, flush=True)
            last_report = time.monotonic()
        time.sleep(1)
    return process.returncode


def pipeline(job):
    install_stop_handlers()
    assert_source(job)
    lock = (job / "pipeline.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError("Another process already holds this job lock")
    config, opts = runtime(job)
    python = Path(opts["rgb_python"])
    native = Path(opts["pose_python"])
    runner = Path(__file__)
    if not (job / "pose_report.json").exists():
        rc = child(job, [native, ROOT / "scripts/full_aslc_pose.py", "--job", job], "POSE_START")
        if rc and not (rc == 78 and opts.get("approved_exclusions") and (job / "pose_report.json").exists()):
            return rc
    policy = validate_pose_policy(job, opts)
    atomic_json(job / "effective_pose_check.json", policy)
    progress(job, "POSE_ACCEPTED_WITH_AUDIT", official=policy["official_train_videos"],
             training=policy["effective_train_videos"], excluded=len(policy["excluded_video_ids"]))
    if not Path(config["manifest"]).with_suffix(".jsonl.audit.json").exists():
        rc = child(job, [python, "-m", "signrep_baseline", "prepare-corpus", "--manifest", training_source(job, opts),
            "--held-out-manifest", job / "manifests/official_held_out.jsonl",
            "--training-reference-manifest", job / "manifests/official_train.jsonl",
            "--output", job / "corpus", "--pose-python", native,
            "--weights", opts["teacher_weights"]], "NORMALIZATION")
        if rc:
            return rc
    if not (job / "corpus_check.json").exists():
        rc = child(job, [python, runner, "audit", "--job", job], "CORPUS_AUDIT_START")
        if rc:
            return rc
    while True:
        if stop_requested():
            return 75
        if (job / "pending_validation.json").exists():
            rc = child(job, [python, runner, "validate", "--job", job], "PROXY_VALIDATION_START")
            if rc:
                return rc
        rc = child(job, [python, runner, "train", "--job", job], "TRAIN_START_OR_RESUME")
        if rc == 76:
            continue
        if rc:
            return rc
        break
    limit = read_json(job / "control.json")["stage_limit"]
    candidates = [Path(config["output"]) / "best.model.pt", Path(config["output"]) / f"step{limit}.model.pt"]
    metrics = []
    pins = set()
    for checkpoint in candidates:
        if not checkpoint.exists() or sha256(checkpoint) in pins:
            continue
        pin = sha256(checkpoint)
        pins.add(pin)
        phase = "full_val_" + pin[:16]
        result = evaluate(job, checkpoint, full_gallery(job, opts),
                          job / "manifests/official_val.jsonl", phase)
        if result is None:
            return 75
        metrics.append({"checkpoint": str(checkpoint), **result})
    selected = max(metrics, key=lambda m: m["metrics"]["mean"]["per_instance_top1"])
    atomic_json(job / "review.json", {"stage_limit": limit, "candidates": metrics,
                "selected": selected, "test_evaluated": False,
                "detail": "Review validation before extending training or authorizing final test"})
    progress(job, "REVIEW_REQUIRED", step=limit, selected=selected["checkpoint"],
             validation_top1=selected["metrics"]["mean"]["per_instance_top1"],
             validation_MRR=selected["metrics"]["mean"]["MRR"], test_evaluated=False)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "train", "validate", "audit"])
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = args.job.resolve()
    install_stop_handlers()
    try:
        disk_check(job)
        assert_source(job)
        fn = {"run": pipeline, "train": train, "validate": validation, "audit": corpus_audit}[args.stage]
        code = fn(job)
        if code == 75:
            progress(job, "STOPPED", detail="Restart the service to resume committed caches/checkpoint")
            return 0 if args.stage == "run" else 75
        if code:
            progress(job, "FAILED", returncode=code)
        return code
    except (ValueError, OSError) as error:
        progress(job, "REVIEW_REQUIRED_ERROR", error=str(error), auto_restart=False)
        traceback.print_exc()
        return 78
    except Exception as error:
        progress(job, "CRASHED", error=repr(error), auto_resume=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
