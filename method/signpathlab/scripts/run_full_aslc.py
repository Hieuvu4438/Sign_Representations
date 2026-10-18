"""One-seed E04 full official ASL Citizen pipeline; no classifier stage or auto retry."""
import argparse
from contextlib import nullcontext
import fcntl
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import traceback
import sys
import yaml

from signpath.cache.hashing import file_hash
from signpath.cache.index import CachedCorpus
from signpath.cli import doctor, write_json
from signpath.config import load_config
from signpath.data.audit import audit_directory, audit_records
from signpath.data.manifest import read_manifest
from signpath.evaluation.dictionary import evaluate_dictionary
from signpath.evaluation.protocols import development_corpora, verify_pipeline
from signpath.inference.extract import load_encoder
from signpath.preprocessing_full import build_full_cache
from signpath.targets.fitting import fit_targets, load_targets
from signpath.training.cached_runner import run_cached_training
from signpath.training.profile import profile
from signpath.training.runner import runtime_implementation_hash
from signpath.training.interruptions import install_handlers, raise_if_requested, JobInterrupted
from signpath.training.recovery import choose_resume_checkpoint, validate_job_restart, quarantine_uncommitted_targets
from full_job_service_support import progress_reporter, reuse_completed_cache, wait_for_gpu_dependency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_aslc/E04_seed17.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dino-batch-size", type=int, choices=(16,32,64), default=16)
    parser.add_argument("--assets-file", default="registries/local_extraction.yaml")
    parser.add_argument("--job-dir", default="artifacts/full_aslc/job_seed17")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--auto-resume", action="store_true", help="Resume interruption/reboot; do not retry scientific failures")
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="test")
    parser.add_argument("--wait-before-training-job", help="Finish the other job before GPU profile/SSL training")
    parser.add_argument("--progress-job-dir", action="append", default=[], help="Also log this job from the same process")
    parser.add_argument("--reuse-cache-root", action="append", default=[], help="Hardlink checksum-verified identical feature caches")
    args = parser.parse_args()
    install_handlers()
    if args.workers not in (1,2):
        raise ValueError("User process limit: one Python process with at most2CPU worker threads")
    resolved = load_config(args.config)
    c = resolved.values
    job = resolved.path(args.job_dir)
    job.mkdir(parents=True, exist_ok=True)
    lock = (job / ".job.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_hash = runtime_implementation_hash()
    execution_policy = {"wait_before_training_job": str(resolved.path(args.wait_before_training_job)) if args.wait_before_training_job else None,
        "reuse_cache_roots": [str(resolved.path(p)) for p in args.reuse_cache_root]}
    state_path = job / "status.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if not (args.resume or args.auto_resume):
            raise FileExistsError("Job already has status; explicit --resume required")
        validate_job_restart(previous, resolved.hash, source_hash, args.evaluation_split,
            automatic=args.auto_resume and not args.resume)
        expected_execution = {"cpu_threads": args.workers, "dino_batch_size": args.dino_batch_size, "python_processes": 1}
        if previous.get("cache_execution", expected_execution) != expected_execution:
            raise ValueError("Resume cache execution options changed")
        if previous.get("execution_policy", execution_policy) != execution_policy:
            raise ValueError("Resume execution policy changed")
        if previous["status"] == "completed":
            print("Job already completed; no training repeated", flush=True)
            return
        args.resume = True
    state = {"status": "running", "stage": "startup", "pid": os.getpid(), "config_hash": resolved.hash,
        "runtime_implementation_hash": source_hash, "started_unix": time.time(), "config": str(Path(args.config).resolve()),
        "classifier_training": False, "seed": c["experiment"]["seed"],
        "monitoring": "systemd resumes interruptions/reboots; scientific failures require review" if args.auto_resume else "user-managed"}
    state["evaluation_split"] = args.evaluation_split
    state["cache_execution"] = {"cpu_threads": args.workers, "dino_batch_size": args.dino_batch_size, "python_processes": 1}
    state["execution_policy"] = execution_policy
    def stage(name):
        raise_if_requested()
        if runtime_implementation_hash() != source_hash:
            raise ValueError("Runtime source changed while job was running; stop to prevent mixed implementations")
        state.update(stage=name, updated_unix=time.time())
        write_json(state_path, state)
        print(f"STAGE {name}", flush=True)
    def once(name, action):
        stage(name)
        marker = job / f"stage.{name}.json"
        if marker.exists() and args.resume:
            report = json.loads(marker.read_text())
            if report["config_hash"] != resolved.hash or report["source_hash"] != source_hash:
                raise ValueError("Stage marker identity mismatch")
            return report["result"]
        begin = time.monotonic()
        result = action()
        write_json(marker, {"config_hash": resolved.hash, "source_hash": source_hash,
            "elapsed_seconds": time.monotonic() - begin, "result": result})
        return result
    try:
        write_json(job / "environment.json", doctor())
        write_json(job / "resolved_config.json", resolved.provenance())
        (job / "resolved_config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
        shutil.copyfile(resolved.root / "requirements-main-extraction.lock.txt", job / "requirements.lock.txt")
        if not (job / "source_snapshot.tar.gz").exists():
            with tarfile.open(job / "source_snapshot.tar.gz", "w:gz") as archive:
                for folder in ("src", "configs", "scripts", "registries"):
                    for path in sorted((resolved.root / folder).rglob("*")):
                        if path.is_file() and path.suffix in {".py", ".yaml", ".sh"}:
                            archive.add(path, arcname=str(path.relative_to(resolved.root)))
        state["source_snapshot_sha256"] = file_hash(job / "source_snapshot.tar.gz")
        audit = once("manifest_audit", lambda: audit_directory(resolved.path(c["data"]["train_manifest"]).parent))
        if not audit["passed"]:
            raise ValueError("Official manifest audit failed")
        state["corpus_counts"] = audit["counts"]
        for split in ("train", "val"):
            if args.reuse_cache_root:
                once(f"reuse_cache_{split}", lambda split=split: reuse_completed_cache(resolved, split,
                    execution_policy["reuse_cache_roots"]))
            once(f"cache_{split}", lambda split=split: build_full_cache(resolved, split, args.assets_file, args.device,
                workers=args.workers, dino_batch_size=args.dino_batch_size, progress_path=job / f"cache.{split}.json"))
        train, val = development_corpora(resolved)
        def targets():
            stats = resolved.path(c["geometry"]["projection"]["statistics_path"])
            if stats.with_suffix(stats.suffix + ".json").exists():
                load_targets(resolved, train)
                return {"status": "verified_reuse", "statistics": str(stats)}
            if args.resume:
                matrices = resolved.path(c["geometry"]["projection"]["matrices_path"])
                quarantine_uncommitted_targets((stats, matrices), job / "recovery")
            return fit_targets(resolved, "geometry", "train")
        once("targets_train_only", targets)
        once("pipeline_gate", lambda: verify_pipeline(resolved, job / "pipeline_checks.json"))
        if args.wait_before_training_job:
            once("wait_for_gpu_training", lambda: wait_for_gpu_dependency(execution_policy["wait_before_training_job"]))
        once("training_profile", lambda: profile(resolved, 3, job / "training_profile.json", args.device))
        def training():
            resume = None
            if args.resume:
                resume = choose_resume_checkpoint(resolved.path(c["experiment"]["output_dir"]))
            return run_cached_training(resolved, args.device, resume)
        trained = once("training", training)
        if trained["status"] != "completed":
            raise RuntimeError(f"Training stopped before full budget: {trained['status']}")
        state["training_run"] = trained["run_dir"]
        checkpoint = Path(trained["run_dir"]) / "checkpoints/best.pt"
        state["selected_checkpoint_sha256"] = file_hash(checkpoint)
        if args.evaluation_split == "val":
            final = {"status": "completed", "training_run": trained["run_dir"],
                "checkpoint_sha256": state["selected_checkpoint_sha256"], "successful_steps": trained["successful_steps"],
                "val_metrics": trained["validations"][-1]["metrics"], "evaluation_split": "val",
                "test_metrics": None, "test_reason": "ASLC300 reference comparison uses1500validation queries, no test requested",
                "test_used_for_selection": False, "classifier_trained": False}
            write_json(job / "result.json", final)
            state.update(status="completed", stage="done", finished_unix=time.time())
            write_json(state_path, state)
            return
        once("cache_test", lambda: build_full_cache(resolved, "test", args.assets_file, args.device,
            workers=args.workers, dino_batch_size=args.dino_batch_size, progress_path=job / "cache.test.json"))
        stage("test_audit")
        test = CachedCorpus(resolved.path(c["data"]["test_manifest"]), resolved.path(c["cache"]["root"]), split="test", config=c)
        audit_test = audit_records([*train.records.values(), *val.records.values(), *test.records.values()], verify_files=True)
        write_json(job / "test_audit.json", audit_test)
        if not audit_test["passed"]:
            raise ValueError("Test overlap audit failed")
        model, identity, encoder_config, digest = load_encoder(checkpoint, args.device)
        if identity != test.identity or digest != state["selected_checkpoint_sha256"]:
            raise ValueError("Frozen model/test input identity mismatch")
        evaluated = once("retrieval_test", lambda: evaluate_dictionary(model, train, test, encoder_config,
            job / "retrieval_test", device=args.device, split="test"))
        final = {"status": "completed", "training_run": trained["run_dir"], "checkpoint_sha256": digest,
            "successful_steps": trained["successful_steps"], "val_metrics": trained["validations"][-1]["metrics"],
            "test_metrics": evaluated["metrics"], "test_used_for_selection": False, "classifier_trained": False}
        write_json(job / "result.json", final)
        state.update(status="completed", stage="done", finished_unix=time.time())
        write_json(state_path, state)
    except JobInterrupted as error:
        state.update(status="interrupted", error=str(error), finished_unix=time.time())
        write_json(state_path, state)
        print(str(error), flush=True)
        sys.exit(75)
    except BaseException as error:
        state.update(status="failed", error=str(error), finished_unix=time.time())
        write_json(state_path, state)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        # Parse only operational log paths here; main owns all scientific args.
        logging_parser = argparse.ArgumentParser(add_help=False)
        logging_parser.add_argument("--job-dir", default="artifacts/full_aslc/job_seed17")
        logging_parser.add_argument("--progress-job-dir", action="append", default=[])
        logging_args, _ = logging_parser.parse_known_args()
        # Only the full service owns the reporter for both jobs after reboot.
        logging_context = (progress_reporter([logging_args.job_dir, *logging_args.progress_job_dir])
            if logging_args.progress_job_dir else nullcontext())
        with logging_context:
            main()
    except (ValueError, OSError, RuntimeError):
        # Scientific/config/gate failures should not trigger service retries.
        sys.exit(2)
