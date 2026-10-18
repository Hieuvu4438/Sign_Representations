"""Real manifest/cache training, registered validation probes and best/last resume."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import torch

from ..cache.hashing import canonical_hash, file_hash
from ..cli import doctor, write_json
from ..data.dataset import BatchProvider
from ..evaluation.islr import fit_probe
from ..evaluation.protocols import audit_development, development_corpora, require_pipeline_gate
from ..models.teacher_student import ResearchModel
from ..targets.fitting import load_targets
from .checkpoint import capture_rng, restore_rng, load_checkpoint, save_checkpoint
from .distributed import all_true, initialize, rank, world_size
from .engine import Trainer
from .runner import json_finite, runtime_implementation_hash, seed_runtime


def run_cached_training(resolved, device="cpu", resume=None):
    c = resolved.values
    if c["cache"]["kind"] != "per_frame_dino" or c["runtime"]["allow_missing_provenance"]:
        raise ValueError("Scientific training requires real per-frame cache with complete provenance")
    retrieval_selection = c["experiment"]["primary_validation_metric"] == "retrieval_dcg"
    if not retrieval_selection and (c["experiment"]["primary_validation_metric"] != "islr_per_class_top1" or c["downstream"]["head"] != "linear" or c["downstream"]["representation"] != "temporal"):
        raise ValueError("Pretraining checkpoint selection requires registered frozen temporal linear PC Top1 probe")
    device = initialize(device)
    torch.set_num_threads(1)
    seed_runtime(c["experiment"]["seed"])
    train, val = development_corpora(resolved)
    audit = audit_development(resolved, train, val)
    gate_hash = require_pipeline_gate(resolved, train, val)
    geometry, cluster = load_targets(resolved, train)
    loader = train.pretrain_loader(c["data"]["num_workers"], c["data"]["persistent_workers"])
    provider = BatchProvider(c, loader, train.pretrain_sources(), geometry, seed=c["experiment"]["seed"] + rank() * 1009,
                             cluster_targets=cluster)
    identities = {"train_corpus": train.fingerprint, "val_corpus": val.fingerprint, "cache_identity": train.identity,
                  "pipeline_gate_sha256": gate_hash, "target_hash": geometry.fingerprint() if geometry else None,
                  "cluster_hash": file_hash(resolved.path(c["cache"]["root"]).parent / "targets/kmeans/metadata.json") if cluster else None,
                  "runtime_implementation_hash": runtime_implementation_hash(), "protocol": c["experiment"]["evaluation_protocol"]}
    identities["world_size"] = world_size()
    directory = resolved.path(c["experiment"]["output_dir"]) / canonical_hash({"config": resolved.hash, "identities": identities})[:12]
    trainer = Trainer(ResearchModel(c), c, device)
    best_score, best_step = None, None
    if resume:
        if Path(resume).resolve().parent.parent != directory.resolve():
            raise ValueError("Resume checkpoint does not belong to this config/data/implementation run")
        state = load_checkpoint(resume, trainer, c, identities, mask_generator=provider.mask_generator)
        provider.load_state_dict(state["provider"])
        best_score, best_step = state["best_score"], state["best_step"]
        # Power loss may leave logs newer than the last durable model/sampler state.
        # Preserve those abandoned records separately before replaying the same draws.
        if rank() == 0:
            from .recovery import reconcile_logs
            reconcile_logs(directory, trainer.attempted_batches)
    else:
        if not all_true(not directory.exists(), device):
            raise FileExistsError(f"Run exists; use --resume: {directory}")
        startup_error = None
        if rank() == 0:
            try:
                directory.mkdir(parents=True)
                write_json(directory / "resolved_config.json", resolved.provenance())
                write_json(directory / "environment.json", doctor())
                write_json(directory / "identities.json", identities)
                from .provenance import write_run_provenance
                write_run_provenance(directory, resolved, train, val, audit, identities)
            except Exception as caught:
                startup_error = str(caught)
        if world_size() > 1:
            message = [startup_error]
            torch.distributed.broadcast_object_list(message, src=0)
            startup_error = message[0]
        if startup_error:
            raise OSError(f"Run artifact initialization failed: {startup_error}")
    if world_size() > 1:
        torch.distributed.barrier()
    def checkpoint(path):
        save_checkpoint(path, trainer, c, identities, mask_generator=provider.mask_generator,
            data_state={"provider": provider.state_dict(), "best_score": best_score, "best_step": best_step})
    if not resume:
        # Even a failure before update1 has a valid initialization/sampler checkpoint.
        checkpoint(directory / "checkpoints/last.pt")
    def registered_validation():
        validation_started = time.monotonic()
        rng = capture_rng(provider.mask_generator)
        payload = [None]
        if rank() == 0:
            try:
                if retrieval_selection:
                    from ..evaluation.dictionary import evaluate_dictionary
                    evaluated = evaluate_dictionary(trainer.core, train, val, c,
                        directory / "retrieval" / f"val_step{trainer.successful_step:09d}", device=device)
                    payload[0] = {"score": evaluated["metrics"]["DCG"], "metrics": evaluated["metrics"],
                        "trials": [], "step": trainer.successful_step, "selection_metric": "retrieval_dcg", "classifier_trained": False}
                else:
                    probe = fit_probe(train, val, c, trainer.core, c, device=device)
                    payload[0] = {"score": probe.metrics["per_class_top1"], "metrics": probe.metrics,
                                  "trials": list(probe.trials), "step": trainer.successful_step}
            except Exception as caught:
                payload[0] = {"error": str(caught)}
        if world_size() > 1:
            torch.distributed.broadcast_object_list(payload, src=0)
        restore_rng(rng, provider.mask_generator)
        if "error" in payload[0]:
            raise ValueError(f"Registered validation failed: {payload[0]['error']}")
        trainer.elapsed_runtime += time.monotonic() - validation_started
        return payload[0]
    status, validations, session_start = "completed", [], time.monotonic()
    try:
        while trainer.successful_step < c["training"]["max_successful_steps"]:
            from .interruptions import raise_if_requested
            raise_if_requested()
            if trainer.attempted_updates >= c["training"]["max_successful_steps"] * 10:
                status = "failed_repeated_empty_or_nonfinite_updates"
                break
            if not all_true(trainer.elapsed_runtime < c["runtime"]["max_runtime_hours"] * 3600, device):
                status = "budget_stopped"
                break
            started, before = time.monotonic(), trainer.elapsed_runtime
            error, batches = None, None
            try:
                batches = [provider.next_batch() for _ in range(c["training"]["gradient_accumulation"])]
            except (ValueError, OSError) as caught:
                error = str(caught)
            if not all_true(error is None, device):
                raise ValueError(f"Distributed cache preparation failed: {error or 'peer rank error'}")
            result = trainer.update(batches)
            trainer.elapsed_runtime += max(0, time.monotonic() - started - (trainer.elapsed_runtime - before))
            mask_schedule = canonical_hash([{"samples": batch.sample_ids, "sources": batch.source_frame_index.cpu().tolist(),
                "visible": batch.visible.cpu().tolist()} for batch in batches])
            exposure = sum(int((batch.source_frame_index >= 0).sum()) for batch in batches)
            distinct_exposure = sum(len(torch.unique(row[row >= 0])) for batch in batches for row in batch.source_frame_index)
            per_rank = [{"mask_schedule_hash": mask_schedule, "selected_frames": exposure, "distinct_source_observations": distinct_exposure}]
            if world_size() > 1:
                gathered = [None] * world_size()
                torch.distributed.all_gather_object(gathered, per_rank[0])
                per_rank = gathered
            mask_schedule = canonical_hash([x["mask_schedule_hash"] for x in per_rank])
            if rank() == 0:
                with (directory / "logs.jsonl").open("a") as stream:
                    stream.write(json.dumps(json_finite({**asdict(result), "mask_schedule_hash": mask_schedule,
                        "selected_frame_exposure": sum(x["selected_frames"] for x in per_rank),
                        "distinct_source_frame_exposure": sum(x["distinct_source_observations"] for x in per_rank),
                        "per_rank_exposure": per_rank, "mask_counts": [batch.diagnostics for batch in batches]}), allow_nan=False) + "\n")
                with (directory / "mask_diagnostics.jsonl").open("a") as stream:
                    stream.write(json.dumps({"successful_step": result.successful_step, "attempted_batches": result.attempted_batches,
                        "mask_schedule_hash": mask_schedule, "per_rank_exposure": per_rank,
                        "mask_counts": [batch.diagnostics for batch in batches]}, allow_nan=False) + "\n")
            validate = result.succeeded and (trainer.successful_step % c["training"]["validation_interval"] == 0
                or trainer.successful_step == c["training"]["max_successful_steps"])
            improved = False
            diagnostic = result.succeeded and (trainer.successful_step % c["training"]["diagnostic_interval"] == 0
                or trainer.successful_step == c["training"]["max_successful_steps"])
            if diagnostic:
                from .monitor import training_diagnostics
                saved_rng, started_diagnostic = capture_rng(provider.mask_generator), time.monotonic()
                error = None
                if rank() == 0:
                    try:
                        training_diagnostics(trainer.core, train, c, directory, trainer.successful_step, device)
                    except Exception as caught:
                        error = str(caught)
                if world_size() > 1:
                    message = [error]
                    torch.distributed.broadcast_object_list(message, src=0)
                    error = message[0]
                restore_rng(saved_rng, provider.mask_generator)
                if error:
                    raise ValueError(f"Training diagnostics failed: {error}")
                trainer.elapsed_runtime += time.monotonic() - started_diagnostic
            if validate:
                validation = registered_validation()
                validations.append(validation)
                if best_score is None or validation["score"] > best_score:
                    best_score, best_step, improved = validation["score"], trainer.successful_step, True
                if rank() == 0:
                    write_json(directory / f"validation_step{trainer.successful_step}.json", validation)
            if improved:
                checkpoint(directory / "checkpoints/best.pt")
            if result.succeeded and trainer.successful_step % c["training"]["checkpoint_interval"] == 0:
                step_path = directory / f"checkpoints/step{trainer.successful_step:09d}.pt"
                checkpoint(step_path)
                checkpoint(directory / "checkpoints/last.pt")
                if rank() == 0:
                    for old in sorted(step_path.parent.glob("step*.pt"))[:-c["training"]["keep_last"]]:
                        old.unlink()
        if resume and trainer.successful_step == c["training"]["max_successful_steps"] and not validations:
            # A reboot/stop during final retrieval must retry evaluation, not skip it
            # merely because all optimizer steps are already durable.
            validation = registered_validation()
            validations.append(validation)
            if best_score is None or validation["score"] > best_score:
                best_score, best_step = validation["score"], trainer.successful_step
                checkpoint(directory / "checkpoints/best.pt")
            if rank() == 0:
                write_json(directory / f"validation_step{trainer.successful_step}.json", validation)
    except Exception as caught:
        if rank() == 0:
            write_json(directory / "failure.json", {"status": "failed", "reason": str(caught),
                "successful_steps": trainer.successful_step, "attempted_updates": trainer.attempted_updates})
        raise
    finally:
        checkpoint(directory / "checkpoints/last.pt")
        if hasattr(loader, "close"):
            loader.close()
    report = {"status": status, "run_dir": str(directory), "successful_steps": trainer.successful_step,
        "attempted_updates": trainer.attempted_updates, "attempted_batches": trainer.attempted_batches,
        "best_validation_per_class_top1": best_score if not retrieval_selection else None,
        "best_validation_dcg": best_score if retrieval_selection else None,
        "selection_metric": c["experiment"]["primary_validation_metric"], "best_step": best_step, "validations": validations,
        "precision": trainer.precision, "precision_reason": trainer.precision_reason, "config_hash": resolved.hash,
        "session_seconds": time.monotonic() - session_start, "total_training_seconds": trainer.elapsed_runtime,
        "test_used_for_selection": False, "loader_kind": "threaded_cache_readers_main_owned_sampling_rng",
        "cache_reader_workers": c["data"]["num_workers"], "world_size": world_size(), "identities": identities}
    report["pinned_memory_active"] = c["data"]["pin_memory"] and device.type == "cuda"
    report["temporal_trainable_params"] = sum(p.numel() for p in trainer.core.parameters() if p.requires_grad)
    report["temporal_total_params_including_teacher"] = sum(p.numel() for p in trainer.core.parameters())
    if rank() == 0:
        write_json(directory / "summary.json", json_finite(report))
        from .provenance import write_decision_memo
        write_decision_memo(directory, c, report)
    return report
