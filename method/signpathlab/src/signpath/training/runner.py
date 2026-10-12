"""Bounded training runs with immutable run identity and explicit correctness fixtures."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch

from ..cache.hashing import canonical_hash, file_hash
from ..cli import doctor, write_json
from ..data.collate import TrainingExample, collate_examples
from ..data.dataset import BatchProvider, fitting_geometry, geometry_kwargs
from ..data.fixtures import synthetic_features
from ..geometry.paths import geometry_from_features
from ..masking.closure import ClosureResolver
from ..masking.samplers import MaskSampler
from ..models.heads import LinearHead
from ..models.teacher_student import ResearchModel
from ..models.tokenizer import SignTokenizer
from ..targets.projection import ProjectionObservation, TargetProjectors
from .checkpoint import load_checkpoint, save_checkpoint
from .distributed import initialize, rank, world_size
from .engine import Trainer
from .monitor import representation_diagnostics


def seed_runtime(seed, deterministic=False):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def json_finite(value):
    """Nonfinite diagnostics are explicit nulls; no model value is repaired."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    return value


def runtime_implementation_hash():
    root = Path(__file__).resolve().parents[1]
    return canonical_hash({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))})


def smoke_inputs(config):
    d, seed = config["cache"]["visual_dim"], config["experiment"]["seed"]
    bundles = {f"fixture{i}": synthetic_features(f"fixture{i}", length=48 - (i % 3) * 9,
                visual_dim=d, seed=seed + i, phase=(i % 2) * 2.0) for i in range(6)}
    groups = {f"fixture-source{i}": (key,) for i, key in enumerate(bundles)}
    descriptor_config = geometry_kwargs(config)
    projectors = None
    if config["geometry"]["enabled"] and config["loss"]["geometry_weight"]:
        records = []
        for i, bundle in enumerate(bundles.values()):
            records.extend(ProjectionObservation(f"fixture-source{i}", "train", f"{g.group}/{g.scale}", g.descriptor)
                for g in fitting_geometry(bundle, config) if g.valid)
        projectors = TargetProjectors.fit(records, allowed_train_sources=set(groups), manifest_sha256=canonical_hash(groups),
            extractor_hash=next(iter(bundles.values())).meta["extractor_hash"], descriptor_config=descriptor_config,
            output_dim=config["geometry"]["projection"]["output_dim"], seed=config["geometry"]["projection"]["seed"])
    return bundles, groups, projectors


def full_clip_batch(bundles, config):
    examples = []
    for bundle in bundles:
        tokens = SignTokenizer().build(bundle)
        raw = MaskSampler().sample(tokens, np.random.default_rng(0))
        plan = ClosureResolver().resolve(tokens, raw)
        examples.append(TrainingExample(bundle, tokens, plan))
    return collate_examples(examples, config["model"]["raw_dimensions"], config["model"]["predictor"]["geometry_output_dim"])


def run_smoke(resolved, device="cpu", resume=None):
    config = resolved.values
    if config["cache"]["kind"] != "synthetic_fixture" or not config["runtime"]["deterministic_smoke"]:
        raise ValueError("smoke requires explicit synthetic_fixture and deterministic_smoke config")
    if config["loss"]["cluster_weight"]:
        raise ValueError("Cluster smoke requires fitted synthetic codebooks; use E02/E04 smoke first")
    device = initialize(device)
    torch.set_num_threads(1)
    seed_runtime(config["experiment"]["seed"], deterministic=config["runtime"]["deterministic_smoke"])
    bundles, sources, projectors = smoke_inputs(config)
    identities = {"scope": "synthetic_correctness_only", "sources_hash": canonical_hash(sources),
                  "extractor_hash": next(iter(bundles.values())).meta["extractor_hash"],
                  "descriptor_hash": canonical_hash(geometry_kwargs(config)),
                  "target_transforms_hash": projectors.fingerprint() if projectors is not None else None,
                  "runtime_implementation_hash": runtime_implementation_hash()}
    identities["world_size"] = world_size()
    provider = BatchProvider(config, bundles, sources, projectors, seed=config["experiment"]["seed"] + rank() * 1009)
    trainer = Trainer(ResearchModel(config), config, device)
    run_identity = canonical_hash({"config_hash": resolved.hash, "identities": identities})
    run_dir = resolved.path(config["experiment"]["output_dir"]) / run_identity[:12]
    if resume:
        checkpoint = Path(resume).resolve()
        if checkpoint.parent.parent != run_dir.resolve():
            raise ValueError("Resume checkpoint is outside the configured run identity")
        provider.load_state_dict(load_checkpoint(checkpoint, trainer, config, identities, mask_generator=provider.mask_generator))
    else:
        # Every rank observes the same existing run state before root creates the directory.
        from .distributed import all_true
        if not all_true(not run_dir.exists(), device):
            raise FileExistsError(f"Run already exists; use --resume: {run_dir}")
        if rank() == 0:
            run_dir.mkdir(parents=True)
            write_json(run_dir / "resolved_config.json", resolved.provenance())
            write_json(run_dir / "environment.json", doctor())
            write_json(run_dir / "identities.json", identities)
            if projectors is not None:
                projectors.save(run_dir / "fixture_targets")
    if world_size() > 1:
        torch.distributed.barrier()
    history, status, started = [], "completed_correctness_smoke", time.monotonic()
    max_attempts = config["training"]["max_successful_steps"] * 10
    try:
        while trainer.successful_step < config["training"]["max_successful_steps"]:
            if trainer.attempted_updates >= max_attempts:
                status = "failed_no_progress_after_repeated_attempts"
                break
            from .distributed import all_true
            if not all_true(trainer.elapsed_runtime < config["runtime"]["max_runtime_hours"] * 3600, device):
                status = "budget_stopped"
                break
            cycle_started = time.monotonic()
            error, batches = None, None
            try:
                batches = [provider.next_batch() for _ in range(config["training"]["gradient_accumulation"])]
            except (ValueError, OSError) as caught:
                error = str(caught)
            if not all_true(error is None, device):
                raise ValueError(f"Data preparation failed on at least one rank: {error or 'see peer rank error'}")
            elapsed_before = trainer.elapsed_runtime
            result = trainer.update(batches)
            trainer.elapsed_runtime += max(0.0, time.monotonic() - cycle_started - (trainer.elapsed_runtime - elapsed_before))
            history.append(result)
            if rank() == 0:
                record = {**asdict(result), "synthetic_correctness_only": True,
                          "mask_counts": [list(batch.diagnostics) for batch in batches]}
                with (run_dir / "logs.jsonl").open("a") as stream:
                    stream.write(json.dumps(json_finite(record), allow_nan=False) + "\n")
            interval = config["training"]["checkpoint_interval"]
            if result.succeeded and trainer.successful_step % interval == 0:
                save_checkpoint(run_dir / "checkpoints/last.pt", trainer, config, identities,
                    mask_generator=provider.mask_generator, data_state=provider.state_dict())
    finally:
        save_checkpoint(run_dir / "checkpoints/last.pt", trainer, config, identities,
            mask_generator=provider.mask_generator, data_state=provider.state_dict())
    summary = {"status": status, "scope": "synthetic_correctness_only", "successful_steps": trainer.successful_step,
        "attempted_updates": trainer.attempted_updates, "attempted_batches": trainer.attempted_batches,
        "precision": trainer.precision, "precision_reason": trainer.precision_reason,
        "world_size": world_size(), "config_hash": resolved.hash, "run_dir": str(run_dir),
        "runtime_seconds_this_session": time.monotonic() - started, "training_runtime_seconds_total": trainer.elapsed_runtime,
        "initial_loss_this_session": history[0].loss if history else None, "final_loss_this_session": history[-1].loss if history else None,
        "geometry_target_count_this_session": sum(sum(sum(log["geometry"]["counts"]) for log in h.logs) for h in history),
        "test_used_for_selection": False, "research_claim": None}
    if rank() == 0:
        trainer.core.eval()
        with torch.inference_mode():
            embeddings = trainer.core.extract_embeddings(full_clip_batch(list(bundles.values()), config).to(device))
        # Frozen-encoder supervised toy head is a correctness gate only.
        # inference_mode tensors must become ordinary tensors before a trainable head
        # saves them for backward; cloning outside that context guarantees this.
        x = embeddings.clip.detach().clone()
        labels = torch.arange(len(x), device=device) % 2
        head = LinearHead(x.shape[-1], 2).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=.03)
        for _ in range(30):
            optimizer.zero_grad()
            logits = head(x)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = head(x)
        np.savez(run_dir / "embeddings.npz", dense=embeddings.dense.cpu().numpy(), timestamps=embeddings.timestamps.cpu().numpy(),
            valid=embeddings.valid.cpu().numpy(), clip=embeddings.clip.cpu().numpy(), synthetic_correctness_only=np.array(True))
        torch.save({"state_dict": head.state_dict(), "encoder_config_hash": resolved.hash, "synthetic_correctness_only": True}, run_dir / "toy_head.pt")
        write_json(run_dir / "toy_predictions.json", {"scope": "synthetic_correctness_only_train_examples", "sample_ids": list(bundles),
            "references": labels.cpu().tolist(), "predictions": logits.argmax(dim=-1).cpu().tolist(), "logits": logits.cpu().tolist(),
            "head_loss": float(loss.detach()), "selection": "not_a_validation_or_test_experiment"})
        summary["diagnostics"] = representation_diagnostics(x)
        write_json(run_dir / "summary.json", json_finite(summary))
    return summary
