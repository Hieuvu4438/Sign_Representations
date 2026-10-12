"""Real full-clip preprocessing caches -> JEPA optimizer -> checkpoint -> embeddings/head.

Train examples only. This is an integration gate, not an evaluation/pilot score.
"""
import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch

from signpath.cache.hashing import canonical_hash, file_hash
from signpath.cache.reader import read_bundle
from signpath.cache.schema import IDENTITY_FIELDS
from signpath.cli import doctor, write_json
from signpath.config import load_config
from signpath.data.dataset import BatchProvider, fitting_geometry, geometry_kwargs
from signpath.data.manifest import ManifestRecord, write_manifest
from signpath.geometry.paths import geometry_from_features
from signpath.models.heads import LinearHead
from signpath.models.teacher_student import ResearchModel
from signpath.targets.projection import ProjectionObservation, TargetProjectors
from signpath.training.checkpoint import load_checkpoint, save_checkpoint
from signpath.training.engine import Trainer
from signpath.training.monitor import representation_diagnostics
from signpath.training.runner import full_clip_batch, json_finite, runtime_implementation_hash, seed_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", action="append", required=True)
    parser.add_argument("--official-train-csv", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    resolved = load_config(args.config)
    config = resolved.values
    if config["cache"]["kind"] != "per_frame_dino" or config["runtime"]["allow_missing_provenance"]:
        raise ValueError("Real integration gate requires actual features and complete provenance")
    torch.set_num_threads(1)
    seed_runtime(config["experiment"]["seed"])
    run_identity = canonical_hash({"config_hash": resolved.hash, "runtime_implementation_hash": runtime_implementation_hash(),
                                   "script_sha256": file_hash(__file__)})
    run = resolved.path(config["experiment"]["output_dir"]) / run_identity[:12]
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with Path(args.official_train_csv).open(newline="") as stream:
        annotations = list(csv.DictReader(stream))
    by_stem = {Path(row["Video file"]).stem: row for row in annotations}
    taxonomy = {label: i for i, label in enumerate(sorted({row["Gloss"] for row in annotations}))}
    annotation_hash = file_hash(args.official_train_csv)
    expected, bundles, sources, records = None, {}, {}, []
    for directory in args.cache:
        meta = json.loads((Path(directory) / "meta.json").read_text())
        identity = {key: meta[key] for key in IDENTITY_FIELDS}
        if expected is None:
            expected = identity
        elif expected != identity:
            raise ValueError("All diagnostic clips must share identical foundation/preprocessing assets")
        bundle = read_bundle(directory, expected)
        name = bundle.meta["sample_id"]
        if name not in by_stem:
            raise ValueError(f"Clip is not in supplied official train split: {name}")
        row = by_stem[name]
        video_path = Path(args.video_root) / row["Video file"]
        if file_hash(video_path) != bundle.meta["source_sha256"]:
            raise ValueError("Current raw video differs from cached source")
        source = "asl_citizen:" + name
        records.append(ManifestRecord(1, name, "asl_citizen", "ase", "train", source, str(video_path.resolve()),
            source_sha256=bundle.meta["source_sha256"], signer_id=row["Participant ID"], label_id=taxonomy[row["Gloss"]],
            gloss=row["Gloss"], official_split_version="csv_sha256:" + annotation_hash, availability="available"))
        bundles[name], sources[source] = bundle, (name,)
    manifest = run / "train_diagnostic.jsonl"
    write_manifest(manifest, records)
    observations = []
    for record in records:
        # No gloss/text/labels are available to geometry fitting or the SSL provider.
        view = record.pretrain_view()
        observations.extend(ProjectionObservation(view.source_group_id, "train", f"{g.group}/{g.scale}", g.descriptor)
            for g in fitting_geometry(bundles[view.sample_id], config) if g.valid)
    projectors = TargetProjectors.fit(observations, allowed_train_sources=set(sources), manifest_sha256=file_hash(manifest),
        extractor_hash=expected["extractor_hash"], descriptor_config=geometry_kwargs(config),
        output_dim=config["geometry"]["projection"]["output_dim"], seed=config["geometry"]["projection"]["seed"])
    projectors.save(run / "train_targets")
    provider = BatchProvider(config, bundles, sources, projectors)
    trainer = Trainer(ResearchModel(config), config, args.device)
    identities = {"scope": "real_minibatch_correctness_only", "manifest_sha256": file_hash(manifest),
        "annotation_sha256": annotation_hash, "feature_identity": expected,
        "targets_metadata_sha256": file_hash(run / "train_targets/metadata.json"),
        "runtime_implementation_hash": runtime_implementation_hash(), "script_sha256": file_hash(__file__)}
    write_json(run / "resolved_config.json", resolved.provenance())
    write_json(run / "identities.json", identities)
    write_json(run / "environment.json", doctor())
    results = []
    while trainer.successful_step < config["training"]["max_successful_steps"]:
        if trainer.attempted_updates > 10 * config["training"]["max_successful_steps"]:
            raise RuntimeError("Too many empty/nonfinite updates in real integration gate")
        batch = provider.next_batch()
        result = trainer.update([batch])
        results.append(result)
        with (run / "logs.jsonl").open("a") as stream:
            stream.write(json.dumps(json_finite({**asdict(result), "mask_counts": list(batch.diagnostics)}), allow_nan=False) + "\n")
    checkpoint = run / "checkpoints/last.pt"
    save_checkpoint(checkpoint, trainer, config, identities, mask_generator=provider.mask_generator, data_state=provider.state_dict())
    reloaded = Trainer(ResearchModel(config), config, args.device)
    provider.load_state_dict(load_checkpoint(checkpoint, reloaded, config, identities, mask_generator=provider.mask_generator))
    for name, value in trainer.core.state_dict().items():
        torch.testing.assert_close(value, reloaded.core.state_dict()[name], atol=0, rtol=0)
    reloaded.core.eval()
    full = full_clip_batch(list(bundles.values()), config).to(args.device)
    with torch.inference_mode():
        embeddings = reloaded.core.extract_embeddings(full)
    np.savez(run / "embeddings.npz", dense=embeddings.dense.cpu().numpy(), timestamps=embeddings.timestamps.cpu().numpy(),
        valid=embeddings.valid.cpu().numpy(), clip=embeddings.clip.cpu().numpy(), model_hash=np.array(file_hash(checkpoint)),
        preprocessing_hash=np.array(expected["extractor_hash"]))
    clip = embeddings.clip.detach().clone()
    # Labels are first consumed here, by a separate frozen downstream classifier.
    global_labels = sorted({record.label_id for record in records})
    local_labels = {label: i for i, label in enumerate(global_labels)}
    labels = torch.tensor([local_labels[record.label_id] for record in records], device=args.device)
    head = LinearHead(clip.shape[-1], len(global_labels)).to(args.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=.03)
    for _ in range(30):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(clip), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = head(clip)
    torch.save({"head": head.state_dict(), "class_ids": global_labels, "encoder_hash": file_hash(checkpoint)}, run / "head.pt")
    write_json(run / "predictions.json", {"scope": "real_train_minibatch_correctness_only", "split": "train",
        "sample_ids": list(bundles), "reference_ids": [record.label_id for record in records],
        "predicted_ids": [global_labels[index] for index in logits.argmax(dim=-1).cpu().tolist()],
        "logits": logits.cpu().tolist(), "class_ids": global_labels, "encoder_hash": file_hash(checkpoint)})
    report = {"status": "passed", "scope": "real_train_minibatch_correctness_only", "successful_steps": trainer.successful_step,
        "attempted_updates": trainer.attempted_updates, "checkpoint_reload_exact": True,
        "clip_lengths": full.lengths.cpu().tolist(), "source_hashes": [r.source_sha256 for r in records],
        "initial_loss": results[0].loss, "final_loss": results[-1].loss,
        "geometry_targets_total": sum(sum(sum(log["geometry"]["counts"]) for log in result.logs) for result in results),
        "head_loss_on_train_minibatch": float(loss.detach()), "dense_shape": list(embeddings.dense.shape),
        "config_hash": resolved.hash, "feature_identity": expected, "precision": trainer.precision,
        "runtime_seconds": time.monotonic() - started, "research_claim": None, "test_used_for_selection": False,
        "diagnostics": representation_diagnostics(clip), "run_dir": str(run)}
    write_json(run / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
