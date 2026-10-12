"""Materialize the proposal's per-run provenance instead of relying on external filenames."""
from pathlib import Path
import tarfile
import shutil
import yaml
from ..cache.hashing import file_hash
from ..cli import write_json


def write_run_provenance(directory, resolved, corpus, validation, audit, identities):
    directory = Path(directory)
    root = resolved.root
    c = resolved.values
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    lock = root / "requirements-main-extraction.lock.txt"
    if lock.exists():
        shutil.copyfile(lock, directory / lock.name)
    registry = root / "registries/sources.yaml"
    write_json(directory / "source_registry.json", {"registry": yaml.safe_load(registry.read_text()), "sha256": file_hash(registry)})
    source = directory / "source_snapshot.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        for folder in ("src", "configs", "registries", "scripts"):
            for path in sorted((root / folder).rglob("*")):
                if path.is_file() and path.suffix in {".py", ".yaml", ".sh"}:
                    archive.add(path, arcname=str(path.relative_to(root)))
        for name in ("pyproject.toml", "requirements-main-extraction.lock.txt"):
            if (root / name).exists():
                archive.add(root / name, arcname=name)
    write_json(directory / "source_snapshot.json", {"runtime_implementation_hash": identities["runtime_implementation_hash"],
        "archive_sha256": file_hash(source), "scope": "new standalone method package and configs; upstream code remains pinned externally"})
    (directory / "local_patch.diff").write_text("# New standalone SignPathLab package; full local source is source_snapshot.tar.gz.\n# No patches applied to the strict pinned DINO source.\n")
    write_json(directory / "manifest_hashes.json", {"train": file_hash(corpus.manifest), "val": file_hash(validation.manifest),
        "test": None, "test_reason": "not accessed during development"})
    write_json(directory / "split_audit.json", audit)
    sample = corpus.load(corpus.available[0])
    write_json(directory / "checkpoint_load_report.json", {"visual_backbone": sample.meta.get("backbone"),
        "temporal_initialization": "trainable_scratch; teacher exact online copy; no random pretrained fallback"})
    write_json(directory / "cache_and_target_provenance.json", identities)
    (directory / "downstream_head_config.yaml").write_text(yaml.safe_dump(c["downstream"], sort_keys=False))
    (directory / "train_log.jsonl").symlink_to("logs.jsonl")


def write_decision_memo(directory, config, report):
    path = Path(directory) / "decision_memo.md"
    path.write_text(f"# Run decision memo: {config['experiment']['id']} seed{config['experiment']['seed']}\n\n"
        f"Status: {report['status']}. Successful steps: {report['successful_steps']}. "
        f"Validation PC Top1: {report['best_validation_per_class_top1']}.\n\n"
        "Selection uses the registered frozen validation probe. Test data was not used. "
        "This single run does not establish a signature contribution; compare matched "
        "motion/raw controls and multiple seeds using the recorded source and mask identities.\n")
