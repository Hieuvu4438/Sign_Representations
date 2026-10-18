"""Recovery is identity checked and preserves discarded post-checkpoint logs."""
import json
from pathlib import Path
import time


def reconcile_logs(directory, attempted_batches):
    directory = Path(directory)
    for name in ("logs.jsonl", "mask_diagnostics.jsonl"):
        path = directory / name
        if not path.exists():
            continue
        retained, abandoned = [], []
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
                keep = record.get("attempted_batches", float("inf")) <= attempted_batches
            except ValueError:
                keep = False
            (retained if keep else abandoned).append(line)
        if abandoned:
            archive = directory / "recovery"
            archive.mkdir(exist_ok=True)
            (archive / f"{time.time_ns()}_{name}").write_text("\n".join(abandoned) + "\n")
            temp = path.with_suffix(path.suffix + ".recovery.tmp")
            temp.write_text("\n".join(retained) + ("\n" if retained else ""))
            temp.replace(path)


def choose_resume_checkpoint(output_root):
    """Prefer last; fall back to durable step snapshot after abrupt power loss."""
    runs = [p for p in Path(output_root).glob("*") if p.is_dir() and (p / "identities.json").exists()]
    if len(runs) > 1:
        raise ValueError("Multiple scientific run identities under output root; choose explicitly")
    if not runs:
        return None
    directory = runs[0]
    last = directory / "checkpoints/last.pt"
    if last.exists():
        return last
    steps = sorted((directory / "checkpoints").glob("step*.pt"))
    if steps:
        return steps[-1]
    logs = directory / "logs.jsonl"
    if not logs.exists() or not logs.stat().st_size:
        # Initial checkpoint publication can be interrupted before update1. Keep
        # startup evidence, then recreate the identical seeded initialization.
        archive = Path(output_root) / ".recovery"
        archive.mkdir(exist_ok=True)
        directory.rename(archive / f"{directory.name}_startup_{time.time_ns()}")
        return None
    raise ValueError("Initialized run has no durable checkpoint; inspect startup artifacts before recovery")


def quarantine_uncommitted_targets(paths, archive):
    """Only used when the target metadata commit marker is absent on resume."""
    existing = [Path(p) for p in paths if Path(p).exists()]
    if existing:
        destination = Path(archive) / f"partial_targets_{time.time_ns()}"
        destination.mkdir(parents=True, exist_ok=False)
        for index, path in enumerate(existing):
            path.rename(destination / f"{index}_{path.name}")
        return destination
    return None


def validate_job_restart(previous, config_hash, source_hash, evaluation_split, *, automatic=False):
    if previous["config_hash"] != config_hash or previous["runtime_implementation_hash"] != source_hash:
        raise ValueError("Resume job source/config changed")
    if previous.get("evaluation_split", evaluation_split) != evaluation_split:
        raise ValueError("Resume evaluation split changed")
    if automatic and previous["status"] in {"failed", "stopped_by_user"}:
        raise ValueError("Automatic restart refuses a scientific failure or user-stopped job; explicit review/resume required")
