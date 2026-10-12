"""Immutable stage records alongside a latest view, with explicit unknown compute fields."""
from datetime import datetime, timezone
from pathlib import Path
import time
from ..cache.hashing import canonical_hash
from ..cli import write_json


def stage_record(root, stage, split, report, *, elapsed_seconds, device=None):
    record = {"schema_version": 1, "stage": stage, "split": split, "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": elapsed_seconds, "device": str(device) if device is not None else None,
        "gpu_hours": None, "gpu_hours_missing_reason": "GPU kernel/exclusive allocation accounting not instrumented",
        "report": report}
    record["record_hash"] = canonical_hash(record)
    path = Path(root) / "stage_ledger" / f"{stage}_{split}_{time.time_ns()}_{record['record_hash'][:12]}.json"
    write_json(path, record)
    return path
