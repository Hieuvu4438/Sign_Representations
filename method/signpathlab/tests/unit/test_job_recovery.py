import json
from pathlib import Path
import pytest
from signpath.training.recovery import (choose_resume_checkpoint, reconcile_logs,
    validate_job_restart, quarantine_uncommitted_targets)


def test_recovery_keeps_only_committed_attempts_and_archives_later_or_partial_records(tmp_path):
    for name in ("logs.jsonl", "mask_diagnostics.jsonl"):
        (tmp_path / name).write_text('\n'.join([json.dumps({"attempted_batches": x}) for x in (4, 8, 12)]) + '\n{"partial":')
    reconcile_logs(tmp_path, 8)
    for name in ("logs.jsonl", "mask_diagnostics.jsonl"):
        assert [json.loads(x)["attempted_batches"] for x in (tmp_path / name).read_text().splitlines()] == [4, 8]
    archives = list((tmp_path / "recovery").glob("*.jsonl"))
    assert len(archives) == 2 and all('"partial"' in p.read_text() for p in archives)


def test_checkpoint_selection_fallback_and_initial_publication_recovery(tmp_path):
    run = tmp_path / "one_run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "identities.json").write_text("{}")
    assert choose_resume_checkpoint(tmp_path) is None
    assert list((tmp_path / ".recovery").glob("one_run_startup_*"))
    (run / "checkpoints").mkdir(parents=True)
    (run / "identities.json").write_text("{}")
    (run / "checkpoints/step000000100.pt").write_bytes(b"test")
    assert choose_resume_checkpoint(tmp_path).name == "step000000100.pt"
    (run / "checkpoints/last.pt").write_bytes(b"test")
    assert choose_resume_checkpoint(tmp_path).name == "last.pt"


def test_automatic_resume_accepts_interruption_but_refuses_scientific_failure_or_identity_change():
    previous = {"status": "interrupted", "config_hash": "a", "runtime_implementation_hash": "b", "evaluation_split": "val"}
    validate_job_restart(previous, "a", "b", "val", automatic=True)
    for status in ("failed", "stopped_by_user"):
        with pytest.raises(ValueError, match="refuses"):
            validate_job_restart({**previous, "status": status}, "a", "b", "val", automatic=True)
    with pytest.raises(ValueError, match="changed"):
        validate_job_restart(previous, "a", "other", "val", automatic=True)


def test_partial_target_files_are_archived_before_regeneration(tmp_path):
    first, second = tmp_path / "stats.npz", tmp_path / "matrices.npz"
    first.write_bytes(b"incomplete")
    archive = quarantine_uncommitted_targets((first, second), tmp_path / "recovery")
    assert not first.exists() and (archive / "0_stats.npz").read_bytes() == b"incomplete"
