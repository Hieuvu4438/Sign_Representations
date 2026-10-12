from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import yaml

from model_helpers import ROOT


def test_smoke_console_module_artifacts_and_completed_resume(tmp_path):
    config = tmp_path / "smoke.yaml"
    config.write_text(yaml.safe_dump({"extends": str(ROOT / "configs/smoke.yaml"),
        "experiment": {"output_dir": str(tmp_path / "runs")}, "training": {"max_successful_steps": 2}}))
    command = [sys.executable, "-m", "signpath.cli", "smoke", "--config", str(config), "--device", "cpu"]
    first = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert first.returncode == 0, first.stdout + first.stderr
    summary = json.loads(first.stdout)
    assert summary["scope"] == "synthetic_correctness_only" and summary["successful_steps"] == 2
    run = Path(summary["run_dir"])
    for name in ("resolved_config.json", "environment.json", "checkpoints/last.pt", "embeddings.npz", "toy_predictions.json", "toy_head.pt"):
        assert (run / name).is_file()
    with np.load(run / "embeddings.npz", allow_pickle=False) as arrays:
        assert arrays["dense"].shape[0] == 6 and bool(arrays["synthetic_correctness_only"])
        assert np.isfinite(arrays["dense"]).all()
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert repeated.returncode == 2 and "already exists" in repeated.stderr
    resumed = subprocess.run(command + ["--resume", str(run / "checkpoints/last.pt")], capture_output=True, text=True, timeout=60)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert json.loads(resumed.stdout)["successful_steps"] == 2
    assert len((run / "logs.jsonl").read_text().splitlines()) == 2
