from pathlib import Path
import json
import subprocess
import sys


def test_deployment_cli_help_and_causal_missing_run(tmp_path):
    for command in ("embed", "export"):
        help_result = subprocess.run([sys.executable, "-m", "signpath.cli", command, "--help"], capture_output=True, text=True)
        assert help_result.returncode == 0 and "--run" in help_result.stdout
    result = subprocess.run([sys.executable, "-m", "signpath.cli", "export", "--run", str(tmp_path / "missing"),
        "--output", str(tmp_path / "output"), "--format", "unsupported"], capture_output=True, text=True)
    assert result.returncode == 2 and "Only torch_export" in result.stderr
    assert not (tmp_path / "output").exists()
