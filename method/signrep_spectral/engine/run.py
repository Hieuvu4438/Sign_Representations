"""Owned run directories: exclusive creation, status, command, source and output manifests."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parents[1]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def package_source_hashes() -> Dict[str, str]:
    out = {}
    for p in sorted(PKG_ROOT.rglob("*.py")):
        if "runtime" in p.parts or "__pycache__" in p.parts:
            continue
        out[str(p.relative_to(REPO_ROOT))] = sha256_bytes(p.read_bytes())
    return out


def git_state() -> Dict[str, Any]:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return None

    return {"commit": run("rev-parse", "HEAD"), "status_short": run("status", "--short")}


class RunContext:
    def __init__(self, run_dir: str | Path, command: str, cfg: Dict[str, Any] | None):
        self.dir = Path(run_dir)
        if not self.dir.is_absolute():
            self.dir = REPO_ROOT / self.dir
        self.dir.parent.mkdir(parents=True, exist_ok=True)
        self.dir.mkdir(exist_ok=False)  # exclusive: never reuse a run directory
        self.t0 = time.time()
        self.cpu0 = time.process_time()
        self._write("command.json", {
            "command": command,
            "argv": sys.argv,
            "python": sys.executable,
            "python_version": sys.version,
            "host": platform.node(),
            "cwd": os.getcwd(),
            "started_unix": self.t0,
        })
        if cfg is not None:
            from ..config import config_hash

            self._write("resolved_config.json", cfg)
            self.config_sha256 = config_hash(cfg)
        self._write("source_manifest.json", {"git": git_state(), "package_sources": package_source_hashes()})
        self.status("RUNNING")

    def _write(self, name: str, obj: Any) -> Path:
        p = self.dir / name
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str)
        os.replace(tmp, p)
        return p

    def write_json(self, name: str, obj: Any) -> Path:
        return self._write(name, obj)

    def status(self, state: str, **extra) -> None:
        assert state in ("RUNNING", "SUCCESS", "FAILED", "BUDGET_STOP", "BLOCKED")
        self._write("status.json", {
            "state": state,
            "wall_seconds": time.time() - self.t0,
            "cpu_seconds": time.process_time() - self.cpu0,
            "updated_unix": time.time(),
            **extra,
        })

    def finalize_outputs(self) -> None:
        files = {}
        for p in sorted(self.dir.rglob("*")):
            if p.is_file() and p.name not in ("outputs_manifest.json",):
                files[str(p.relative_to(self.dir))] = {"bytes": p.stat().st_size}
        self._write("outputs_manifest.json", {"files": files, "total_bytes": sum(v["bytes"] for v in files.values())})
