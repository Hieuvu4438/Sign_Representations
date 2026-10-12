"""Authorized PLctx run: full official train split, batch32, 160 epochs, one seed."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PY = REPO / ".venv-signrep/bin/python"
NAME = "PJ_PLctx_FULL_160ep"
LAUNCH = ROOT / "runtime/launch"
RUN = ROOT / "runtime/runs" / NAME
CACHE = ROOT / "runtime/cache/aslc_train_full.npz"
STATUS = LAUNCH / f"{NAME}.status.json"
EPOCHS, BATCH, N_CLIPS = 160, 32, 40154
STEPS_PER_EPOCH = N_CLIPS // BATCH  # existing DataLoader uses drop_last=True
STEPS = EPOCHS * STEPS_PER_EPOCH


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def write_status(state):
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATUS)


def gpu_training_pids():
    """Coordinate with the existing sequential scheduler without interrupting it."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            if not argv or b"python" not in argv[0]:
                continue
            if any(a.endswith((b"/train.py", b"/extract.py", b"/train_graph.py", b"/train_rgb.py"))
                   or a in (b"s2.train", b"s2.extract", b"s1.train", b"s1.extract") for a in argv[1:]):
                found.append(int(entry.name))
        except (OSError, ValueError):
            continue
    return found


def main():
    LAUNCH.mkdir(parents=True, exist_ok=True)
    if STATUS.exists() or RUN.exists():
        raise FileExistsError("Run/status already exists; refusing duplicate launch")
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    config = {"name": NAME, "family": "PL", "target": "ctx", "loss": "cce", "seed": 0,
              "epochs": EPOCHS, "n_clips": N_CLIPS, "batch": BATCH,
              "steps_per_epoch": STEPS_PER_EPOCH, "steps": STEPS,
              "drop_last": True, "omitted_per_epoch": N_CLIPS % BATCH,
              "lr": 3e-4, "wd": 0.05, "workers": 6, "d": 256, "depth": 6,
              "heads": 4, "dp": 128, "pdepth": 4, "cache": str(CACHE),
              "init": "scratch", "precision": "bf16", "save_every_steps": 500,
              "estimated_train_hours": 6.2, "estimate_is_not_a_hard_time_limit": True}
    state = {"run": NAME, "launcher_pid": os.getpid(), "started": now(),
             "status": "running", "config": config, "stages": []}
    state["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted((ROOT / "pj").rglob("*.py"))}
    (LAUNCH / f"{NAME}.config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_status(state)

    def run_stage(phase, command, stage_env):
        record = {"phase": phase, "started": now(), "command": list(map(str, command)),
                  "status": "running", "log": str(LAUNCH / f"{NAME}.{phase}.log")}
        state["current_stage"] = phase; state["stages"].append(record); write_status(state)
        with open(record["log"], "x") as stream:
            child = subprocess.Popen(record["command"], cwd=ROOT, env=stage_env,
                                     stdout=stream, stderr=subprocess.STDOUT)
            record["pid"] = child.pid; write_status(state)
            print(now(), phase, "PID", child.pid, flush=True)
            code = child.wait()
        record.update(exit_code=code, finished=now(), status="complete" if code == 0 else "failed")
        write_status(state)
        if code:
            raise RuntimeError(f"{phase} exited {code}; see {record['log']}")

    try:
        if not CACHE.exists():
            cache_env = dict(env, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            run_stage("cache", [PY, "scripts/build_full_train_cache.py", "--out", CACHE, "--workers", "4"], cache_env)
        import numpy as np
        with np.load(CACHE, allow_pickle=False) as cache:
            assert len(cache["ids"]) == N_CLIPS
        provenance = json.loads(CACHE.with_suffix(".manifest.json").read_text())
        assert provenance["n_clips"] == N_CLIPS and provenance["train_val_test_ids_disjoint"]
        state["cache_provenance"] = provenance; write_status(state)
        while True:
            jobs = gpu_training_pids()
            free = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
            if not jobs and free >= 6000:
                break
            state.update(status="waiting_gpu", blocking_pids=jobs, free_gpu_mib=free); write_status(state)
            print(now(), "waiting GPU", jobs, free, flush=True)
            time.sleep(15)
        state.update(status="running", blocking_pids=[]); write_status(state)
        command = [PY, "-u", "pj/train.py", "--out", RUN, "--cache", CACHE,
                   "--family", "PL", "--target", "ctx", "--loss", "cce",
                   "--steps", str(STEPS), "--batch", str(BATCH), "--seed", "0",
                   "--lr", "0.0003", "--wd", "0.05", "--workers", "6",
                   "--save-every", "500", "--log-every", "50"]
        run_stage("train", command, env)
        manifest = json.loads((RUN / "manifest.json").read_text())
        assert manifest["steps_done"] == STEPS and (RUN / "final.pt").exists()
        state.update(status="complete", finished=now()); write_status(state)
        print(now(), "TRAIN COMPLETE", flush=True)
    except Exception as exc:
        state.update(status="failed", error=repr(exc), finished=now()); write_status(state)
        raise


if __name__ == "__main__":
    main()
