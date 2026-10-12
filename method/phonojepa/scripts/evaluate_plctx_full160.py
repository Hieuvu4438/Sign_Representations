"""Extract and evaluate the authorized full-train PLctx checkpoint on validation."""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PY = REPO / ".venv-signrep/bin/python"
NAME = "PJ_PLctx_FULL_160ep"
RUN = ROOT / "runtime/runs" / NAME
LAUNCH = ROOT / "runtime/launch"
STATUS = LAUNCH / f"{NAME}.evaluation.status.json"
OUT = ROOT / "runtime/eval" / f"{NAME}.json"
REFS = ["PJ_PLctx_16k_grp", "PJ_PLctx_grp", "PJ_PLint_16k_grp", "JH3D", "JV3D", "SR3D", "B13"]


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def write(state):
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATUS)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if STATUS.exists() or OUT.exists():
        raise FileExistsError("Existing evaluation; refusing duplicate launch")
    manifest = json.loads((RUN / "manifest.json").read_text())
    assert manifest["steps_done"] == manifest["steps_planned"] == 200640
    assert manifest["n_clips"] == 40154
    for suffix in ("_grp", "_cls", "_mean"):
        assert not (ROOT / "runtime/feats" / f"{NAME}{suffix}").exists()
    for ref in REFS:
        for ext in ("json", "npz"):
            assert (ROOT / "runtime/eval/models" / f"{ref}.{ext}").exists(), ref
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    state = {"status": "running", "launcher_pid": os.getpid(), "started": now(),
             "checkpoint": str(RUN / "final.pt"), "checkpoint_sha256": digest(RUN / "final.pt"),
             "pretraining_clips": 40154, "weights": "teacher", "protocol": "both",
             "note": "Same evaluation sets/metrics as earlier runs; larger pretraining corpus and budget.",
             "references": REFS, "stages": [],
             "source_sha256": {str(p.relative_to(ROOT)): digest(p)
                               for p in sorted((ROOT / "pj").rglob("*.py"))}}
    write(state)

    def run_stage(phase, command):
        record = {"phase": phase, "status": "running", "started": now(),
                  "command": list(map(str, command)),
                  "log": str(LAUNCH / f"{NAME}.{phase}.log")}
        state["current_stage"] = phase; state["stages"].append(record); write(state)
        start = time.monotonic()
        with open(record["log"], "x") as stream:
            child = subprocess.Popen(record["command"], cwd=ROOT, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT)
            record["pid"] = child.pid; write(state)
            print(now(), phase, "PID", child.pid, flush=True)
            code = child.wait()
        record.update(status="complete" if code == 0 else "failed", exit_code=code,
                      finished=now(), seconds=round(time.monotonic() - start, 2))
        write(state)
        if code:
            raise RuntimeError(f"{phase} exited {code}: {record['log']}")

    try:
        for split in ("val", "aslc300"):
            run_stage(f"extract_{split}", [PY, "-u", "pj/extract.py", "--kind", "model", "--ckpt", RUN / "final.pt",
                      "--weights", "teacher", "--name", NAME, "--set", split])
        import numpy as np
        for suffix in ("_grp", "_cls", "_mean"):
            for split, nclips in (("val", 10304), ("aslc300", 4500)):
                folder = ROOT / "runtime/feats" / f"{NAME}{suffix}" / split
                offsets = np.load(folder / "offsets.npy")
                windows = np.load(folder / "windows.npy", mmap_mode="r")
                assert len(offsets) == nclips + 1 and offsets[-1] == len(windows)
                assert np.isfinite(windows).all()
                expected = (ROOT / "runtime/feats/PJ_PLctx_16k_grp" / split / "clips.jsonl").read_bytes()
                assert (folder / "clips.jsonl").read_bytes() == expected, "Evaluation row/order mismatch"
        run_stage("evaluate", [PY, "-u", "pj/evaluate.py", "--names", f"{NAME}_grp,{NAME}_cls",
                  "--refs", ",".join(REFS), "--protocol", "both", "--out", OUT])
        results = json.loads(OUT.read_text())
        for name in (f"{NAME}_grp", f"{NAME}_cls"):
            d = results["models"][name]
            assert d["loso"]["white512"]["n"] == 10304
            assert d["aslc300"]["retrieval_white512"]["n"] == 1500
            assert all(math.isfinite(value) for value in (
                d["loso"]["white512"]["R@1"], d["aslc300"]["retrieval_white512"]["R@1"],
                d["aslc300"]["linear_probe"]["macro_top1"]))
        state.update(status="complete", finished=now(), result=str(OUT)); write(state)
        print(now(), "EVALUATION COMPLETE", OUT, flush=True)
    except Exception as exc:
        state.update(status="failed", error=repr(exc), finished=now()); write(state)
        raise


if __name__ == "__main__":
    main()
