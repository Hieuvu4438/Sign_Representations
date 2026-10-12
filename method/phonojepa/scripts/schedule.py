"""Sequential runner for the approved proposal12 plan (economy, 1 seed, model S, proposal11 protocol). NOT started
automatically — launch only after the user's go:  setsid nohup python scripts/schedule.py > runtime/logs/schedule.log 2>&1 &
Dry run (prints the plan):                         python scripts/schedule.py --dry

Rules (server): one GPU job at a time; wait for our own GPU jobs and for enough free memory; OOM/crash -> retry with
--resume; state in runtime/schedule_state.json (a restarted runner skips finished jobs).
Every training run = 4,000 steps x 32 clips on the 3,000-video ASLC002 corpus (same as obs_jepa A/B and EX16), then
extraction (val, aslc300) with the teacher, then the shared obs_jepa evaluation against SR3D / JH3D / JV3D / B12 / B13.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PJ = Path(__file__).resolve().parents[1]
REPO = PJ.parents[1]
PY = str(REPO / ".venv-signrep" / "bin" / "python")
RT = PJ / "runtime"; LOG = RT / "logs"; EV = RT / "eval"
STATE = RT / "schedule_state.json"
STEPS = 4000
DRY = "--dry" in sys.argv
ONLY = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), None)   # e.g. --only=P1
MAX_TRY = 4
REFS = "SR3D,JH3D,JV3D,ASLC002,B12,B13"

# ------------------------------------------------------------------ the plan (proposal12 §12, decisions §17)
P1 = {   # anti-collapse family (A6) on the MVM cell C-PL-both, + schedule check (P1b, 4x steps)
    "PJ_PLboth": ["--family", "PL", "--target", "both"],                                         # = MVM default (cce)
    "PJ_PLboth_proto": ["--family", "PL", "--target", "both", "--loss", "proto"],
    "PJ_PLboth_l1sig": ["--family", "PL", "--target", "both", "--loss", "l1sig"],
    "PJ_PLboth_l1sig_noema": ["--family", "PL", "--target", "both", "--loss", "l1sig", "--no-ema"],
    "PJ_PLctx_16k": ["--family", "PL", "--target", "ctx", "--steps", "16000"],
    "PJ_PLint_16k": ["--family", "PL", "--target", "int", "--steps", "16000"],
}
P2 = {   # factorial 3 x 3 (+ PJ_PLboth from P1) and matched baselines
    "PJ_PLctx": ["--family", "PL", "--target", "ctx"],
    "PJ_PLint": ["--family", "PL", "--target", "int"],
    "PJ_R4ctx": ["--family", "R4", "--target", "ctx"],          # ~ the original idea (anatomical tubes, contextual target)
    "PJ_R4int": ["--family", "R4", "--target", "int"],
    "PJ_R4both": ["--family", "R4", "--target", "both"],
    "PJ_R0ctx": ["--family", "R0", "--target", "ctx"],
    "PJ_R0int": ["--family", "R0", "--target", "int"],
    "PJ_R0both": ["--family", "R0", "--target", "both"],
    "PJ_Bglob": ["--family", "PL", "--target", "none", "--glob"],
    "PJ_BMAE": ["--family", "R0", "--target", "int", "--baseline-rate", "0.6", "--int-part", "level"],
    "PJ_BMAMP": ["--family", "R0", "--target", "int", "--baseline-rate", "0.9", "--int-part", "delta"],
    "PJ_PLboth_hands": ["--family", "PL", "--target", "both", "--hands-only"],
    "PJ_PLboth_shufint": ["--family", "PL", "--target", "both", "--shuffle-int"],
    "PJ_B1rand": ["--family", "PL", "--target", "both", "--steps", "0"],                       # random-init floor
}
P3 = {   # gates G1-G3 on the MVM cell and the leak/handedness controls
    "PJ_G1glob": ["--family", "PL", "--target", "both", "--glob"],
    "PJ_G2full": ["--family", "PLfull", "--target", "both", "--use-r1", "--curriculum"],
    "PJ_G3morph": ["--family", "PL", "--target", "both", "--morph"],
    "PJ_A8g_randhand": ["--family", "PL", "--target", "both", "--handedness", "random"],
}
PHASES = {"P1": P1, "P2": P2, "P3": P3}


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def say(*a):
    print(now(), *a, flush=True)


def load_state():
    return json.load(open(STATE)) if STATE.exists() else {}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=1))


def free_mib():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def own_gpu_jobs():
    pids, me = [], os.getpid()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == me:
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except Exception:
            continue
        if ".venv-signrep/bin/python" in cmd and any(k in cmd for k in ("pj/train.py", "pj/extract.py", "oj/train_", "oj/extract.py",
                                                                         "jt/train.py", "jt/extract.py", "jt/ex16.py")):
            pids.append(int(p.name))
    return pids


def wait_gpu(need):
    last = 0
    while True:
        busy, fm = own_gpu_jobs(), free_mib()
        if not busy and fm >= need:
            return
        if time.time() - last > 900:
            say(f"waiting: own jobs {busy}, free {fm} MiB < need {need}"); last = time.time()
        time.sleep(60)


def run(name, cmd, need, done, resume_flag=None, cleanup=None):
    if DRY:
        say(f"DRY {name} need={need} done={done()}: {' '.join(cmd[2:])[:200]}"); return True
    st = load_state()
    if st.get(name) == "done" or done():
        st[name] = "done"; save_state(st); return True
    for t in range(1, MAX_TRY + 1):
        wait_gpu(need)
        c = list(cmd) + ([resume_flag] if resume_flag and t > 1 else [])
        say(f"START {name} try {t}")
        with open(LOG / f"{name}.log", "a") as fh:
            fh.write(f"\n===== {now()} try {t}\n"); fh.flush()
            rc = subprocess.call(c, cwd=str(PJ), stdout=fh, stderr=subprocess.STDOUT)
        if rc == 0 and done():
            st = load_state(); st[name] = "done"; save_state(st); say(f"DONE {name}"); return True
        txt = (LOG / f"{name}.log").read_text(errors="ignore")[-20000:]
        oom = "OutOfMemoryError" in txt or "CUDA out of memory" in txt
        say(f"FAIL {name} try {t} rc={rc} oom={oom}")
        if cleanup:
            cleanup()
        time.sleep(300 if oom else 60)
    st = load_state(); st[name] = "failed"; save_state(st); return False


def train_extract(name, args):
    out = RT / "runs" / name
    steps = int(args[args.index("--steps") + 1]) if "--steps" in args else STEPS
    ok = run(f"train_{name}", [PY, "pj/train.py", "--out", str(out), *args, *([] if "--steps" in args else ["--steps", str(STEPS)])],
             6000, lambda: (out / "final.pt").exists(), resume_flag="--resume")
    if not ok:
        return False
    if not DRY:
        m = json.load(open(out / "manifest.json"))
        if m.get("steps_done", -1) < 0.95 * steps:                                   # never evaluate an unfinished run
            say(f"ABORT {name}: steps_done {m.get('steps_done')} < planned {steps}"); return False
    for s in ("val", "aslc300"):
        d = RT / "feats" / f"{name}_grp" / s
        ok &= run(f"extract_{name}_{s}", [PY, "pj/extract.py", "--kind", "model", "--ckpt", str(out / "final.pt"),
                                          "--weights", "teacher", "--name", name, "--set", s], 4000,
                  lambda d=d: (d / "windows.npy").exists(),
                  cleanup=lambda name=name, s=s: [shutil.rmtree(RT / "feats" / f"{name}{k}" / s, ignore_errors=True) for k in ("_grp", "_mean", "_cls")])
    return ok


def evaluate(tag, names):
    out = EV / f"{tag}.json"
    return run(f"eval_{tag}", [PY, "pj/evaluate.py", "--names", ",".join(names), "--refs", REFS, "--out", str(out)], 0,
               lambda: out.exists())


def acquire_lock():
    """single runner: an exclusive flock on runtime/schedule.lock held for the process lifetime (the cron watchdog
    scripts/watchdog.sh only restarts the runner when this lock is free)."""
    import fcntl
    fh = open(RT / "schedule.lock", "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        say("another runner holds the lock -> exit"); sys.exit(0)
    fh.write(str(os.getpid())); fh.flush()
    return fh


def main():
    LOG.mkdir(parents=True, exist_ok=True); EV.mkdir(parents=True, exist_ok=True)
    _lock = acquire_lock() if not DRY else None  # noqa: F841  (kept open)
    say("START schedule", "DRY" if DRY else "")
    # B0 (training-free descriptor) in the same protocol
    for s in ("val", "aslc300"):
        run(f"extract_B0_{s}", [PY, "pj/extract.py", "--kind", "descriptor", "--name", "PJ_B0", "--set", s], 2000,
            lambda s=s: (RT / "feats" / "PJ_B0" / s / "windows.npy").exists())
    for ph, arms in PHASES.items():
        if ONLY and ph != ONLY:
            continue
        for n, args in arms.items():
            train_extract(n, args)
        # evaluate every arm extracted so far (any phase), so e.g. P2 always includes PJ_PLboth from P1
        all_arms = [n for arms_ in PHASES.values() for n in arms_]
        names = ["PJ_B0"] + [f"{n}{k}" for n in all_arms for k in ("_grp", "_cls")
                             if DRY or ((RT / "feats" / f"{n}{k}" / "val" / "windows.npy").exists()
                                        and (RT / "feats" / f"{n}{k}" / "aslc300" / "windows.npy").exists())]
        evaluate(ph, names)
    say("ALL DONE")


if __name__ == "__main__":
    main()
