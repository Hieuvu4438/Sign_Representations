"""Job scheduler for the proposal11 vs SignRep plan (docs/proposal11/BAO_CAO_CHIEN_LUOC_SO_SANH_SIGNREP.md), 1 seed.

- one GPU job at a time (user rule); before every job: wait for the previous foreign/own job to end and for enough
  free GPU memory (other users share the GPU);
- OOM / crash -> wait and retry (training jobs resume from their ckpt when the trainer supports --resume);
- state in runtime/schedule_state.json: a restarted scheduler skips finished jobs (server crashes);
- gates decided automatically with rules fixed in the report (recipe choice, B gate, best-A choice).
Launch:  setsid nohup python oj/schedule.py > runtime/logs/schedule.log 2>&1 &
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

OJ = Path(__file__).resolve().parents[1]
REPO = OJ.parents[1]
JT = REPO / "method" / "jepa_targets"
PY = str(REPO / ".venv-signrep" / "bin" / "python")
RT = OJ / "runtime"; LOG = RT / "logs"; EV = RT / "eval"
STATE = RT / "schedule_state.json"
STEPS, BATCH, MICRO = 4000, 32, 16
MAX_TRY = 6
DRY = "--dry" in sys.argv


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
    """our own GPU python jobs started outside this scheduler (e.g. a previous runner's child)."""
    pids = []
    me = os.getpid()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == me:
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            if p.stat().st_uid != os.getuid():
                continue
        except Exception:
            continue
        if ".venv-signrep/bin/python" in cmd and any(k in cmd for k in ("jt/ex16.py", "jt/train.py", "jt/extract.py", "jt/evaluate.py",
                                                                         "oj/train_rgb.py", "oj/train_graph.py", "oj/extract.py")):
            pids.append(int(p.name))
    return pids


def wait_gpu(need):
    t0 = time.time(); last = 0
    while True:
        busy = own_gpu_jobs(); fm = free_mib()
        if not busy and fm >= need:
            return
        if time.time() - last > 900:
            say(f"waiting: own jobs {busy}, free {fm} MiB < need {need}" if busy or fm < need else "waiting"); last = time.time()
        time.sleep(60)


def run(name, cmd, cwd, need, done, cleanup=None, resume_flag=None):
    """run one job until `done()` holds; returns True on success."""
    if DRY:
        say(f"DRY {name} (need {need} MiB, done={done()}): {' '.join(map(str, cmd))[:220]}"); return True
    st = load_state()
    if st.get(name) == "done" or done():
        st[name] = "done"; save_state(st); return True
    for t in range(1, MAX_TRY + 1):
        wait_gpu(need)
        if done():  # finished meanwhile (e.g. by a previous runner's job we waited for): never re-run / clean up
            st = load_state(); st[name] = "done"; save_state(st); say(f"DONE {name} (found finished)"); return True
        c = list(cmd)
        if resume_flag and t > 1:
            c.append(resume_flag)
        say(f"START {name} try {t}: {' '.join(c[1:])[:200]}")
        with open(LOG / f"{name}.log", "a") as fh:
            fh.write(f"\n===== {now()} try {t}\n"); fh.flush()
            rc = subprocess.call(c, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT)
        if rc == 0 and done():
            st = load_state(); st[name] = "done"; save_state(st); say(f"DONE {name}"); return True
        txt = (LOG / f"{name}.log").read_text(errors="ignore")[-20000:]
        oom = "OutOfMemoryError" in txt or "CUDA out of memory" in txt
        say(f"FAIL {name} try {t} rc={rc} oom={oom}")
        if cleanup and not resume_flag:
            cleanup()
        time.sleep(300 if oom else 60)
    st = load_state(); st[name] = "failed"; save_state(st); return False


def exists(p):
    return lambda: Path(p).exists()


def rm(p):
    return lambda: shutil.rmtree(p, ignore_errors=True)


# ------------------------------------------------------------------ job builders
def extract(name, kind, ckpt, weights, sets=("val", "aslc300"), size=224):
    ok = True
    for s in sets:
        d = RT / "feats" / name / s
        ok &= run(f"extract_{name}_{s}", [PY, "oj/extract.py", "--kind", kind, "--ckpt", str(ckpt), "--weights", weights,
                                          "--name", name, "--set", s, "--size", str(size)],
                  OJ, 8000, exists(d / "windows.npy"), cleanup=rm(d))
    return ok


def evaluate(tag, names, refs=("SR3D",)):
    out = EV / f"{tag}.json"
    return run(f"eval_{tag}", [PY, "oj/evaluate.py", "--names", ",".join(names), "--refs", ",".join(refs), "--out", str(out)],
               OJ, 4000, exists(out))


def train_a(name, recipe, policy, weighting, size=224, extra=()):
    out = RT / name
    cmd = [PY, "oj/train_rgb.py", "--out", str(out), "--recipe", recipe, "--policy", policy, "--weighting", weighting,
           "--size", str(size), "--steps", str(STEPS), "--batch", str(BATCH), "--micro", str(MICRO if size == 224 else 4), *extra]
    need = 20000 if size == 224 else 32000
    if not run(f"train_{name}", cmd, OJ, need, exists(out / "final.pt"), resume_flag="--resume"):
        return False
    return extract(name, "rgb", out / "final.pt", "student", size=size)


def train_b(name, objective):
    out = RT / name
    cmd = [PY, "oj/train_graph.py", "--out", str(out), "--objective", objective, "--steps", str(STEPS), "--batch", str(BATCH)]
    if not run(f"train_{name}", cmd, OJ, 12000, exists(out / "final.pt"), resume_flag="--resume"):
        return False
    return extract(name, "graph", out / "final.pt", "teacher")


def metric(name):
    """primary (ASLC300 linear-probe macro Top-1) and co-primary (LOSO white512 R@1) of a cached model."""
    js = EV / "models" / f"{name}.json"
    if not js.exists():
        return None
    r = json.load(open(js))
    try:
        return r["aslc300"]["linear_probe"]["macro_top1"], r["loso"]["white512"]["R@1"]
    except KeyError:
        return None


def decide(tag, value):
    if DRY:
        say(f"DRY decision {tag}: {value}"); return
    st = load_state(); st.setdefault("decisions", {})[tag] = value; save_state(st); say(f"DECISION {tag}: {value}")


def main():
    for d in (LOG, EV, RT / "feats"):
        d.mkdir(parents=True, exist_ok=True)
    say("scheduler start", os.getpid())
    # ---- finish EX16 (old runner stopped; its JH3D extraction child may still be running)
    run("ex16_ext_JH3D", [PY, "jt/extract.py", "--name", "JH3D", "--kind", "hiera", "--ckpt", "runtime/JH3D/last.pt", "--weights", "model"],
        JT, 8000, exists(JT / "runtime/feats/JH3D/windows.npy"), cleanup=rm(JT / "runtime/feats/JH3D"))
    run("ex16_ext_JH3D_ema", [PY, "jt/extract.py", "--name", "JH3D_ema", "--kind", "hiera", "--ckpt", "runtime/JH3D/last.pt", "--weights", "ema_model"],
        JT, 8000, exists(JT / "runtime/feats/JH3D_ema/windows.npy"), cleanup=rm(JT / "runtime/feats/JH3D_ema"))
    run("ex16_eval", [PY, "jt/evaluate.py", "--feats", "SR3D,JV3D,JH3D,JV3D_ema,JH3D_ema,ASLC002", "--ref", "SR3D", "--out", "runtime/eval_ex16.json"],
        JT, 4000, exists(JT / "runtime/eval_ex16.json"))
    # ---- P0: SR-VJ + baselines in the obs_jepa protocol
    run("train_SRVJ", [PY, "jt/ex16.py", "--arm", "srvj", "--out", "runtime/SRVJ"], JT, 20000, exists(JT / "runtime/SRVJ/last.pt"),
        cleanup=rm(JT / "runtime/SRVJ"))
    extract("A00", "rgb", "vjepa", "student")
    extract("SR3D", "hiera", JT / "runtime/SR3D/last.pt", "model")
    extract("ASLC002", "hiera", REPO / "method/jepa_baseline/runtime/ASLC002/train/last.model.pt", "model")
    extract("JV3D", "vjepa_fc", JT / "runtime/JV3D/last.pt", "encoder")
    extract("JH3D", "hiera", JT / "runtime/JH3D/last.pt", "model")
    extract("SRVJ", "vjepa_fc", JT / "runtime/SRVJ/last.pt", "encoder")
    base = ["SR3D", "SRVJ", "ASLC002", "JV3D", "JH3D", "A00"]
    evaluate("P0", base, refs=("SR3D", "SRVJ"))
    # ---- P1a: background recipe
    train_a("A10_AR", "AR", "generic", "uniform"); train_a("A10_AD", "AD", "generic", "uniform")
    evaluate("P1a", ["A10_AR", "A10_AD"] + base, refs=("SR3D", "SRVJ"))
    st = load_state(); R = st.get("decisions", {}).get("recipe")
    if R is None:
        mr, md = metric("A10_AR"), metric("A10_AD")
        if mr and md:  # rule: higher ASLC300 linear-probe macro; within 0.5 points -> higher LOSO R@1
            R = "AD" if (md[0] - mr[0] > 0.005 or (abs(md[0] - mr[0]) <= 0.005 and md[1] > mr[1])) else "AR"
        else:
            R = "AR"
        decide("recipe", R)
    # ---- P1b: factorial on the chosen recipe
    A = {"A10": f"A10_{R}"}
    for code, pol, w in (("A11", "generic", "balanced"), ("A12", "partial", "uniform"), ("A13", "partial", "balanced"), ("A10c", "generic_cov", "uniform")):
        A[code] = f"{code}_{R}"; train_a(A[code], R, pol, w)
    evaluate("P1", list(A.values()) + base, refs=("SR3D", "SRVJ"))
    # ---- P2: B
    Bn = {"B12": "latent", "B13": "anchors", "B11": "cce", "B10": "raw"}
    for k, obj in Bn.items():
        train_b(k, obj)
    evaluate("P2", list(Bn) + ["SR3D", "SRVJ", "A00"], refs=("SR3D", "SRVJ", "B12"))
    # ---- P3 gate: B13 teacher must match/beat B12 on handshape, location/relations and movement probes
    st = load_state(); gate = st.get("decisions", {}).get("b_gate")
    if gate is None:
        try:
            p13 = json.load(open(EV / "models/B13.json"))["loso"]["probe_R2"]; p12 = json.load(open(EV / "models/B12.json"))["loso"]["probe_R2"]
            gate = all(p13[k] >= p12[k] for k in ("handshape", "location_relations", "movement"))
        except Exception as e:
            say("gate error", e); gate = False
        decide("b_gate", bool(gate))
    Cn = []
    if gate:
        train_a(f"A13r_{R}", R, "partial", "balanced", extra=("--c-reserve",))
        train_a(f"C1_{R}", R, "partial", "balanced", extra=("--c-reserve", "--b-teacher", str(RT / "B13/final.pt")))
        Cn = [f"A13r_{R}", f"C1_{R}"]
        evaluate("P3", Cn + [A["A13"], "SR3D", "SRVJ"], refs=("SR3D", "SRVJ", f"A13r_{R}"))
    # ---- P4 (1 seed): best A config at 384 px, whole-articulator (A14) and jittered-region (A15) controls
    st = load_state(); best = st.get("decisions", {}).get("best_a")
    if best is None:
        cand = {k: metric(v) for k, v in A.items()}
        cand = {k: v for k, v in cand.items() if v}
        best = max(cand, key=lambda k: (cand[k][0], cand[k][1])) if cand else "A13"
        decide("best_a", best)
    pol = {"A10": "generic", "A11": "generic", "A12": "partial", "A13": "partial", "A10c": "generic_cov"}[best]
    wt = {"A10": "uniform", "A11": "balanced", "A12": "uniform", "A13": "balanced", "A10c": "uniform"}[best]
    train_a(f"{best}_{R}_384", R, pol, wt, size=384)
    train_a(f"A14_{R}", R, "whole", wt); train_a(f"A15_{R}", R, "jitter", wt)
    P4 = [f"{best}_{R}_384", f"A14_{R}", f"A15_{R}"]
    evaluate("P4", P4 + [A[best], "SR3D", "SRVJ"], refs=("SR3D", "SRVJ", A[best]))
    # ---- final table (all models, refs SR3D and SR-VJ) + report
    allm = base + list(A.values()) + list(Bn) + Cn + P4
    evaluate("FINAL", allm, refs=("SR3D", "SRVJ"))
    run("report", [PY, "oj/report.py", "--final", str(EV / "FINAL.json"), "--state", str(STATE), "--out", str(RT / "REPORT_1seed.md")],
        OJ, 0, exists(RT / "REPORT_1seed.md"))
    say("ALLDONE")


if __name__ == "__main__":
    main()
