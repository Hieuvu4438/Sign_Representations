#!/usr/bin/env python3
"""Background runner for the final ASM-JEPA job on ASL-Citizen (train -> validation -> official test).

Stages (each one idempotent; state in <job>/job_state.json, one log per stage in <job>/logs/):
  import_test_pose   CPU  official test poses into the npz cache (config is locked before anything runs)
  fit_stats_filter   CPU  train-only statistics for the observation-filtered variant
  pilot_nofilter     GPU  short SSL pilot, original observation rule      } A/B on val R@1 only;
  pilot_filter       GPU  short SSL pilot, audit-driven observation filter } the winner is the final variant
  select_variant     -    pick the variant, delete pilot checkpoints
  pretrain           GPU  60k-step SSL, frozen val retrieval every 2.5k steps, milestones every 10k, best_val
  stage_eval         GPU  frozen gloss-max retrieval (val + test) for every milestone (diagnostic curve) and for
                          best_val (OFFICIAL: selected on val only); best_val embeddings kept
  linear_probe       GPU  linear probe (+ few-shot k=1/2/4) on best_val embeddings, test evaluation
  finetune           GPU  ISLR fine-tuning from best_val, test evaluation of its best_val
  scratch            GPU  supervised scratch baseline (R0), same data pipeline, test evaluation
  report             -    results.csv / final_report.md (cli.report) + summary.md / summary.json

Interruptions: training stages resume from <run>/checkpoints/last.pt; finished sub-steps are skipped. A stage
that fails is retried (max --max-failures per stage, counted across restarts); a stop/kill/reboot is not a
failure. Re-running `start` (or the cron watchdog `ensure`) continues where the job stopped.
GPU etiquette: before each GPU stage the runner waits until at most --max-other-gpu-jobs other GPU processes of
this user are running and --min-free-gb GPU memory is free (server rule: at most 2 concurrent jobs).
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

METHOD = Path(__file__).resolve().parents[1]
PY = "/home/haipd/Sign_Representations/.venv-signrep/bin/python"
VARIANTS = {"filter": ("configs/pretrain/final.yaml", "configs/eval/finetune_final.yaml"),
            "nofilter": ("configs/pretrain/final_nofilter.yaml", "configs/eval/finetune_final_nofilter.yaml")}
STAGES = ["import_test_pose", "fit_stats_filter", "pilot_nofilter", "pilot_filter", "select_variant", "pretrain",
          "stage_eval", "linear_probe", "finetune", "scratch", "report"]
GPU_STAGES = {"pilot_nofilter", "pilot_filter", "pretrain", "stage_eval", "linear_probe", "finetune", "scratch"}
FEW_SHOT = (1, 2, 4)
# reference points already measured in this repo (ASL-Citizen test, gallery = train, signrep_style ties)
REFERENCES = [("SignRep public (RGB, YouTube-SL-25), local measurement", 70.42, 49.02, 79.29),
              ("SignRep paper Table 4 (avg)", 61.40, 37.47, 68.77),
              ("PhonoJEPA PLctx full160 (skeleton, ASL-Citizen train)", 39.62, 16.68, 36.90)]


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, p)


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass  # a line cut by a kill
    return out


class StageFailed(RuntimeError):
    pass


class Stopped(RuntimeError):
    pass


class Job:
    def __init__(self, args):
        self.args = args
        self.root = (METHOD / args.job_dir).resolve()
        self.logs = self.root / "logs"
        self.state_path = self.root / "job_state.json"
        self.child: subprocess.Popen | None = None
        self.stop_requested = False
        self.state = read_json(self.state_path) or {"created": now(), "status": "new", "stages": {},
                                                    "settings": vars(args)}

    # ------------------------------------------------------------------ bookkeeping
    def save(self):
        self.state["updated"] = now()
        write_json(self.state_path, self.state)

    def log(self, msg: str):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        with open(self.root / "job.log", "a") as f:
            f.write(line + "\n")

    def st(self, name: str) -> dict:
        return self.state["stages"].setdefault(name, {"status": "pending", "failures": 0, "runs": 0})

    def sub_done(self, stage: str, key: str) -> bool:
        return bool(self.st(stage).get("substeps", {}).get(key))

    def mark_sub(self, stage: str, key: str):
        self.st(stage).setdefault("substeps", {})[key] = now()
        self.save()

    @property
    def variant(self) -> str:
        return self.state.get("variant") or "filter"

    def pretrain_cfg(self) -> str:
        return VARIANTS[self.variant][0]

    def finetune_cfg(self) -> str:
        return VARIANTS[self.variant][1]

    def p(self, *parts) -> Path:
        return self.root.joinpath(*parts)

    def rel(self, path: Path) -> str:
        return os.path.relpath(path, METHOD)

    # ------------------------------------------------------------------ process control
    def run(self, log_name: str, cmd: list[str]):
        if self.stop_requested:
            raise Stopped()
        log = self.logs / f"{log_name}.log"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.log(f"  $ {' '.join(cmd)}  (log: {self.rel(log)})")
        with open(log, "a") as f:
            f.write(f"\n===== {now()} $ {' '.join(cmd)}\n")
            f.flush()
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            self.child = subprocess.Popen(cmd, cwd=METHOD, stdout=f, stderr=subprocess.STDOUT, env=env)
            rc = self.child.wait()
            self.child = None
        if self.stop_requested:
            raise Stopped()
        if rc != 0:
            raise StageFailed(f"{log_name}: exit code {rc} (see {self.rel(log)})")

    def handle_signal(self, signum, _frame):
        self.stop_requested = True
        self.log(f"signal {signum}: stopping child and runner (resumable)")
        child = self.child
        if child and child.poll() is None:
            # never wait() here: the main thread is inside child.wait() and owns Popen's wait lock
            child.terminate()
            t = threading.Timer(90, lambda: child.poll() is None and child.kill())
            t.daemon = True
            t.start()

    def gpu_others(self) -> tuple[int, float, list[str]]:
        """(# other GPU processes of this user, free GPU memory GB, descriptions)."""
        try:
            apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=30).stdout.split()
            free = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=30).stdout.split()
        except (OSError, subprocess.TimeoutExpired):
            return 0, 0.0, ["nvidia-smi unavailable"]
        uid, mine = os.getuid(), []
        for pid in apps:
            try:
                if os.stat(f"/proc/{pid}").st_uid == uid and int(pid) != os.getpid():
                    cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                    mine.append(f"{pid}: {cmd[:120]}")
            except (OSError, ValueError):
                continue
        return len(mine), (float(free[0]) / 1024 if free else 0.0), mine

    def wait_gpu(self, stage: str):
        if self.args.max_other_gpu_jobs < 0:
            return
        announced = 0.0
        while True:
            n, free, who = self.gpu_others()
            if n <= self.args.max_other_gpu_jobs and free >= self.args.min_free_gb:
                self.st(stage).pop("waiting", None)
                self.save()
                return
            if time.time() - announced > 1800:
                self.log(f"{stage}: waiting for a GPU slot (other GPU jobs of this user: {n} > "
                         f"{self.args.max_other_gpu_jobs} or free {free:.1f} GB < {self.args.min_free_gb}): {who}")
                announced = time.time()
            self.st(stage)["waiting"] = {"since": self.st(stage).get("waiting", {}).get("since", now()),
                                         "other_gpu_jobs": who, "free_gb": round(free, 1)}
            self.state["status"] = "waiting_gpu"
            self.save()
            for _ in range(60):
                if self.stop_requested:
                    raise Stopped()
                time.sleep(1)

    # ------------------------------------------------------------------ helpers
    def smoke(self, kind: str) -> list[str]:
        """Tiny-scale overrides for a plumbing smoke test of the runner (never real results)."""
        if not self.args.smoke:
            return []
        common = ["--set", "data.num_workers=2"]
        return {"pretrain": ["--limit-train-samples", "256", "--limit-eval-samples", "300",
                             "--set", "train.batch_size_per_gpu=16"] + common,
                "pretrain_main": ["--set", "train.max_steps=24", "--set", "train.warmup_steps=2",
                                  "--set", "loss.teacher_temperature_warmup_steps=2", "--set", "eval.frozen_every_steps=12",
                                  "--set", "train.milestone_every_steps=12", "--set", "train.checkpoint_every_steps=12"],
                "extract": ["--limit", "300", "--workers", "2"],
                "linear": ["--set", "train.epochs=2", "--set", "train.warmup_epochs=0"],
                "classifier": ["--limit-train-samples", "128", "--limit-eval-samples", "200", "--max-steps", "3",
                               "--set", "train.batch_size_per_gpu=16"] + common,
                "evaluate": ["--limit", "200", "--workers", "2"]}[kind]

    def pretrain_cmd(self, cfg: str, run_dir: Path, extra: list[str]) -> list[str]:
        cmd = [PY, "-m", "asm_jepa.cli.pretrain", "--config", cfg, "--run-dir", self.rel(run_dir),
               "--device", self.args.device] + self.smoke("pretrain")
        last = run_dir / "checkpoints" / "last.pt"
        if last.is_file():
            cmd += ["--resume", self.rel(last)]
        return cmd + extra

    def extract(self, cfg: str, ckpt: Path, split: str, out: Path, tag: str):
        if out.is_file():
            return
        tmp = out.with_name(out.stem + ".partial.npz")
        tmp.unlink(missing_ok=True)
        self.run(tag, [PY, "-m", "asm_jepa.cli.extract_embeddings", "--config", cfg, "--checkpoint", self.rel(ckpt),
                       "--split", split, "--output", self.rel(tmp), "--device", self.args.device] + self.smoke("extract"))
        os.replace(tmp, out)

    def retrieval(self, gallery: Path, queries: Path, out: Path, tag: str):
        if out.is_file():
            return
        self.run(tag, [PY, "-m", "asm_jepa.cli.retrieval", "--gallery", self.rel(gallery), "--queries",
                       self.rel(queries), "--protocol", "gloss_max", "--device", self.args.device,
                       "--output", self.rel(out)])

    # ------------------------------------------------------------------ stages
    def s_import_test_pose(self):
        self.run("import_test_pose", [PY, "-m", "asm_jepa.cli.import_pose_cache", "--source-dir",
                                      "data/raw/ASL_Citizen/keypoints", "--split", "test", "--resume",
                                      "--workers", str(self.args.cpu_workers)])

    def s_fit_stats_filter(self):
        self.run("fit_stats_filter", [PY, "-m", "asm_jepa.cli.fit_stats", "--config", VARIANTS["filter"][0],
                                      "--workers", str(self.args.cpu_workers)])

    def _pilot(self, variant: str):
        n = self.args.pilot_steps
        extra = ["--set", f"train.max_steps={n}", "--set", f"train.warmup_steps={n // 10}",
                 "--set", f"loss.teacher_temperature_warmup_steps={n // 10}",
                 "--set", f"eval.frozen_every_steps={n // 2}", "--set", "train.milestone_every_steps=0",
                 "--set", "train.keep_step_checkpoints=1", "--set", f"seed={self.args.seed}"]
        self.run(f"pilot_{variant}", self.pretrain_cmd(VARIANTS[variant][0], self.p("pilots", variant), extra))

    def s_pilot_nofilter(self):
        self._pilot("nofilter")

    def s_pilot_filter(self):
        self._pilot("filter")

    def s_select_variant(self):
        res = {}
        for v in VARIANTS:
            b = read_json(self.p("pilots", v, "reports", "best_val.json"))
            if b is None:
                raise StageFailed(f"pilot {v} has no best_val.json")
            res[v] = {"step": b["step"], "r1": b["r1"], "dcg": b["dcg"]}
        win = max(res, key=lambda v: (res[v]["r1"], res[v]["dcg"], v == "filter"))
        self.state["variant"] = win
        self.state["variant_selection"] = {"rule": "max val gloss-max R@1 (then DCG) after the pilot",
                                           "pilot_steps": self.args.pilot_steps, "pilots": res, "selected": win}
        self.save()
        for v in VARIANTS:
            shutil.rmtree(self.p("pilots", v, "checkpoints"), ignore_errors=True)
        self.log(f"variant selected: {win} {res}")

    def s_pretrain(self):
        self.run("pretrain", self.pretrain_cmd(self.pretrain_cfg(), self.p("pretrain"),
                                               ["--set", f"seed={self.args.seed}"] + self.smoke("pretrain_main")))
        rows = [r for r in read_jsonl(self.p("pretrain", "metrics.jsonl")) if "loss/total" in r]
        if not rows or not (self.p("pretrain", "checkpoints", "best_val.pt").is_file()):
            raise StageFailed("pretrain finished without metrics or best_val.pt")

    def s_stage_eval(self):
        cfg, ck_dir = self.pretrain_cfg(), self.p("pretrain", "checkpoints")
        cks = sorted(ck_dir.glob("milestone_*.pt")) + [ck_dir / "best_val.pt"]
        for ck in cks:
            name = ck.stem
            official = name == "best_val"
            emb = self.p("pretrain", "embeddings") if official else self.p("stage_eval", name, "embeddings")
            rep = self.p("pretrain", "reports") if official else self.p("stage_eval", name)
            if (rep / "frozen_test.json").is_file() and (rep / "frozen_val.json").is_file():
                continue
            emb.mkdir(parents=True, exist_ok=True)
            for split in ("train", "val", "test"):
                self.extract(cfg, ck, split, emb / f"{split}.npz", f"stage_eval_{name}")
            for split in ("val", "test"):
                self.retrieval(emb / "train.npz", emb / f"{split}.npz", rep / f"frozen_{split}.json",
                               f"stage_eval_{name}")
            if not official:
                shutil.rmtree(emb, ignore_errors=True)  # ~0.45 GB per checkpoint; reports are kept
            self.log(f"stage_eval {name}: " + _fmt_retr(read_json(rep / "frozen_test.json"), "test"))

    def s_linear_probe(self):
        emb = self.p("pretrain", "embeddings")
        for k in (None,) + FEW_SHOT:
            name = "linear" if k is None else f"linear_k{k}"
            run_dir = self.p(name)
            if (run_dir / "reports" / "test.json").is_file():
                continue
            shutil.rmtree(run_dir, ignore_errors=True)  # probes are minutes long: restart cleanly
            cmd = [PY, "-m", "asm_jepa.cli.train_classifier", "--config", "configs/eval/linear.yaml",
                   "--train-embeddings", self.rel(emb / "train.npz"), "--val-embeddings", self.rel(emb / "val.npz"),
                   "--run-dir", self.rel(run_dir), "--set", f"seed={self.args.seed}", "--device", self.args.device]
            cmd += self.smoke("linear")
            if k:
                cmd += ["--few-shot-k", str(k), "--few-shot-seed", str(self.args.seed)]
            self.run(name, cmd)
            self.run(name, [PY, "-m", "asm_jepa.cli.evaluate_classifier", "--checkpoint",
                            self.rel(run_dir / "checkpoints" / "best_val.pt"), "--embeddings",
                            self.rel(emb / "test.npz"), "--split", "test", "--output",
                            self.rel(run_dir / "reports" / "test.json"), "--device", self.args.device])

    def _classifier(self, name: str, init: str):
        run_dir = self.p(name)
        if not self.sub_done(name, "train"):
            cmd = [PY, "-m", "asm_jepa.cli.train_classifier", "--config", self.finetune_cfg(), "--init", init,
                   "--run-dir", self.rel(run_dir), "--set", f"seed={self.args.seed}",
                   "--device", self.args.device] + self.smoke("classifier")
            last = run_dir / "checkpoints" / "last.pt"
            if last.is_file():
                cmd += ["--resume", self.rel(last)]
            self.run(name, cmd)
            self.mark_sub(name, "train")
        if not (run_dir / "reports" / "test.json").is_file():
            self.run(name, [PY, "-m", "asm_jepa.cli.evaluate_classifier", "--checkpoint",
                            self.rel(run_dir / "checkpoints" / "best_val.pt"), "--split", "test", "--output",
                            self.rel(run_dir / "reports" / "test.json"), "--device", self.args.device]
                     + self.smoke("evaluate"))

    def s_finetune(self):
        self._classifier("finetune", self.rel(self.p("pretrain", "checkpoints", "best_val.pt")))

    def s_scratch(self):
        self._classifier("scratch", "scratch")

    def s_report(self):
        runs = [self.p(n) for n in ("pretrain", "linear", "finetune", "scratch") if self.p(n).is_dir()]
        self.run("report", [PY, "-m", "asm_jepa.cli.report", "--runs", *[self.rel(r) for r in runs],
                            "--output", self.rel(self.p("final_report.md"))])
        write_summary(self)

    # ------------------------------------------------------------------ main loop
    def main(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock = open(self.root / "runner.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another runner holds the lock; exiting")
            return 0
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        self.state.update(status="running", runner_pid=os.getpid(), runner_started=now())
        self.save()
        self.log(f"runner start pid {os.getpid()} job {self.root}")
        skip = set(self.args.skip or [])
        if self.args.pilot_steps <= 0:
            skip |= {"pilot_nofilter", "pilot_filter", "select_variant"}
        try:
            for name in STAGES:
                s = self.st(name)
                if s["status"] in ("done", "skipped"):
                    continue
                if name in skip or (name == "scratch" and self.args.no_scratch):
                    s["status"] = "skipped"
                    self.save()
                    continue
                while True:
                    if name in GPU_STAGES:
                        self.wait_gpu(name)
                    s.update(status="running", started=s.get("started") or now(), runs=s["runs"] + 1)
                    self.state.update(status="running", current_stage=name)
                    self.save()
                    t0 = time.time()
                    self.log(f"== stage {name} (run {s['runs']}, failures so far {s['failures']})")
                    try:
                        getattr(self, f"s_{name}")()
                        s["seconds"] = s.get("seconds", 0) + time.time() - t0
                        s.update(status="done", ended=now())
                        self.save()
                        self.log(f"== stage {name} done ({(time.time() - t0) / 60:.1f} min)")
                        write_summary(self)
                        break
                    except StageFailed as e:
                        s["seconds"] = s.get("seconds", 0) + time.time() - t0
                        s["failures"] += 1
                        s["last_error"] = str(e)
                        self.log(f"!! stage {name} failed ({s['failures']}/{self.args.max_failures}): {e}")
                        if s["failures"] >= self.args.max_failures:
                            s["status"] = "failed"
                            self.state["status"] = "failed"
                            self.save()
                            self.log("job FAILED; fix the cause, then `scripts/final_job.sh reset` and `start`")
                            return 1
                        s["status"] = "retry_wait"
                        self.save()
                        for _ in range(self.args.retry_wait):
                            if self.stop_requested:
                                raise Stopped()
                            time.sleep(1)
        except Stopped:
            cur = self.state.get("current_stage")
            if cur:
                self.st(cur)["status"] = "interrupted"
            self.state["status"] = "stopped"
            self.save()
            self.log("runner stopped; `scripts/final_job.sh start` resumes")
            return 0
        self.state.update(status="done", current_stage=None, finished=now())
        self.save()
        self.log("job DONE: " + self.rel(self.p("summary.md")))
        return 0


# ---------------------------------------------------------------------- reporting
def _pct(x):
    return "–" if x is None else f"{100 * x:.2f}"


def _fmt_retr(r, split):
    if not r:
        return f"{split}: n/a"
    m, s = r["metrics"], r["signrep_style"]
    return (f"{split} R@1 {_pct(m['r1'])} R@5 {_pct(m['r5'])} R@10 {_pct(m['r10'])} MRR {_pct(m['mrr'])} "
            f"DCG {_pct(m['dcg'])} | signrep_style DCG {_pct(s['DCG'])} Rec@1 {_pct(s['Rec@1'])} Rec@5 {_pct(s['Rec@5'])}")


def _du(p: Path) -> float:
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


def write_summary(job: Job) -> None:
    root, st = job.root, job.state
    out = {"updated": now(), "status": st.get("status"), "variant": st.get("variant"),
           "variant_selection": st.get("variant_selection"), "stages": st.get("stages")}
    L = [f"# ASM-JEPA final job — ASL-Citizen (seed {job.args.seed})", "",
         f"Cập nhật {now()} · trạng thái **{st.get('status')}** · variant **{st.get('variant') or 'chưa chọn'}** "
         f"(config `{job.pretrain_cfg()}`)", ""]
    sel = st.get("variant_selection")
    if sel:
        L += ["## A/B pilot (chọn bằng val, không dùng test)", "", "| variant | step | val R@1 % | val DCG % |",
              "|---|---:|---:|---:|"]
        L += [f"| {v}{' ✔' if v == sel['selected'] else ''} | {r['step']} | {_pct(r['r1'])} | {_pct(r['dcg'])} |"
              for v, r in sel["pilots"].items()]
        L.append("")
    m = read_jsonl(root / "pretrain" / "metrics.jsonl")
    train_rows = [r for r in m if "loss/total" in r]
    val_rows = [r for r in m if r.get("kind") == "frozen_val"]
    if train_rows:
        last = train_rows[-1]
        sps = [r["samples_per_s"] for r in train_rows if r.get("samples_per_s")]
        out["train"] = {"step": last["step"], "loss_total": last.get("loss/total"),
                        "median_samples_per_s": sorted(sps)[len(sps) // 2] if sps else None,
                        "peak_vram_gb": max(r.get("peak_vram_gb", 0) for r in train_rows)}
        L += ["## Pretrain", "", f"step {last['step']} · loss {last.get('loss/total', float('nan')):.4f} · "
              f"median {out['train']['median_samples_per_s'] or 0:.0f} samples/s · "
              f"peak VRAM {out['train']['peak_vram_gb']:.2f} GB", ""]
    if val_rows:
        out["val_curve"] = [{"step": r["step"], "r1": r["val/frozen_gloss_max/r1"],
                             "r5": r["val/frozen_gloss_max/r5"], "dcg": r["val/frozen_gloss_max/dcg"]}
                            for r in val_rows]
        L += ["### Validation trong lúc train (frozen gloss-max, gallery train, query val)", "",
              "| step | R@1 % | R@5 % | R@10 % | MRR % | DCG % | balanced256 R@1 % |", "|---:|---:|---:|---:|---:|---:|---:|"]
        L += [f"| {r['step']} | {_pct(r['val/frozen_gloss_max/r1'])} | {_pct(r['val/frozen_gloss_max/r5'])} | "
              f"{_pct(r['val/frozen_gloss_max/r10'])} | {_pct(r['val/frozen_gloss_max/mrr'])} | "
              f"{_pct(r['val/frozen_gloss_max/dcg'])} | {_pct(r.get('val/frozen_gloss_max_balanced256/r1'))} |"
              for r in val_rows]
        L.append("")
    best = read_json(root / "pretrain" / "reports" / "best_val.json")
    if best:
        L += [f"best_val: step {best['step']} (val R@1 {_pct(best['r1'])} %, DCG {_pct(best['dcg'])} %)", ""]
    stage = []
    for d in sorted((root / "stage_eval").glob("milestone_*")) + [root / "pretrain" / "reports"]:
        v, t = read_json(d / "frozen_val.json"), read_json(d / "frozen_test.json")
        if v or t:
            stage.append((d.name if d.name != "reports" else "best_val (OFFICIAL)", v, t))
    if stage:
        out["stage_eval"] = {n: {"val": v and v["metrics"], "test": t and t["metrics"],
                                 "test_signrep_style": t and t["signrep_style"],
                                 "step": (v or t).get("checkpoint_step")} for n, v, t in stage}
        L += ["## Sign Video Retrieval theo giai đoạn (frozen, gloss-max, gallery = official train)", "",
              "Milestone test chỉ là đường cong chẩn đoán; con số chính thức là **best_val** (chọn bằng val).", "",
              "| checkpoint | step | val R@1 | val R@5 | val DCG | test R@1 | test R@5 | test R@10 | test MRR | test DCG "
              "| test DCG (SignRep) | test Rec@1 (SignRep) | test Rec@5 (SignRep) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for n, v, t in stage:
            vm, tm, ts = (v or {}).get("metrics", {}), (t or {}).get("metrics", {}), (t or {}).get("signrep_style", {})
            L.append(f"| {n} | {(v or t).get('checkpoint_step')} | {_pct(vm.get('r1'))} | {_pct(vm.get('r5'))} | "
                     f"{_pct(vm.get('dcg'))} | {_pct(tm.get('r1'))} | {_pct(tm.get('r5'))} | {_pct(tm.get('r10'))} | "
                     f"{_pct(tm.get('mrr'))} | {_pct(tm.get('dcg'))} | {_pct(ts.get('DCG'))} | {_pct(ts.get('Rec@1'))} | "
                     f"{_pct(ts.get('Rec@5'))} |")
        L.append("")
        off = read_json(root / "pretrain" / "reports" / "frozen_test.json")
        if off:
            ci = off.get("bootstrap_ci_signers") or {}
            L += ["### Kết quả chính thức (test, best_val) và mốc tham chiếu trong repo", "",
                  "| method | DCG | Rec@1 % | Rec@5 % |", "|---|---:|---:|---:|",
                  f"| **ASM-JEPA final ({job.variant})** | {_pct(off['signrep_style']['DCG'])} | "
                  f"{_pct(off['signrep_style']['Rec@1'])} | {_pct(off['signrep_style']['Rec@5'])} |"]
            L += [f"| {n} | {a:.2f} | {b:.2f} | {c:.2f} |" for n, a, b, c in REFERENCES]
            if ci.get("r1"):
                L += ["", f"95 % CI theo signer (R@1, tie class-id): {_pct(ci['r1']['lo'])}–{_pct(ci['r1']['hi'])} %"]
            L.append("")
    cls = []
    for n in ("linear",) + tuple(f"linear_k{k}" for k in FEW_SHOT) + ("finetune", "scratch"):
        t = read_json(root / n / "reports" / "test.json")
        v = read_json(root / n / "reports" / "best_val.json") or (read_json(root / n / "reports" / "val.json") or {}).get("best_val")
        if t:
            cls.append((n, v, t["metrics"]))
    if cls:
        out["classifiers"] = {n: {"test": m, "val_best": v} for n, v, m in cls}
        L += ["## Recognition (classifier ranking, test)", "", "| run | val top-1 (best) | test top-1 | test top-5 | "
              "class-macro top-1 | DCG | MRR |", "|---|---:|---:|---:|---:|---:|---:|"]
        L += [f"| {n} | {_pct((v or {}).get('top1_instance'))} | {m['top1_instance_pct']:.2f} | "
              f"{m['top5_instance_pct']:.2f} | {m['top1_class_macro_pct']:.2f} | {m['dcg_pct']:.2f} | "
              f"{m['mrr_pct']:.2f} |" for n, v, m in cls]
        L.append("")
    L += ["## Stage", "", "| stage | trạng thái | thời gian (phút) | lần chạy | lỗi |", "|---|---|---:|---:|---:|"]
    for n in STAGES:
        s = st.get("stages", {}).get(n, {})
        L.append(f"| {n} | {s.get('status', 'pending')} | {s.get('seconds', 0) / 60:.1f} | {s.get('runs', 0)} | "
                 f"{s.get('failures', 0)} |")
    disk = {"job_dir_gb": round(_du(root), 2)}
    out["disk"] = disk
    L += ["", f"Dung lượng thư mục job: {disk['job_dir_gb']} GB", ""]
    write_json(root / "summary.json", out)
    (root / "summary.md").write_text("\n".join(L) + "\n")


def status(job: Job) -> int:
    st = job.state
    print(f"job {job.root}\nstatus: {st.get('status')}  current: {st.get('current_stage')}  variant: "
          f"{st.get('variant')}  updated: {st.get('updated')}")
    pid = st.get("runner_pid")
    alive = bool(pid) and Path(f"/proc/{pid}").exists() and "final_job" in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
    print(f"runner pid {pid} alive: {alive}")
    for n in STAGES:
        s = st.get("stages", {}).get(n, {})
        extra = f" waiting: {s['waiting']}" if s.get("waiting") else ""
        err = f" last_error: {s['last_error']}" if s.get("last_error") and s.get("status") != "done" else ""
        print(f"  {n:18s} {s.get('status', 'pending'):12s} {s.get('seconds', 0) / 60:7.1f} min{extra}{err}")
    cur = st.get("current_stage")
    run_dir = {"pretrain": "pretrain", "pilot_filter": "pilots/filter", "pilot_nofilter": "pilots/nofilter",
               "finetune": "finetune", "scratch": "scratch"}.get(cur)
    if run_dir:
        rows = read_jsonl(job.p(run_dir, "metrics.jsonl"))
        tr = [r for r in rows if "loss/total" in r or ("loss" in r and "epoch" not in r)]
        va = [r for r in rows if r.get("kind") == "frozen_val" or "val/cls/top1_instance" in r]
        if tr:
            r = tr[-1]
            msg = f"  last train row: step {r['step']} loss {r.get('loss/total', r.get('loss')):.4f}"
            if "samples_per_s" in r and cur in ("pretrain", "pilot_filter", "pilot_nofilter"):
                total = job.args.pilot_steps if cur.startswith("pilot") else 60000
                eta_h = (total - r["step"]) * 128 / max(r["samples_per_s"], 1) / 3600
                msg += f" {r['samples_per_s']:.0f} samples/s, ETA ~{eta_h:.1f} h (excl. validation)"
            print(msg)
        if va:
            r = va[-1]
            k = "val/frozen_gloss_max/r1" if "val/frozen_gloss_max/r1" in r else "val/cls/top1_instance"
            print(f"  last val: step {r['step']} {k} {100 * r[k]:.2f} %")
    print(f"summary: {job.rel(job.p('summary.md'))}   logs: {job.rel(job.logs)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "status", "summary", "reset"))
    ap.add_argument("--job-dir", default="runs/final_s0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pilot-steps", type=int, default=5000, help="0 disables the A/B pilot (variant = filter)")
    ap.add_argument("--max-other-gpu-jobs", type=int, default=1,
                    help="wait until at most this many other GPU processes of this user run (-1: never wait)")
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    ap.add_argument("--max-failures", type=int, default=3)
    ap.add_argument("--retry-wait", type=int, default=120, help="seconds between a failure and its retry")
    ap.add_argument("--cpu-workers", type=int, default=12)
    ap.add_argument("--no-scratch", action="store_true", help="skip the supervised scratch baseline (R0)")
    ap.add_argument("--skip", nargs="*", choices=STAGES)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="tiny-scale plumbing test (numbers are meaningless)")
    args = ap.parse_args()
    os.chdir(METHOD)
    job = Job(args)
    if args.command == "status":
        return status(job)
    if args.command == "summary":
        write_summary(job)
        print((job.root / "summary.md").read_text())
        return 0
    if args.command == "reset":
        for s in job.state.get("stages", {}).values():
            if s.get("status") in ("failed", "retry_wait", "running", "interrupted"):
                s.update(status="pending", failures=0)
        job.state["status"] = "reset"
        job.save()
        print("failure counters reset")
        return 0
    return job.main()


if __name__ == "__main__":
    sys.exit(main())
