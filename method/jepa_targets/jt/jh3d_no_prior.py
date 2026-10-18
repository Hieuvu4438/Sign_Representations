"""Run and record a matched JH3D prior-loss ablation; no test-set access or automatic retries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RUN = ROOT / "runtime/ablations/jh3d_no_prior_20261013"
NAME = "JH3D_NOPRIOR_20261013"
RAW = NAME + "_raw"
BASE_RAW = "JH3D_RAW_20261013"
RECHECK = "JH3D_RECHECK_20261013"


def dump(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_stage(name, args, timeout_s=14400):
    command = [sys.executable, *args]
    started = time.time()
    with (RUN / f"{name}.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            while True:
                code = proc.poll()
                state = {"stage": name, "command": command, "pid": proc.pid,
                         "runner_pid": os.getpid(), "elapsed_s": time.time() - started,
                         "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                         "status": "running" if code is None else "completed" if code == 0 else "failed",
                         "exit_code": code}
                if name == "train":
                    progress = ROOT / "runtime" / NAME / "train_log.jsonl"
                    if progress.exists():
                        lines = progress.read_text().splitlines()
                        if lines:
                            state["training"] = json.loads(lines[-1])
                dump(RUN / "status.json", state)
                if code is not None:
                    if code:
                        raise RuntimeError(f"{name} failed (exit {code}); see {RUN / (name + '.log')}")
                    return state
                if time.time() - started > timeout_s:
                    print(f"Hard timeout for {name}; terminating its process group", flush=True)
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    raise TimeoutError(f"{name} exceeded {timeout_s} seconds")
                time.sleep(30)
        finally:
            if proc.poll() is None:
                print(f"Runner exiting while {name} is still active (PID {proc.pid})", flush=True)


def check_smoke():
    import torch
    initial = torch.load(REPO / "method/jepa_baseline/runtime/ASLC002/initial.model.pt",
                         map_location="cpu", weights_only=True)["model"]
    ck = torch.load(RUN / "smoke/last.pt", map_location="cpu", weights_only=True)
    frozen = [k for k in initial if k.startswith(("fc.", "cls_head.", "active_head."))]
    assert all(torch.equal(initial[k], ck["model"][k]) for k in frozen), "Prior-only heads received updates"
    assert any(not torch.equal(initial[k], ck["model"][k]) for k in initial if k.startswith("backbone.")), "Backbone did not learn"
    assert ck["config"]["prior_w"] == 0 and ck["step"] == 6
    logs = [json.loads(line) for line in (RUN / "smoke/train_log.jsonl").read_text().splitlines()]
    assert all(x["reconstruction"] == 0 and x["loss"] == x["latent"] for x in logs)
    dump(RUN / "smoke_check.json", {"passed": True, "steps": ck["step"],
                                   "prior_heads_and_fc_unchanged": True, "backbone_updated": True,
                                   "total_loss_equals_latent_loss": True})


def summarize():
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from jt.evaluate import load_feats, loso, pool, whiten
    from p4.retrieval import paired_cluster_bootstrap, per_query
    results = json.loads((RUN / "evaluation.json").read_text())
    comparison = {}
    for base, new, key in (("JH3D", NAME, "pipeline_embedding"), (BASE_RAW, RAW, "backbone_before_fc")):
        c0, w0, o0, s0 = load_feats(base)
        c1, w1, o1, s1 = load_feats(new)
        assert c0 == c1 and np.array_equal(o0, o1) and np.array_equal(s0, s1), "Evaluation examples/windows differ"
        labels = {g: i for i, g in enumerate(sorted({c["gloss"] for c in c0}))}
        y = np.array([labels[c["gloss"]] for c in c0])
        sg = np.array([c["signer"] for c in c0])
        r0 = loso(pool(whiten(w0), o0, "mean"), y, sg, len(labels))
        r1 = loso(pool(whiten(w1), o1, "mean"), y, sg, len(labels))
        bs = paired_cluster_bootstrap(per_query(r0)["R@1"], per_query(r1)["R@1"], sg, n_boot=10000)
        comparison[key] = {"baseline": base, "ablation": new,
                           "R@1_baseline": results[base]["white512"]["R@1"],
                           "R@1_no_prior": results[new]["white512"]["R@1"],
                           "no_prior_minus_baseline": bs,
                           "pooling": {mode: {"baseline": results[base][mode]["R@1"],
                                              "no_prior": results[new][mode]["R@1"]}
                                       for mode in ("mean", "bins2", "white512", "white512_bins2")}}
    dump(RUN / "comparison.json", comparison)
    rows = []
    for key, c in comparison.items():
        bs = c["no_prior_minus_baseline"]
        rows.append(f"| {key} | {100*c['R@1_baseline']:.2f} | {100*c['R@1_no_prior']:.2f} | "
                    f"{100*bs['diff']:+.2f} | [{100*bs['ci95'][0]:+.2f}, {100*bs['ci95'][1]:+.2f}] |")
    (RUN / "RESULTS.md").write_text(
        "## Material Passport\n\n- Origin Skill: academic-research-suite / experiment-agent\n"
        "- Origin Mode: run\n- Origin Date: 2026-10-13\n"
        "- Verification Status: ANALYZED (single training seed)\n- Version Label: jh3d-no-prior-v1\n\n"
        "## Experiment Result\n\nJH3D ablation: remove all nine geometric priors plus activity supervision; "
        "retain latent JEPA prediction. Both use Kinetics-MAE initialization, the same 3,000-video "
        "ASLC002 corpus, seed 20261008, six workers, 16 videos / 32 views per batch, 4,000 steps, "
        "80% masking, predictor, optimizer and EMA 0.99925. Existing JH3D checkpoint is the reference.\n\n"
        "Primary metric: ASL Citizen validation leave-one-signer-out R@1, white512 + mean pooling. "
        "Whitening is fit on validation windows as in the original exploratory EX16 protocol; "
        "no test-set access. No seed replication or checkpoint selection.\n\n"
        "| Readout | JH3D R@1 (%) | No prior R@1 (%) | Delta (pp) | Signer bootstrap 95% CI (pp) |\n"
        "|---|---:|---:|---:|---|\n" + "\n".join(rows) + "\n\n"
        "The pipeline readout includes fc: its weights are learned by prior loss in JH3D and remain "
        "at initialization in the ablation. The backbone readout uses mean tokens before fc for "
        "both arms, to compare encoder representations independently of the projection head. "
        "Confidence intervals resample validation signers, not training seeds; a single seed cannot "
        "establish robustness across training randomness. Full pooling, probe and geometry results "
        "are in evaluation.json; per-signer deltas are in comparison.json.\n")
    return comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true")
    a = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    if a.smoke_only:
        run_stage("smoke", ["jt/ex16.py", "--arm", "jh3d", "--prior-w", "0", "--steps", "6",
                            "--videos", "2", "--out", str(RUN / "smoke")], timeout_s=600)
        check_smoke()
        print("SMOKE PASSED", flush=True)
        return
    assert json.loads((RUN / "smoke_check.json").read_text())["passed"]
    baseline_config = json.loads((ROOT / "runtime/JH3D/config.json").read_text())
    expected = {"arm": "jh3d", "steps": 4000, "videos": 16, "jepa_w": 1.0,
                "ema": 0.99925, "workers": 6, "seed": 20261008}
    assert all(baseline_config[k] == v for k, v in expected.items())
    import torch
    baseline = torch.load(ROOT / "runtime/JH3D/last.pt", map_location="cpu", weights_only=True)
    assert baseline["step"] == 4000
    del baseline
    files = [ROOT / "jt/ex16.py", ROOT / "jt/extract.py", ROOT / "jt/evaluate.py",
             ROOT / "jt/jh3d_no_prior.py", ROOT / "runtime/JH3D/last.pt",
             REPO / "method/jepa_baseline/runtime/ASLC002/initial.model.pt",
             REPO / "method/jepa_baseline/runtime/ASLC002/corpus/pretrain.jsonl"]
    protocol = {"id": "jh3d-no-prior-20261013", "baseline_config": baseline_config,
                "intervention": "prior_w=0; no prior/activity gradient", "primary": "val LOSO white512 mean R@1",
                "secondary_control": "mean backbone tokens before fc, both arms",
                "test_access": False, "seeds": [20261008], "per_stage_timeout_s": 14400,
                "input_sha256": {str(p): digest(p) for p in files},
                "torch_version": torch.__version__, "python": sys.version}
    dump(RUN / "protocol.json", protocol)
    run_stage("train", ["jt/ex16.py", "--arm", "jh3d", "--prior-w", "0",
                        "--out", f"runtime/{NAME}", "--steps", "4000", "--videos", "16",
                        "--seed", "20261008", "--workers", "6", "--jepa-w", "1", "--ema", "0.99925"])
    run_stage("extract_no_prior", ["jt/extract.py", "--kind", "hiera", "--ckpt", f"runtime/{NAME}/last.pt",
                                   "--name", NAME, "--latent-name", RAW, "--weights", "model", "--workers", "6"])
    run_stage("extract_baseline", ["jt/extract.py", "--kind", "hiera", "--ckpt", "runtime/JH3D/last.pt",
                                   "--name", RECHECK, "--latent-name", BASE_RAW, "--weights", "model", "--workers", "6"])
    import numpy as np
    old = np.load(ROOT / "runtime/feats/JH3D/windows.npy")
    check = np.load(ROOT / "runtime/feats" / RECHECK / "windows.npy")
    assert old.shape == check.shape and np.allclose(old, check, atol=0.005, rtol=0.005), "Baseline extraction changed"
    dump(RUN / "baseline_recheck.json", {"passed": True, "exact": bool(np.array_equal(old, check)),
                                         "max_abs_error": float(np.abs(old.astype(float)-check).max())})
    run_stage("evaluate", ["jt/evaluate.py", "--feats", f"JH3D,{NAME},{BASE_RAW},{RAW}",
                           "--ref", "JH3D", "--out", str(RUN / "evaluation.json")])
    summary = summarize()
    dump(RUN / "status.json", {"status": "completed", "runner_pid": os.getpid(),
                               "result": str(RUN / "RESULTS.md"), "comparison": summary})
    print("COMPLETED", RUN / "RESULTS.md", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        state = json.loads((RUN / "status.json").read_text()) if (RUN / "status.json").exists() else {}
        dump(RUN / "status.json", {**state, "status": "failed", "error": str(exc)})
        raise
