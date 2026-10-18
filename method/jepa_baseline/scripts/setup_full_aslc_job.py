"""Prepare frozen manifests/runtime and install an SSH-independent user service."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from signrep_baseline.cache import sha256
from signrep_baseline.cli import load_config
from signrep_baseline.job_support import atomic_json, read_json, progress

SERVICE = "signrep-full-aslc-kinetics.service"


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def build_manifests(job, dataset, seed):
    source = {}
    csv_pins = {}
    for split in ("train", "val", "test"):
        csv_path = dataset / "splits" / (split + ".csv")
        csv_pins[str(csv_path)] = sha256(csv_path)
        with csv_path.open() as file:
            source[split] = list(csv.DictReader(file))
    classes = sorted({row["Gloss"] for rows in source.values() for row in rows})
    labels = {name: i for i, name in enumerate(classes)}
    rows = {}
    counts = {}
    for split, original in source.items():
        rows[split] = []
        sizes = 0
        for row in original:
            path = dataset / "videos" / row["Video file"]
            if not path.is_file():
                raise ValueError(f"Official dataset video missing: {path}")
            sizes += path.stat().st_size
            ident = path.stem
            rows[split].append({"video_id": "asl_citizen:" + ident,
                "source_id": ident, "dataset": "asl_citizen", "source_split": split,
                "split": split, "video": str(path), "label": labels[row["Gloss"]],
                "gloss": row["Gloss"], "signer_id": hashlib.sha256(
                ("asl_citizen:" + row["Participant ID"]).encode()).hexdigest()[:16]})
        if len({r["video_id"] for r in rows[split]}) != len(original):
            raise ValueError("Official split has duplicate video identities")
        counts[split] = {"videos": len(original), "classes": len({r["label"] for r in rows[split]}),
                         "signers": len({r["signer_id"] for r in rows[split]}), "bytes": sizes}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for key in ("video_id", "signer_id"):
            if {r[key] for r in rows[left]} & {r[key] for r in rows[right]}:
                raise ValueError("Official source/signer splits overlap")
    for split in rows:
        write_rows(job / "manifests" / f"official_{split}.jsonl", rows[split])
    write_rows(job / "manifests/official_held_out.jsonl", rows["val"] + rows["test"])
    write_rows(job / "source.jsonl", [{**r, "split": "pretrain"} for r in rows["train"]])
    rng = random.Random(seed)
    gallery, queries = [], []
    for label in range(len(classes)):
        train = [r for r in rows["train"] if r["label"] == label]
        val = [r for r in rows["val"] if r["label"] == label]
        rng.shuffle(train)
        rng.shuffle(val)
        selected, used = [], set()
        for row in train:
            if row["signer_id"] not in used:
                selected.append(row)
                used.add(row["signer_id"])
            if len(selected) == 2:
                break
        if len(selected) < 2:
            selected = train[:2]
        if len(selected) != 2 or not val:
            raise ValueError("Every class needs 2 training and 1 validation proxy videos")
        gallery.extend(selected)
        queries.append(val[0])
    write_rows(job / "manifests/gallery_proxy.jsonl", gallery)
    write_rows(job / "manifests/val_proxy.jsonl", queries)
    atomic_json(job / "inventory.json", {"official_splits": counts, "classes": classes,
                "csv_pins": csv_pins, "proxy_seed": seed, "proxy_gallery": len(gallery),
                "proxy_queries": len(queries), "test_used_for_selection": False,
                "full_train_membership": len(rows["train"]), "invalid_pose_policy": "stop_and_report"})
    return rows, csv_pins


def snapshot(job):
    code = job / "code"
    code.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "src", code / "src", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "vendor/skeleton_3d_uplift", code / "vendor/skeleton_3d_uplift",
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (code / "scripts").mkdir(exist_ok=True)
    for name in ("pose_worker.py", "full_aslc_pose.py", "full_aslc_job.py"):
        shutil.copy2(ROOT / "scripts" / name, code / "scripts" / name)
    (code / "evidence").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "evidence/teacher_weights.json", code / "evidence/teacher_weights.json")
    manifest = {str(p.relative_to(code)): sha256(p) for p in sorted(code.rglob("*")) if p.is_file()}
    atomic_json(job / "code_manifest.json", manifest)


def reuse_raw(job, train, teacher_weights):
    import numpy as np
    pilot = ROOT / "runtime/ASLC002"
    if not (pilot / "source.jsonl").exists():
        return 0
    existing = [json.loads(s) for s in (pilot / "source.jsonl").read_text().splitlines() if s]
    lookup = {r["video_id"]: (i, r) for i, r in enumerate(existing)}
    output = job / "corpus/raw"
    receipts = job / "pose_receipts"
    output.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    reused = 0
    for index, row in enumerate(train):
        match = lookup.get(row["video_id"])
        if not match:
            continue
        old = pilot / "corpus/raw" / f"{match[0]:08d}.npz"
        if not old.exists():
            continue
        with np.load(old, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))
            flags = data["body_detected"].copy()
            if (meta["schema"] != "signrep_teacher_v1" or meta["body_root_mode"] != "canonical"
                    or meta["weights"] != teacher_weights or meta["video_sha256"] != sha256(row["video"])):
                continue
            if not np.array_equal(data["frame_indices"], np.arange(meta["decoded_frames"])):
                continue
        target = output / f"{index:08d}.npz"
        if not target.exists():
            try:
                os.link(old, target)
            except OSError:
                shutil.copy2(old, target)
        valid = int((np.convolve(flags.astype(int), np.ones(16, int), mode="valid") == 16).sum()) if len(flags) >= 16 else 0
        atomic_json(receipts / f"{index:08d}.json", {"index": index, "video_id": row["video_id"],
                    "decoded_frames": len(flags), "valid_body_windows": valid, "raw": str(target),
                    "raw_sha256": sha256(target), "status": "PASS" if valid else "NO_VALID_WINDOW",
                    "seconds": 0, "reused_from": str(old)})
        reused += 1
    return reused


def unit_text(job, options):
    return f'''[Unit]
Description=SignRep full ASL-Citizen Kinetics initialization; 50k review and resumable training
StartLimitIntervalSec=1800
StartLimitBurst=3

[Service]
Type=simple
WorkingDirectory={WORKSPACE}
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=OMP_NUM_THREADS=2
Environment=MKL_NUM_THREADS=2
Environment=OPENBLAS_NUM_THREADS=1
Environment=TF_CPP_MIN_LOG_LEVEL=2
ExecStart={options["rgb_python"]} -u {job}/code/scripts/full_aslc_job.py run --job {job}
Restart=on-failure
RestartPreventExitStatus=78
RestartSec=45
TimeoutStopSec=240
KillMode=mixed
Nice=5
CPUQuota=1000%
MemoryHigh=22G
MemoryMax=26G
StandardOutput=append:{job}/console.log
StandardError=inherit

[Install]
WantedBy=default.target
'''


def setup(args):
    job = args.job.resolve()
    if (job / "config.json").exists():
        raise ValueError("Job already initialized; use systemctl start/restart to resume, not setup")
    if shutil.disk_usage(ROOT).free < 35 * 2**30:
        raise ValueError("Prepare at least 35 GiB free before launching the full job")
    job.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    if config["initialization"] != "kinetics_mae":
        raise ValueError("Full job must initialize only from Kinetics MAE")
    if (config["total_steps"], config["warmup_steps"], config["batch_size"]) != (100000, 10000, 20):
        raise ValueError("Full job recipe differs from the approved 100k/10k/20 configuration")
    config["manifest"] = str(job / "corpus/pretrain.jsonl")
    config["output"] = str(job / "train")
    config["validation_gallery"] = str(job / "manifests/gallery_proxy.jsonl")
    config["validation_queries"] = str(job / "manifests/val_proxy.jsonl")
    config["save_interval"] = 500
    options = {"rgb_python": str(WORKSPACE / ".venv-signrep/bin/python"),
               "pose_python": str(WORKSPACE / ".venv-shubert-native/bin/python"),
               "teacher_weights": str(ROOT / "runtime/teacher/models"),
               "pose_workers": 4, "data_workers": 4, "cpu_threads": 2,
               "worker_threads": 2, "prefetch_factor": 2, "gpu_memory_fraction": .22,
               "inference_batch_size": 4, "retrieval_query_chunk": 128,
               "checkpoint_seconds": 300, "service": SERVICE}
    pins = read_json(ROOT / "evidence/kinetics_weights.json")
    if sha256(config["checkpoint"]) != pins["sha256"]:
        raise ValueError("Official Kinetics checkpoint hash mismatch")
    weights = {p.name: sha256(p) for p in Path(options["teacher_weights"]).glob("*.ckpt")}
    teacher_pin = read_json(ROOT / "evidence/teacher_weights.json")
    for name, value in teacher_pin["weights"].items():
        if weights[Path(name).name] != value["sha256"]:
            raise ValueError("Pose teacher checkpoint hash mismatch")
    atomic_json(job / "config.json", config)
    atomic_json(job / "options.json", options)
    atomic_json(job / "control.json", {"stage_limit": 50000, "test_authorized": False})
    progress(job, "SETUP", detail="Building immutable full-corpus membership")
    rows, csv_pins = build_manifests(job, args.dataset.resolve(), config["seed"])
    snapshot(job)
    progress(job, "SETUP", detail="Verifying and reusing original raw pose caches")
    reused = reuse_raw(job, rows["train"], weights)
    input_paths = [job / "config.json", job / "options.json", job / "source.jsonl"]
    input_paths += list((job / "manifests").glob("*.jsonl"))
    input_paths += [Path(config["checkpoint"])] + list(Path(options["teacher_weights"]).glob("*.ckpt"))
    input_pins = {str(p): sha256(p) for p in input_paths}
    input_pins.update(csv_pins)
    atomic_json(job / "input_pins.json", input_pins)
    unit_dir = Path.home() / ".config/systemd/user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit = unit_dir / SERVICE
    unit.write_text(unit_text(job, options))
    subprocess.run(["systemd-analyze", "--user", "verify", str(unit)], check=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", SERVICE], check=True)
    atomic_json(job / "setup.json", {"created_unix": time.time(), "service": SERVICE,
                "job": str(job), "raw_caches_reused": reused, "train_videos": len(rows["train"]),
                "stage_limit": 50000, "total_schedule_steps": 100000, "training_started": False})
    progress(job, "READY", train_videos=len(rows["train"]), raw_caches_reused=reused,
             stage_limit=50000, service=SERVICE, started=False)
    if args.start:
        subprocess.run(["systemctl", "--user", "start", SERVICE], check=True)
    print(json.dumps(read_json(job / "setup.json"), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, default=ROOT / "runtime/full_aslc_kinetics")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pretrain_full_aslc_kinetics.yaml")
    parser.add_argument("--dataset", type=Path, default=Path("/home/dongvk/datasets/ASL_Citizen"))
    parser.add_argument("--start", action="store_true")
    setup(parser.parse_args())


if __name__ == "__main__":
    main()
