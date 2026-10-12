"""Locked WLASL2000 train-gallery/test-query evaluation, with no downstream training.

CPU commands: prepare, public, report, status. `run` queues frozen GPU extraction.
No optimizer, classifier fitting, whitening fitting, or validation-based selection.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "method/obs_jepa"
DEFAULT_OUT = ROOT / "runtime/wlasl2000_frozen"
DATA = Path("/home/dongvk/datasets/WLASL2000")
PYTHON = REPO / ".venv-signrep/bin/python"
PAPER = "https://arxiv.org/html/2503.08529v1"
ORDER = ["SignRep_public", "JH3D", "JV3D", "SR3D", "SRVJ", "ASLC002", "A00",
         "A11_AD", "A10_AD", "A10_AR", "A12_AD", "A13_AD", "A10c_AD", "B11", "B12", "B13", "B10"]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).open()]


def protocol(out):
    p = json.loads((out / "protocol.json").read_text())
    for split in ("train", "test"):
        if sha256(out / f"{split}.jsonl") != p["manifest_sha256"][split]:
            raise RuntimeError("Locked manifest changed")
    return p


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        p = protocol(out)
        print("LOCK EXISTS", p["created_at"], flush=True)
        return p
    meta = json.loads((DATA / "preprocess/nslt_2000.json").read_text())
    classes = [line.split("\t", 1)[1].strip()
               for line in (DATA / "preprocess/wlasl_class_list.txt").read_text().splitlines() if line.strip()]
    assert len(classes) == 2000 and len(set(classes)) == 2000
    manifests = {}
    for split, count in (("train", 14296), ("test", 2879)):
        rows = []
        for vid in sorted(meta):
            r = meta[vid]
            if r["subset"] != split:
                continue
            video = DATA / "WLASL2000" / f"{vid}.mp4"
            kp = DATA / "keypoint" / f"{vid}.pkl"
            if not video.is_file() or not video.stat().st_size or not kp.is_file():
                raise RuntimeError(f"Missing video/keypoint: {vid}")
            label = int(r["action"][0])
            rows.append(dict(sample_id=vid, video=str(video), gloss=classes[label], label=label,
                             split=split, role="gallery" if split == "train" else "query"))
        assert len(rows) == count
        assert len({r["label"] for r in rows}) == 2000
        manifests[split] = rows
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert not ({r["sample_id"] for r in manifests["train"]} & {r["sample_id"] for r in manifests["test"]})
    detail_path = REPO / "docs/proposal11/SO_SANH_METHODS_DCG_RECALL_ASL_CITIZEN.csv"
    with detail_path.open(encoding="utf-8-sig", newline="") as f:
        details = {r["Method"]: r for r in csv.DictReader(f)}
    models = {}
    for name in ORDER:
        if name == "SignRep_public":
            spec = dict(kind="public", ckpt=str(REPO / "checkpoints/signrep/ckpt.pt"), weights="model", size=224)
        else:
            old = json.loads((ROOT / f"runtime/feats/{name}/aslc300/meta.json").read_text())
            spec = {key: old[key] for key in ("kind", "ckpt", "weights", "size")}
        cp = Path(spec["ckpt"]) if spec["ckpt"] != "vjepa" else (
            REPO / "method/signrel_jepa/assets/vjepa2_1_vitb_dist_vitG_384.pt")
        spec.update(checkpoint_file=str(cp), checkpoint_sha256=sha256(cp), details=details[name])
        spec["activity_weighted_available"] = name in {"SignRep_public", "SR3D", "JH3D", "SRVJ", "ASLC002"}
        spec["window_frames"] = 64 if spec["kind"] == "graph" else 16
        spec["window_stride"] = 32 if spec["kind"] == "graph" else 2
        spec["preprocessing"] = ("COCO-WholeBody 67-node mapping; ASL Citizen train normalization and confidence thresholds unchanged; effective fps<=25"
                                 if spec["kind"] == "graph" else
                                 "RGB native frames; letterbox 224" if spec["kind"] == "rgb" else "RGB native frames; squash 224")
        models[name] = spec
    p = dict(created_at=now(), source=PAPER, dataset="WLASL2000 local pretrimmed videos",
             dataset_root=str(DATA), annotation_sha256=sha256(DATA / "preprocess/nslt_2000.json"),
             classlist_sha256=sha256(DATA / "preprocess/wlasl_class_list.txt"),
             manifest_sha256={s: sha256(out / f"{s}.jsonl") for s in manifests},
             n_classes=2000, gallery_split="train", gallery_videos=14296, query_split="test", query_videos=2879,
             downstream_training=False, validation_used=False, checkpoint_selection="Existing final checkpoints fixed before WLASL test evaluation",
             similarity="Cosine on clip-L2 embeddings; gloss score=max over gallery videos in gloss",
             rank="1 + number of OTHER glosses scoring >= the true gloss (pessimistic ties)",
             metrics="DCG=mean(1/log2(1+r)); Rec@K=mean(r<=K); output scaled by 100",
             pooling="Raw window mean then clip L2; weighted: max(left_activity,right_activity), clamp [0,1], uniform fallback when sum<1e-6",
             short_rgb="Repeat last frame to 16; full windows only, no extra tail window",
             short_graph="Existing 64-frame training interface: pad unobserved nodes; no synthetic pose; keep zero embeddings",
             adaptations=["B models require 64-frame landmark windows, stride32, rather than SignRep RGB16/stride2",
                          "A models retain letterbox spatial preprocessing; all local WLASL videos in existing cache have fps25",
                          "JV3D checkpoint does not contain its trained activity head; report avg only",
                          "A/B/A00 have no activity head; report avg only",
                          "Local resized/trimmed WLASL copy; this does not assert identical video release/preprocessing to paper"],
             models=models)
    stats = ROOT / "runtime/b_stats_T64.npz"
    p["graph_statistics"] = dict(path=str(stats), sha256=sha256(stats))
    save_json(out / "protocol.json", p)
    (out / "logs").mkdir(exist_ok=True)
    (out / "results").mkdir(exist_ok=True)
    print("LOCKED", out, "17 models; 14296 gallery + 2879 test query; 2000 glosses", flush=True)
    return p


def l2(x):
    x = np.asarray(x, np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite embedding")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def pool_windows(features, active=None):
    x = np.asarray(features, np.float32)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Empty/nonfinite windows")
    result = {"avg": x.mean(0)}
    fallback = False
    if active is not None:
        a = np.asarray(active, np.float32)
        if a.shape != (len(x), 2) or not np.isfinite(a).all():
            raise ValueError("Invalid activity outputs")
        w = np.clip(a.max(1), 0, 1)
        fallback = bool(w.sum() < 1e-6)
        result["weighted"] = result["avg"] if fallback else (x * w[:, None]).sum(0) / w.sum()
    return result, fallback


def score_retrieval(g, q, gl, ql, n_classes, chunk=128):
    g, q = l2(g), l2(q)
    gl, ql = np.asarray(gl, int), np.asarray(ql, int)
    if set(np.unique(gl)) != set(range(n_classes)) or not set(ql) <= set(gl):
        raise ValueError("Gallery/query class coverage mismatch")
    # Sort once, then reduce contiguous gallery groups to class maxima.
    order = np.argsort(gl, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(gl[order]) != 0])
    G = np.ascontiguousarray(g[order].T)
    ranks, tied = [], 0
    for i in range(0, len(q), chunk):
        sim = q[i:i + chunk] @ G
        scores = np.maximum.reduceat(sim, starts, axis=1)
        labels = ql[i:i + chunk]
        true = scores[np.arange(len(labels)), labels]
        r = (scores >= true[:, None]).sum(1)
        ties = (scores == true[:, None]).sum(1) > 1
        # Independent nearest-video and sort checks wherever there is no true-score tie.
        nearest = gl[order][sim.argmax(1)] == labels
        sorted_rank = (np.argsort(-scores, axis=1, kind="stable") == labels[:, None]).argmax(1) + 1
        assert np.array_equal(nearest[~ties], (r <= 1)[~ties])
        assert np.array_equal(r[~ties], sorted_rank[~ties])
        tied += int(ties.sum()); ranks.append(r)
    r = np.concatenate(ranks)
    return dict(DCG=float(np.mean(1 / np.log2(1 + r))), Rec1=float(np.mean(r <= 1)),
                Rec5=float(np.mean(r <= 5)), queries=len(r), true_score_ties=tied), r


def evaluate_pooled(out, name, gallery, query, meta):
    p = protocol(out)
    gr, qr = (read_rows(out / f"{s}.jsonl") for s in ("train", "test"))
    result = dict(method=name, evaluated_at=now(), frozen=True, downstream_training=False,
                  checkpoint_sha256=p["models"][name]["checkpoint_sha256"], variants={}, extraction=meta)
    for variant in ("avg", "weighted"):
        if variant not in gallery:
            continue
        assert len(gallery[variant]) == len(gr) and len(query[variant]) == len(qr)
        m, r = score_retrieval(gallery[variant], query[variant], [x["label"] for x in gr],
                               [x["label"] for x in qr], p["n_classes"])
        np.save(out / "results" / f"{name}_{variant}_ranks.npy", r)
        result["variants"][variant] = m
        print(name, variant, json.dumps(m), flush=True)
    save_json(out / "results" / f"{name}.json", result)
    report(out)
    return result


def public(out):
    p = protocol(out)
    name = "SignRep_public"
    if (out / f"results/{name}.json").exists():
        return
    pooled, metadata = {}, {}
    for split in ("train", "test"):
        cache = REPO / f"method/p4_signssl/runtime/feats/wlasl_{split}_s2"
        cm = json.loads((cache / "meta.json").read_text())
        assert cm["sha256"] == p["models"][name]["checkpoint_sha256"]
        assert cm["stride"] == 2 and cm["perturb"] == "none" and cm["dataset"] == "wlasl"
        rows, cached = read_rows(out / f"{split}.jsonl"), read_rows(cache / "clips.jsonl")
        assert len(rows) == len(cached)
        for r, c in zip(rows, cached):
            assert r["sample_id"] == c["sample_id"] and r["gloss"] == c["gloss"] and r["split"] == c["split"]
            assert Path(r["video"]).resolve() == Path(c["path"]).resolve()
        windows = np.load(cache / "windows_features.npy", mmap_mode="r")
        active = np.load(cache / "windows_active.npy", mmap_mode="r")
        offsets = np.load(cache / "offsets.npy")
        starts = np.load(cache / "window_start.npy")
        values = {"avg": [], "weighted": []}; fb = 0
        assert len(offsets) == len(rows) + 1 and offsets[-1] == len(windows) == len(active) == len(starts)
        for i, row in enumerate(cached):
            a, b = offsets[i:i + 2]
            expected = np.arange(0, max(row["n_frames"], 16) - 15, 2)
            assert np.array_equal(starts[a:b], expected)
            v, fallback = pool_windows(windows[a:b], active[a:b]); fb += fallback
            for key in values:
                values[key].append(v[key])
        pooled[split] = {key: np.stack(v) for key, v in values.items()}
        metadata[split] = dict(cache=str(cache), original_meta=cm, zero_activity_fallback=fb)
    evaluate_pooled(out, name, pooled["train"], pooled["test"], metadata)


def report(out):
    p = protocol(out)
    selection_path = out / "selection.json"
    selected = json.loads(selection_path.read_text())["selected_models"] if selection_path.exists() else list(p["models"])
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(p["models"]):
        raise ValueError("Invalid selected model list")
    # Retain an already computed public reference without queuing it for extraction.
    references = ["SignRep_public"] if "SignRep_public" not in selected and (out / "results/SignRep_public.json").exists() else []
    path = out / "comparison.csv"
    headers = ["Method", "Mô tả tổng quan", "Input", "Dataset train method", "Số video train method",
               "Số video train thêm trên WLASL2000", "Dataset eval", "Số gloss", "Gallery split", "Số video gallery",
               "Query split", "Số video query", "Cách eval", "Pooling", "Window frame", "Stride", "Preprocessing",
               "DCG (x100)", "Rec@1 (%)", "Rec@5 (%)", "Trạng thái", "Ghi chú"]
    rows = []
    for name in references + selected:
        spec = p["models"][name]
        rp = out / "results" / f"{name}.json"
        result = json.loads(rp.read_text()) if rp.exists() else None
        variants = ["avg", "weighted"] if spec["activity_weighted_available"] else ["avg"]
        d = spec["details"]
        for v in variants:
            m = result["variants"].get(v) if result else None
            notes = "Checkpoint cuối giữ nguyên; không chọn model/hyperparameter theo WLASL test."
            if name in references:
                notes += " Kết quả tham chiếu đã có sẵn; không nằm trong hàng đợi chạy mới."
            if not spec["activity_weighted_available"]:
                notes += " Không có activity head trong checkpoint đã lưu; chỉ báo avg."
            rows.append([name, d["Mô tả tổng quan method"], d["Input"], d["Dataset train method"], d["Số video train method"],
                         0, "WLASL2000 (local pretrimmed)", 2000, "train", 14296, "test", 2879,
                         "Encoder đóng băng; raw window pooling -> L2 video -> cosine; gloss=max gallery cosine; DCG=mean(1/log2(1+r)); Rec@K=mean(r<=K); không classifier/PCA/fine-tune",
                         v, spec["window_frames"], spec["window_stride"], spec["preprocessing"],
                         *([f"{100*m[k]:.2f}" for k in ("DCG", "Rec1", "Rec5")] if m else ["", "", ""]),
                         "done" if m else "pending", notes])
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f); writer.writerow(headers); writer.writerows(rows)
    tmp.replace(path)
    done = sum((out / f"results/{n}.json").exists() for n in selected)
    save_json(out / "progress.json", dict(updated_at=now(), done_models=done, total_models=len(selected),
                                          selected_models=selected, retained_reference_models=references,
                                          csv=str(path), scope="Selected EX16/proposal11 methods; avg + available activity-weighted variants"))
    print(f"REPORT {done}/{len(selected)} selected models (+{len(references)} cached references): {path}", flush=True)


def own_gpu_jobs():
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    jobs = []
    for line in text.splitlines():
        try:
            pid = int(line.split(",")[0]); proc = Path(f"/proc/{pid}")
            if proc.stat().st_uid == os.getuid():
                jobs.append(pid)
        except (ValueError, FileNotFoundError, PermissionError):
            continue
    # Include jobs before they allocate CUDA memory, closing the initialization gap.
    known = ("pj/train.py", "pj/extract.py", "pj/evaluate.py", "oj/train_rgb.py", "oj/train_graph.py",
             "oj/extract.py", "oj/evaluate.py", "jt/train.py", "jt/ex16.py", "jt/extract.py", "jt/evaluate.py")
    for proc in Path("/proc").iterdir():
        try:
            if not proc.name.isdigit() or int(proc.name) == os.getpid() or proc.stat().st_uid != os.getuid():
                continue
            args = proc.joinpath("cmdline").read_bytes().decode(errors="replace").split("\0")
            if args and "python" in Path(args[0]).name and any(arg.endswith(known) for arg in args[1:]):
                jobs.append(int(proc.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return sorted(set(jobs))


def gpu_ready(busy, free_mib, min_free_mib, allow_shared_gpu):
    return free_mib >= min_free_mib and (allow_shared_gpu or not busy)


def run(out, names, min_free_mib, timeout_hours, allow_shared_gpu=False):
    prepare(out)
    with (out / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save_json(out / "selection.json", dict(updated_at=now(), selected_models=names,
                                               protocol_unchanged=True, cached_public_reference_retained=True,
                                               allow_shared_gpu=allow_shared_gpu))
        save_json(out / "runner.json", dict(pid=os.getpid(), started_at=now(), status="running", selected=names,
                                            allow_shared_gpu=allow_shared_gpu))
        if "SignRep_public" in names:
            public(out)
        report(out)
        for name in names:
            if name == "SignRep_public" or (out / f"results/{name}.json").exists():
                continue
            last = 0
            while True:
                busy = own_gpu_jobs()
                free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
                if gpu_ready(busy, free, min_free_mib, allow_shared_gpu):
                    break
                if time.monotonic() - last >= 60:
                    state = dict(pid=os.getpid(), status="waiting_gpu", selected=names, next_method=name, own_gpu_pids=busy,
                                 free_mib=free, need_mib=min_free_mib, updated_at=now(), allow_shared_gpu=allow_shared_gpu)
                    save_json(out / "runner.json", state); print("WAIT", json.dumps(state), flush=True); last = time.monotonic()
                time.sleep(10)
            cmd = [str(PYTHON), "-u", str(ROOT / "oj/extract_wlasl.py"), "--out", str(out), "--name", name]
            # Only exclusive mode asks the existing schedulers to wait for this worker.
            if not allow_shared_gpu:
                cmd += ["--scheduler-tag", "oj/extract.py:wlasl2000"]
            print("START", now(), name, flush=True)
            with (out / "logs" / f"{name}.log").open("a") as log:
                child = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                save_json(out / "runner.json", dict(pid=os.getpid(), child_pid=child.pid, status="extracting", selected=names, method=name, started_at=now(), allow_shared_gpu=allow_shared_gpu))
                try:
                    rc = child.wait(timeout=timeout_hours * 3600)
                except subprocess.TimeoutExpired:
                    print("HARD TIMEOUT", name, flush=True)
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); child.wait()
                    rc = 124
            if rc:
                save_json(out / "runner.json", dict(pid=os.getpid(), status="failed", selected=names, method=name, exit_code=rc, updated_at=now(), allow_shared_gpu=allow_shared_gpu))
                raise RuntimeError(f"{name} failed rc={rc}; see logs/{name}.log; no automatic retry")
            report(out)
        save_json(out / "runner.json", dict(pid=os.getpid(), status="complete", selected=names, updated_at=now(), allow_shared_gpu=allow_shared_gpu))
        print("COMPLETE", now(), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "public", "report", "status", "run"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--names", default=None, help="Comma-separated models; resume the persisted selection when omitted")
    ap.add_argument("--min-free-mib", type=int, default=8000)
    ap.add_argument("--timeout-hours", type=float, default=12)
    ap.add_argument("--allow-shared-gpu", action=argparse.BooleanOptionalAction, default=None,
                    help="Allow other same-user GPU jobs; resume the persisted policy when omitted")
    a = ap.parse_args(); a.out = a.out.resolve()
    selection_path = a.out / "selection.json"
    names = (a.names.split(",") if a.names is not None else
             json.loads(selection_path.read_text())["selected_models"] if selection_path.exists() else list(ORDER))
    if not names or len(names) != len(set(names)) or not set(names) <= set(ORDER):
        ap.error("Unknown, duplicate, or empty model name")
    if a.command == "prepare":
        prepare(a.out); report(a.out)
    elif a.command == "public":
        public(a.out)
    elif a.command == "report":
        report(a.out)
    elif a.command == "run":
        selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
        allow_shared_gpu = (a.allow_shared_gpu if a.allow_shared_gpu is not None else
                            selection.get("allow_shared_gpu", False))
        try:
            run(a.out, names, a.min_free_mib, a.timeout_hours, allow_shared_gpu)
        except Exception as e:
            # Preserve model-specific failure details written by run().
            state_path = a.out / "runner.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if state.get("pid") == os.getpid():
                state.update(status="failed", error=str(e), updated_at=now()); save_json(state_path, state)
            raise
    else:
        for filename in ("progress.json", "runner.json"):
            path = a.out / filename
            print(filename, path.read_text() if path.exists() else "not started")


if __name__ == "__main__":
    main()
