"""One-shot human-readable log from existing artifacts; never touches training state."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def last_json_record(path):
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 131072))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeError):
            continue
    return None


def progress_lines(job):
    job = Path(job)
    status_path = job / "status.json"
    if not status_path.exists():
        return ["Đang chờ job tạo status.json."]
    state = json.loads(status_path.read_text())
    lines = [f"Trạng thái: {state['status']} | Giai đoạn: {state['stage']} | PID: {state.get('pid', '?')}"]
    execution = state.get("cache_execution", {})
    if execution:
        lines.append(f"Python process: {execution['python_processes']} | CPU threads: {execution['cpu_threads']} | DINO batch: {execution['dino_batch_size']}")
    for split in ("train", "val", "test"):
        path = job / f"cache.{split}.json"
        if path.exists():
            progress = json.loads(path.read_text())
            done, total = progress["completed"], progress["total"]
            lines.append(f"Cache {split}: {done}/{total} ({100 * done / total:.2f}%) | {progress['status']} | thời gian lượt này: {progress['elapsed_seconds'] / 3600:.2f} giờ")
    config_path = job / "resolved_config.json"
    if config_path.exists():
        recorded = json.loads(config_path.read_text())
        config = recorded["config"]
        root = Path(config["experiment"]["output_dir"])
        if not root.is_absolute():
            root = Path(recorded["project_root"]) / root
        logs = sorted(root.glob("*/logs.jsonl"), key=lambda p: p.stat().st_mtime_ns)
        if logs:
            last = last_json_record(logs[-1])
            if last is not None:
                loss = f"{last['loss']:.6f}" if isinstance(last.get("loss"), (int, float)) else "n/a"
                lines.append(f"Train: update {last['successful_step']}/{config['training']['max_successful_steps']} | loss={loss} | attempted batches={last['attempted_batches']} | update success={last['succeeded']}")
                if last.get("reason"):
                    lines.append(f"Skip reason: {last['reason']}")
        elif state["stage"].startswith("cache_"):
            lines.append("Optimizer chưa bắt đầu; đang chuẩn bị cache.")
    if state.get("error"):
        lines.append(f"Thông báo: {state['error']}")
    result_path = job / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        for split in ("val", "test"):
            metrics = result.get(f"{split}_metrics")
            if metrics:
                lines.append(f"Retrieval {split}: DCG={metrics['DCG'] * 100:.2f} | Rec@1={metrics['Rec@1'] * 100:.2f}% | Rec@5={metrics['Rec@5'] * 100:.2f}%")
    return lines


def write_snapshot(job):
    job = Path(job)
    timestamp = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(timespec="seconds")
    lines = progress_lines(job)
    snapshot = f"[{timestamp}]\n" + "\n".join(lines) + "\n"
    job.mkdir(parents=True, exist_ok=True)
    with (job / "progress.txt").open("a", encoding="utf-8") as stream:
        stream.write(snapshot + "\n")
    temporary = job / "status_latest.txt.tmp"
    temporary.write_text(snapshot, encoding="utf-8")
    temporary.replace(job / "status_latest.txt")
    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", default="artifacts/aslc3000/job_seed17")
    args = parser.parse_args()
    print(write_snapshot(args.job_dir), end="")


if __name__ == "__main__":
    main()
