"""One-shot user status command; never polls or restarts the experiment."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", default="artifacts/full_aslc/job_seed17")
    args = parser.parse_args()
    job = Path(args.job_dir)
    report = json.loads((job / "status.json").read_text())
    try:
        os.kill(report["pid"], 0)
        process_alive = b"run_full_aslc.py" in Path(f"/proc/{report['pid']}/cmdline").read_bytes()
    except (OSError, KeyError):
        process_alive = False
    report["process_alive_now"] = process_alive
    if report["stage"].startswith("cache_"):
        path = job / f"cache.{report['stage'].removeprefix('cache_')}.json"
        if path.exists():
            report["cache_progress"] = json.loads(path.read_text())
    recorded = job / "resolved_config.json"
    training_root = (Path(json.loads(recorded.read_text())["config"]["experiment"]["output_dir"])
                     if recorded.exists() else Path("artifacts/full_aslc/E04_seed17"))
    logs = sorted(training_root.glob("*/logs.jsonl"))
    if logs:
        with logs[-1].open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 65536))
            for line in reversed(stream.read().splitlines()):
                try:
                    last = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                report["last_training_update"] = {key: last[key] for key in ("successful_step", "attempted_batches", "loss", "succeeded", "reason")}
                report["training_log"] = str(logs[-1])
                break
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
