"""Start/stop/status and an explicit validation-reviewed extension of the same job."""

import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SERVICE = "signrep-full-aslc-kinetics.service"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "start", "stop", "restart", "continue-100k"])
    parser.add_argument("--job", type=Path, default=ROOT / "runtime/full_aslc_kinetics")
    args = parser.parse_args()
    job = args.job.resolve()
    if args.action == "status":
        subprocess.run(["systemctl", "--user", "show", SERVICE, "-p", "ActiveState", "-p", "SubState",
                        "-p", "MainPID", "-p", "NRestarts"])
        if (job / "status.txt").exists():
            print((job / "status.txt").read_text(), end="")
        print("Tail:", "tail -f", job / "progress.txt")
        return
    if args.action == "continue-100k":
        active = subprocess.run(["systemctl", "--user", "is-active", "--quiet", SERVICE]).returncode == 0
        if active:
            raise SystemExit("Stop the existing service before changing the stage limit")
        review = job / "review.json"
        if not review.exists() or json.loads(review.read_text())["stage_limit"] != 50000:
            raise SystemExit("The 50k stage must finish full validation review first")
        control = json.loads((job / "control.json").read_text())
        if control.get("test_authorized"):
            raise SystemExit("Do not extend a run after its test protocol has been authorized")
        control["stage_limit"] = 100000
        temporary = job / "control.json.tmp"
        with temporary.open("w") as file:
            json.dump(control, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, job / "control.json")
        subprocess.run(["systemctl", "--user", "reset-failed", SERVICE], check=True)
        subprocess.run(["systemctl", "--user", "start", SERVICE], check=True)
        return
    if args.action in {"start", "restart"}:
        subprocess.run(["systemctl", "--user", "reset-failed", SERVICE], check=True)
    subprocess.run(["systemctl", "--user", args.action, SERVICE], check=True)


if __name__ == "__main__":
    main()
