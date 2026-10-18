#!/usr/bin/env bash
set -euo pipefail
SIGNPATH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SIGNPATH_ROOT"
SIGNPATH_JOB=artifacts/full_aslc/job_seed17
mkdir -p "$SIGNPATH_JOB"
if [ -e "$SIGNPATH_JOB/status.json" ]; then
  echo "Job already has status; inspect it and use an explicit resume command."
  exit 2
fi
nohup setsid env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$SIGNPATH_ROOT/.venv-locked/bin/python" scripts/run_full_aslc.py \
  --config configs/full_aslc/E04_seed17.yaml --device cuda:0 --workers 1 \
  --job-dir "$SIGNPATH_JOB" > "$SIGNPATH_JOB/output.log" 2>&1 < /dev/null &
echo "$!" > "$SIGNPATH_JOB/launcher.pid"
echo "Background job launched. PID: $!; status/log: $SIGNPATH_JOB"
