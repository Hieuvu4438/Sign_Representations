#!/usr/bin/env bash
set -euo pipefail
PP_A1_ROOT=/home/haipd/Sign_Representations/method/phonopath_jepa
PP_A1_PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
PP_A1_JOB="$PP_A1_ROOT/runtime/full160"
case "${1:-status}" in
  status|pause|resume|checkpoint)
    exec "$PP_A1_PY" "$PP_A1_JOB/ops/full160_job.py" "${1:-status}" --job "$PP_A1_JOB"
    ;;
  logs)
    exec tail -n 30 -F "$PP_A1_JOB/train_stdout.log" "$PP_A1_JOB/service.log"
    ;;
  *)
    echo 'Usage: a1_full160.sh status|pause|resume|checkpoint|logs' >&2
    exit 2
    ;;
esac
