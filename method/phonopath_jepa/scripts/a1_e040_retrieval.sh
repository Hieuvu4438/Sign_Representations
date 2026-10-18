#!/usr/bin/env bash
set -euo pipefail
PP_EVAL_JOB=/home/haipd/Sign_Representations/method/phonopath_jepa/runtime/retrieval_e040
PP_EVAL_PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
case "${1:-status}" in
 status|pause|resume) exec "$PP_EVAL_PY" "$PP_EVAL_JOB/ops/retrieval_job.py" "${1:-status}" --job "$PP_EVAL_JOB" ;;
 logs) exec tail -n 20 -F "$PP_EVAL_JOB/service.log" "$PP_EVAL_JOB/extracting_train.log" ;;
 *) echo 'Usage: a1_e040_retrieval.sh status|logs|pause|resume' >&2; exit 2 ;;
esac
