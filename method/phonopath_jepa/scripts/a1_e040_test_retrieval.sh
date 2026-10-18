#!/usr/bin/env bash
set -euo pipefail
PP_TEST_JOB=/home/haipd/Sign_Representations/method/phonopath_jepa/runtime/retrieval_e040_test
PP_TEST_PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
case "${1:-status}" in
 status|pause|resume) exec "$PP_TEST_PY" "$PP_TEST_JOB/ops/test_retrieval_job.py" "${1:-status}" --job "$PP_TEST_JOB" ;;
 logs) exec tail -n 20 -F "$PP_TEST_JOB/service.log" ;;
 *) echo 'Usage: a1_e040_test_retrieval.sh status|logs|pause|resume' >&2; exit 2 ;;
esac
