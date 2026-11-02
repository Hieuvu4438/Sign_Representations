#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d /home/haipd/Sign_Representations/method/phonopath_jepa/runtime ]]; then
  if [[ "${1:-status}" == status ]]; then
    cat /home/haipd/Sign_Representations/method/phonopath_jepa/ARCHIVED.md
    exit 0
  fi
  echo 'Run archived; checkpoints/runtime were deleted. See method/phonopath_jepa/ARCHIVED.md.' >&2
  exit 1
fi
PP_FINAL_TEST_JOB=/home/haipd/Sign_Representations/method/phonopath_jepa/runtime/retrieval_e160_test
PP_FINAL_TEST_PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
case "${1:-status}" in
 status|pause|resume) exec "$PP_FINAL_TEST_PY" "$PP_FINAL_TEST_JOB/ops/official_test_job.py" "${1:-status}" --job "$PP_FINAL_TEST_JOB" ;;
 logs) exec tail -n 20 -F "$PP_FINAL_TEST_JOB/service.log" "$PP_FINAL_TEST_JOB/extracting_train.log" ;;
 *) echo 'Usage: a1_e160_official_test.sh status|logs|pause|resume' >&2; exit 2 ;;
esac
