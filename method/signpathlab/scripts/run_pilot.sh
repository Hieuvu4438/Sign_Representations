#!/usr/bin/env bash
set -euo pipefail
SIGNPATH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SIGNPATH_ROOT"
if [ "$#" -eq 0 ]; then
  set -- --configs configs/verified_pilot/E*_seed*.yaml --device cuda:0 --output artifacts/verified_pilot/all_runs.json
fi
exec "${SIGNPATH_PYTHON:-$SIGNPATH_ROOT/.venv-locked/bin/python}" -m signpath.cli experiments run "$@"
