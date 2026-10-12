#!/usr/bin/env bash
set -euo pipefail
SIGNPATH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SIGNPATH_ROOT"
if [ "$#" -eq 0 ]; then
  set -- --output artifacts/doctor.json
fi
exec "${SIGNPATH_PYTHON:-$SIGNPATH_ROOT/.venv-locked/bin/python}" -m signpath.cli doctor "$@"
