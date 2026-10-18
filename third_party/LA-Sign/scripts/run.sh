#!/usr/bin/env bash
set -euo pipefail
LA_SIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LA_SIGN_ROOT"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$LA_SIGN_ROOT/.venv/bin/python" loop_training.py "$@"
