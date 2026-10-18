#!/usr/bin/env bash
set -euo pipefail
LA_SIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LA_SIGN_ROOT"
command -v uv >/dev/null || { echo 'Install uv, then rerun this script.'; exit 1; }
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python "${LA_SIGN_PYTHON:-3.10}" .venv
fi
uv pip install --python .venv/bin/python -r requirements.txt
uv pip check --python .venv/bin/python
.venv/bin/python scripts/prepare_data.py "$@"
