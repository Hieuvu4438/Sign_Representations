#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNISIGN_PYTHON="${UNISIGN_PYTHON:-$REPO_ROOT/../../.venv-unisign/bin/python}"
if [[ ! -x "$UNISIGN_PYTHON" ]]; then
    echo "Missing inference Python: $UNISIGN_PYTHON (see docs/POSE_INFERENCE_SETUP.md)" >&2
    exit 1
fi
# The shared PyTorch 2.1.1 installation supplies CUDA 12 and cuDNN 8.
UNISIGN_LIBRARY_PATH="$("$UNISIGN_PYTHON" -c 'import importlib.util, pathlib, sys; print(":".join([str(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib"), str(pathlib.Path(sys.base_prefix) / "lib"), str(pathlib.Path(sys.base_prefix) / "targets/x86_64-linux/lib")]))')"
export LD_LIBRARY_PATH="$UNISIGN_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "$UNISIGN_PYTHON" "$REPO_ROOT/script/extract_pose_features.py" "$@"
