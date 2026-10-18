#!/usr/bin/env bash
# usage: ckpt_readout.sh <config> <checkpoint> <tag>   -> runs/diagnostics/emb_<tag>/ + readout_<tag>.json (train/val only)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
D=runs/diagnostics/emb_$3
mkdir -p $D
for s in train val; do
  [ -f $D/$s.npz ] || $PY -m asm_jepa.cli.extract_embeddings --config $1 --checkpoint $2 --split $s --output $D/$s.npz --workers 8 > $D/extract_$s.log 2>&1
done
echo "== $3"; CUDA_VISIBLE_DEVICES="" $PY scripts/diagnostics/readout_diag.py $D --brief --out runs/diagnostics/readout_$3.json
