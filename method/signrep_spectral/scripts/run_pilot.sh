#!/usr/bin/env bash
# Pilot chain (dev only): each arm -> train -> export -> extract train,val -> retrieval/probe on val.
# Usage: scripts/run_pilot.sh <tag> <arm>...   arms: B0 B1 M1 B2 B3 B4
set -euo pipefail
cd "$(dirname "$0")/../../.."
TAG=$1; shift
PY=${SIGNREP_PY:-.venv-signrep/bin/python}
export PYTHONDONTWRITEBYTECODE=1
C=method/signrep_spectral/configs; R=method/signrep_spectral/runtime; P="--paths-config $C/local_paths.json"
declare -A CFG=([B1r]=baseline_geom_var [M1r]=spectral_mvp_var [B1]=baseline_geom [M1]=spectral_mvp [B2]=velocity_control [B3]=acceleration_control [B4]=window_time_control)
run() {  # skips a step whose run dir already finished with SUCCESS
  local rd="${@: -1}"
  if [ -f "$rd/status.json" ] && grep -q '"SUCCESS"' "$rd/status.json"; then echo "[skip] $rd"; return; fi
  echo "[$(date +%T)] $*"
  "$PY" -m method.signrep_spectral.cli "$@" 2>&1 | { grep --line-buffered -v -i warn || true; }
  grep -q '"SUCCESS"' "$rd/status.json" || { echo "step failed: $rd"; exit 1; }
}
for ARM in "$@"; do
  if [ "$ARM" = B0 ]; then
    run extract --config $C/pilot_base.json $P --reference-checkpoint --splits train,val --run-dir $R/${TAG}_B0_features
  else
    cfg=$C/${CFG[$ARM]}.json
    run train --config $cfg $P --run-dir $R/${TAG}_${ARM}_train
    run export --config $cfg $P --checkpoint $R/${TAG}_${ARM}_train/checkpoints/final.pt \
      --output $R/${TAG}_${ARM}_export/encoder.pt --run-dir $R/${TAG}_${ARM}_export
    run extract --config $cfg $P --encoder $R/${TAG}_${ARM}_export/encoder.pt --splits train,val --run-dir $R/${TAG}_${ARM}_features
  fi
  run evaluate-retrieval --config $C/pilot_base.json $P --features-dir $R/${TAG}_${ARM}_features --query-split val --run-dir $R/${TAG}_${ARM}_retrieval_val
  run evaluate-probe --config $C/pilot_base.json $P --features-dir $R/${TAG}_${ARM}_features --eval-split val --run-dir $R/${TAG}_${ARM}_probe_val
  run evaluate-probe --config $C/pilot_base.json $P --features-dir $R/${TAG}_${ARM}_features --eval-split val \
    --probe-config '{"standardize": true, "lr": 1e-3, "epochs": 40}' --run-dir $R/${TAG}_${ARM}_probe2_val
done
echo "[$(date +%T)] chain done"
