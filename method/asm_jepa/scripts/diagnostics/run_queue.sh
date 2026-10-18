#!/usr/bin/env bash
# Sequential 5k-step diagnostic runs (one GPU job at a time), resumable: finished runs are skipped, an interrupted
# run resumes from its last.pt. Usage: scripts/diagnostics/run_queue.sh [variant ...]   (default: all)
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
OUT=runs/diagnostics
ALL=(control noglobal base mae lowface nobodyface temp lr1e4)
VARS=("${@:-${ALL[@]}}")
STEPS=5000
mkdir -p $OUT
for v in "${VARS[@]}"; do
  R=$OUT/$v
  [ -f $R/DONE ] && continue
  CMD=($PY -m asm_jepa.cli.pretrain --config configs/diagnostics/d_$v.yaml --run-dir $R
       --set train.max_steps=$STEPS --set train.warmup_steps=500 --set loss.teacher_temperature_warmup_steps=500
       --set eval.frozen_every_steps=1000 --set train.milestone_every_steps=0 --set train.checkpoint_every_steps=1000
       --set train.keep_step_checkpoints=1)
  [ -f $R/checkpoints/last.pt ] && CMD+=(--resume $R/checkpoints/last.pt)
  echo "[$(date -Is)] start $v" | tee -a $OUT/queue.log
  "${CMD[@]}" >> $OUT/$v.log 2>&1 || { echo "[$(date -Is)] FAILED $v (see $OUT/$v.log)" | tee -a $OUT/queue.log; continue; }
  cfg=configs/diagnostics/d_$v.yaml
  scripts/diagnostics/ckpt_readout.sh $cfg $R/checkpoints/last.pt q_$v >> $OUT/$v.log 2>&1 \
    || echo "[$(date -Is)] readout failed $v" | tee -a $OUT/queue.log
  rm -rf $OUT/emb_q_$v; rm -f $R/checkpoints/step_*.pt
  touch $R/DONE
  echo "[$(date -Is)] done $v" | tee -a $OUT/queue.log
done
$PY scripts/diagnostics/summarize.py > $OUT/summary.md
echo "[$(date -Is)] queue finished" | tee -a $OUT/queue.log
