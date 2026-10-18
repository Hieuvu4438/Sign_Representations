#!/bin/bash
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
cd /home/haipd/Sign_Representations/docs/proposal17/diagnostics
export OMP_NUM_THREADS=4
EMB=/tmp/claude-1007/-home-haipd-Sign-Representations/32c5829a-820b-4c3b-9d20-e66b08713d70/scratchpad/emb
rm -f $EMB/val_bal300_morph-* $EMB/val_morph-*
step() { echo "=== $1 $(date +%T)"; shift; nice -n 8 "$@"; }
for m in arm hand all nonuni; do for s in 1.15 1.3 1.5; do
  step "morph_${m}_${s}" $PY emb_variants.py --split val --every 2 --balanced 300 --morph $m --morph_scale $s --workers 3 --threads 6 --only teacher
done; done
for m in face arm hand all; do
  step "morph1500_$m" $PY emb_variants.py --split val --every 2 --morph $m --n_morph 1500 --workers 3 --threads 6 --only teacher
done
step morph_dose $PY morph_dose.py
step morph_effect $PY morph_effect.py results/morph_effect.json
step routed_controls $PY routed_controls.py
step cls_early $PY cls_early.py
echo R2FIXDONE
