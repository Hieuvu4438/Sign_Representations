#!/bin/bash
# Sequential CPU diagnostics answering the independent review R1 (docs/proposal17/reviews/R1_independent_review.md). One process at a time.
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
cd /home/haipd/Sign_Representations/docs/proposal17/diagnostics
export OMP_NUM_THREADS=4
step() { echo "=== $1 $(date +%T)"; shift; nice -n 8 "$@"; }
step retention $PY retention_scalefree.py
step ladder_extra $PY ladder_extra.py
step nn_fullpool $PY nn_precision_fullpool.py
step kfree $PY kfree_cluster_corral.py
step coral_mb $PY coral_head_mb.py
step style_vars $PY style_vars.py results/style_vars.json
step morph_clean $PY emb_variants.py --split val --every 2 --balanced 300 --workers 3 --threads 6 --only teacher
for m in arm hand all nonuni; do for s in 1.15 1.3 1.5; do
  step "morph_$m_$s" $PY emb_variants.py --split val --every 2 --balanced 300 --morph $m --morph_scale $s --workers 3 --threads 6 --only teacher
done; done
step morph_dose $PY morph_dose.py
step morph_effect_rerun $PY morph_effect.py results/morph_effect.json
step cls_early $PY cls_early.py
echo R1FIXDONE
