#!/bin/bash
# EX15 follow-up (rule in EX15-v1: REL beat J -> run J+P+REL+MOT). Waits for run_ex15b.sh ALLDONE.
cd /home/haipd/Sign_Representations/method/jepa_targets
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
L=runtime/logs
until grep -q ALLDONE $L/ex15b_chain.log; do sleep 120; done
name=JPRM
[ -f runtime/$name/last.pt ] || $PY jt/train.py --out runtime/$name --latent --priors POSE,REL,MOT > $L/train_$name.log 2>&1
[ -f runtime/feats/$name/windows.npy ] || $PY jt/extract.py --ckpt runtime/$name/last.pt --name $name > $L/extract_$name.log 2>&1
$PY jt/evaluate.py --feats A0_vjepa,$name --ref A0_vjepa --out runtime/eval_$name.json > $L/eval_$name.log 2>&1
echo "DONE $name $(date -Iseconds)"; echo ALLDONE
