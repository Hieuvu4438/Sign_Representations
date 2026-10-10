#!/bin/bash
# EX15-v1 runner: one GPU job at a time (user rule). Train -> extract val -> evaluate vs A0, per arm.
cd /home/haipd/Sign_Representations/method/jepa_targets
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
L=runtime/logs
run_arm () {  # name, train flags
  name=$1; shift
  if [ ! -f runtime/$name/last.pt ]; then
    $PY jt/train.py --out runtime/$name "$@" > $L/train_$name.log 2>&1 || { echo "FAILED train $name"; return; }
  fi
  [ -f runtime/feats/$name/windows.npy ] || $PY jt/extract.py --ckpt runtime/$name/last.pt --name $name > $L/extract_$name.log 2>&1 || { echo "FAILED extract $name"; return; }
  $PY jt/evaluate.py --feats A0_vjepa,$name --ref A0_vjepa --out runtime/eval_$name.json > $L/eval_$name.log 2>&1 || echo "FAILED eval $name"
  echo "DONE $name $(date -Iseconds)"
}
# Stage 1: paradigm (+ pose priors)
run_arm M --pixel
run_arm J --latent
run_arm MP --pixel --priors POSE
run_arm JP --latent --priors POSE
# Stage 2: targets on top of JEPA
run_arm JREL --latent --priors REL
run_arm JMOT --latent --priors MOT
run_arm JNM --latent --priors NM
run_arm JHOG --latent --hog
echo ALLDONE
