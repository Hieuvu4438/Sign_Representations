#!/bin/bash
# EX15-v1.3 runner (replaces run_ex15.sh after MP training): adds S (SignRep-style) before Stage 2.
cd /home/haipd/Sign_Representations/method/jepa_targets
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
L=runtime/logs
while kill -0 781612 2>/dev/null; do sleep 60; done   # MP training started by run_ex15.sh
run_arm () {
  name=$1; shift
  if [ ! -f runtime/$name/last.pt ]; then
    rm -rf runtime/$name
    $PY jt/train.py --out runtime/$name "$@" > $L/train_$name.log 2>&1 || { echo "FAILED train $name"; return; }
  fi
  [ -f runtime/feats/$name/windows.npy ] || $PY jt/extract.py --ckpt runtime/$name/last.pt --name $name > $L/extract_$name.log 2>&1 || { echo "FAILED extract $name"; return; }
  $PY jt/evaluate.py --feats A0_vjepa,$name --ref A0_vjepa --out runtime/eval_$name.json > $L/eval_$name.log 2>&1 || echo "FAILED eval $name"
  echo "DONE $name $(date -Iseconds)"
}
run_arm MP --pixel --priors POSE
run_arm JP --latent --priors POSE
run_arm S --priors POSE --vicreg 1.0
run_arm JREL --latent --priors REL
run_arm JMOT --latent --priors MOT
run_arm JNM --latent --priors NM
run_arm JHOG --latent --hog
echo ALLDONE
