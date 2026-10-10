#!/bin/bash
# EX16-v1 runner: waits for free GPU memory (shared GPU), one job at a time, retries OOM.
cd /home/haipd/Sign_Representations/method/jepa_targets
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
L=runtime/logs; S=/tmp/claude-1007/-home-haipd-Sign-Representations/d41b57da-e74c-49f7-8525-a6e5e8b32980/scratchpad
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
wait_mem () { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "$1" ]; do sleep 120; done; }
retry () {  # name, min_free_MiB, command...
  name=$1; need=$2; shift 2
  for t in 1 2 3 4 5 6; do
    wait_mem $need; echo "start $name try $t $(date -Iseconds)"
    "$@" > $L/$name.log 2>&1 && return 0
    if grep -q OutOfMemoryError $L/$name.log; then echo "OOM $name try $t"; cleanup_$name 2>/dev/null; sleep 300; continue; fi
    echo "FAILED $name (non-OOM)"; return 1
  done; return 1
}
ext () { name=$1; shift; eval "cleanup_extract_$name () { rm -rf runtime/feats/$name; }"; [ -f runtime/feats/$name/windows.npy ] || { rm -rf runtime/feats/$name; retry extract_$name 8000 $PY jt/extract.py --name $name "$@"; }; }
train16 () { arm=$1; need=$2; low=$(echo $arm | tr A-Z a-z); eval "cleanup_train_$arm () { rm -rf runtime/$arm; }"; [ -f runtime/$arm/last.pt ] || retry train_$arm $need $PY jt/ex16.py --arm $low --out runtime/$arm; }
# PRIORITY (user 2026-10-08): SignRep reproduction vs V-JEPA (JV3D) first
train16 SR3D 14000
train16 JV3D 20000
ext SR3D --kind hiera --ckpt runtime/SR3D/last.pt --weights model
ext JV3D --kind vjepa_fc --ckpt runtime/JV3D/last.pt --weights encoder
ext JV3D_ema --kind vjepa_fc --ckpt runtime/JV3D/last.pt --weights ema_encoder
ext ASLC002 --kind hiera --ckpt ../jepa_baseline/runtime/ASLC002/train/last.model.pt --weights model
$PY jt/evaluate.py --feats SR3D,JV3D,JV3D_ema,ASLC002 --ref SR3D --out runtime/eval_ex16_main.json > $L/eval_ex16_main.log 2>&1 || echo "FAILED eval_ex16_main"
echo EX16MAINDONE $(date -Iseconds)
# then JEPA on SignRep's own Hiera-B init (paradigm-only comparison)
cleanup_smoke_jh3d () { rm -rf $S/ex16_jh3d; }
rm -rf $S/ex16_jh3d
if retry smoke_jh3d 12000 $PY jt/ex16.py --arm jh3d --out $S/ex16_jh3d --steps 6 --videos 2; then
  rm -rf $S/ex16_jh3d
  train16 JH3D 22000
  ext JH3D --kind hiera --ckpt runtime/JH3D/last.pt --weights model
  ext JH3D_ema --kind hiera --ckpt runtime/JH3D/last.pt --weights ema_model
else echo "STOPJH: jh3d smoke failed"; fi
$PY jt/evaluate.py --feats SR3D,JV3D,JH3D,JV3D_ema,JH3D_ema,ASLC002 --ref SR3D --out runtime/eval_ex16.json > $L/eval_ex16.log 2>&1 || echo "FAILED eval_ex16"
echo EX16DONE $(date -Iseconds)
# EX15 leftovers that hit OOM (other users' jobs filled the GPU)
for spec in "JHOG --latent --hog" "JPRM --latent --priors POSE,REL,MOT"; do
  set -- $spec; name=$1; shift
  eval "cleanup_train_$name () { rm -rf runtime/$name; }"
  [ -f runtime/$name/last.pt ] || retry train_$name 20000 $PY jt/train.py --out runtime/$name "$@" || continue
  ext $name --ckpt runtime/$name/last.pt
  $PY jt/evaluate.py --feats A0_vjepa,$name --ref A0_vjepa --out runtime/eval_$name.json > $L/eval_$name.log 2>&1
done
echo ALLDONE $(date -Iseconds)
