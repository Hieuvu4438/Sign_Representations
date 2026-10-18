#!/bin/bash
# runs after run_r1_fixes.sh (one process at a time)
cd /home/haipd/Sign_Representations/docs/proposal17/diagnostics
while ! grep -q R1FIXDONE results/r1_fixes.log; do sleep 30; done
OMP_NUM_THREADS=4 nice -n 8 /home/haipd/Sign_Representations/.venv-signrep/bin/python nn_head_cls.py > results/nn_head_cls.log 2>&1
OMP_NUM_THREADS=4 nice -n 8 /home/haipd/Sign_Representations/.venv-signrep/bin/python routed_coral.py > results/routed_coral.log 2>&1
echo R1FIX2DONE >> results/nn_head_cls.log
