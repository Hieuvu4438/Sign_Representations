#!/bin/bash
# sequential extraction (1 process at a time; server rule). Logs in results/extract.log
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
cd /home/haipd/Sign_Representations/docs/proposal17/diagnostics
nice -n 10 $PY emb_variants.py --split val --every 2 --workers 3 --threads 6
nice -n 10 $PY emb_variants.py --split train --every 2 --workers 3 --threads 6
for m in face arm hand all; do
  nice -n 10 $PY emb_variants.py --split val --every 2 --morph $m --n_morph 1500 --workers 3 --threads 6 --only teacher
done
echo ALLDONE
