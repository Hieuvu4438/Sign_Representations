# CONFIRM001 — actual four-recipe held-out inference, no scoring

P003_v1, H002, scientific configSHA7b94d089b1b5a9a7192f267deb0f0ef94fac5b17c7ed6b4fa90019e3b9a83536.
Run ID CONFIRM001_20261004_H002_P003v1_7b94d089_s1701; max7100s conservative
GPU wall allowance; H002 remaining7142.295945s/program remaining11854.951374s.
PRETEST001 complete1217queries/11participants/100classes,94140nativeframes,
38238densewindows,10019eachsparsepolicy,68295actualfour-recipeencoderwindows.
Estimated3588.438359s including1.5safety+60setup; within the registered ceiling.
37,386,240pooled payload bytes planned in RAM; complete ranks/timings on root.
No exclusions; bounded2,057,947crosspairs/653,053eligible,0candidates. This is
not exhaustive uniqueness or known pretraining disjointness.

Original pre-test freeze accde498... retained. Technical warmup gate49f98590...
issued after metadata but before inference/scoring; same fixed dev warmup video,
incorrect train adjective corrected, all scientific configs unchanged,50checks
pass. See P003_v1_warmup_erratum and exact source diff. No failed GPU attempt.

Query inputs/preprocessing independently executed per four recipes; true sparse
selection precedes model, complete100-class rankings for all5methods×3galleries.
Gallery is train-only verified CACHE005, CPU setup separately recorded; query
cost excludes gallery encoding. Full_mean shares dense pass for quality. No test
scores read until complete terminal predictions. Allsource sparse/dense output
parity checked, no per-query accuracy feedback. Actual cost/quality pending.

Actual launch command:
```bash
set -o pipefail
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-signrep/bin/python method/frozen_probe/scripts/predict_confirmatory.py --config method/frozen_probe/configs/CONFIRM001.json --freeze method/research_lab/records/protocols/P003_v1_warmup_gate.json --prepared-run method/frozen_probe/runtime/PRETEST001_20261004_H002_P003v1_741b0823_s1701 --seed 1701 --run-id CONFIRM001_20261004_H002_P003v1_7b94d089_s1701 --output-dir method/frozen_probe/runtime/CONFIRM001_20261004_H002_P003v1_7b94d089_s1701 2>&1 | tee method/frozen_probe/runtime/CONFIRM001_20261004_H002_P003v1_7b94d089_s1701.stdout.log
```

Status/output hashes authoritative; terminal success cannot be assumed from launch.
Next after complete: frozen analyze_confirmatory.py with CONFAN001.json/new
technical gate, separate CPU run, score all methods/gallery subsets/slices/cases.
