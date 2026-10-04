# CONFAN001 — frozen held-out quality/cost, uncertainty and error analysis

P003_v1/H002/seed1701, existing pre-test CONFAN001 configSHA12d05a8ecd5656a021742cc120d8d8b437504596996cc344dbbd20f5d294083b.
CPU120s/oneBLASthread, no GPU. Required: CONFIRM001 terminal completed status,
all output hashes and actualquery/source coverage/parity verified. No scoring of
partial predictions and no inference-job duplicate/restart. Read every full15
ranking matrix; allmetrics/threegalleries, participant2000replicatepairedCI,
original quality/count/cost screen, train-frozen slices and deterministic case
categories. No new selection, policy, metric, threshold or fitting by test.

Command (registered, **not run yet**):
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/analyze_confirmatory.py --config method/frozen_probe/configs/CONFAN001.json --freeze method/research_lab/records/protocols/P003_v1_warmup_gate.json --prediction-run method/frozen_probe/runtime/CONFIRM001_20261004_H002_P003v1_7b94d089_s1701 --seed 1701 --run-id CONFAN001_20261004_H002_P003v1_12d05a8e_s1701 --output-dir method/frozen_probe/runtime/CONFAN001_20261004_H002_P003v1_12d05a8e_s1701
```

Expected outputs analysis.json/metrics.csv/table.md/raw input references/status.
Primary screen unchanged; auxiliary source coverage is separate and descriptive.
After completion independently inspect actual schema/metrics/errors/cost and
reconcile all terminal GPU attempts before any continue/pivot/kill decision.
