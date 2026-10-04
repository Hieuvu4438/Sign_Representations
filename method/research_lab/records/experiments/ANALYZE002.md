# ANALYZE002 — registered first real dev prediction analysis

Registered before execution, H001/P001_v4; ANALYZE002.json seed1701.
Input completed BASE002_20261004_P001v4_s1337, status/output hashes verified
by the analysis CLI. No selection changes after predictions; inherited
P001_v3/P001_v4 primary Top1 mean over2/4 labels and support seeds1701/2/3.
Compare ordered-ridge to strongest aggregate of the seven fixed baselines,
lexical tie break. Separately compare source-shuffled bins. Practical screen:
≥2pp strongest-baseline gain, ≥1pp shuffle gain, positive for≥2 support seeds.

Recompute all108 sets of metrics from complete378-query/100-class rankings;
strict cohort labels/participants and exact ID order. Retain every seed and
label curve. Paired participant bootstrap2000 replicates, seed1701, micro
weighting with repeated participant groups, percentile95% interval conditional
on dev-selected baseline. Six participants; no population/significance claim.
Fixed alpha1 baselines are not tuned; no confirmatory or novelty conclusion.

CPU/BLAS1thread, max60s,0GPU/HPO; immutable exclusive output
method/frozen_probe/runtime/ANALYZE002_20261004_P001v4_s1701.
Exact command registered before execution:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/analyze_baseline.py --config method/frozen_probe/configs/ANALYZE002.json --seed 1701 --run-id ANALYZE002_20261004_P001v4_s1701 --output-dir method/frozen_probe/runtime/ANALYZE002_20261004_P001v4_s1701 --prediction-run method/frozen_probe/runtime/BASE002_20261004_P001v4_s1337
```

## Observed completion

Completed exit0, wall0.169760s,0GPUh/HPO, test_access none.
Evidence [ANALYZE002_status.json](ANALYZE002_status.json).
[Analysis](ANALYZE002_analysis.json): ordered67.1958%, activity-nearest78.3951%,
difference−11.1993pp, conditional95% interval[−14.0257,−6.6321]pp; screen false.
Participant count **five**; [erratum](../protocols/P001_v4_participant_erratum.md).
