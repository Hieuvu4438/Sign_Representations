# REPORT001 — rebuild joint H002 pilot eligibility and table

Registered before implementation/execution, H002/P002_v1, CPU≤60s/1thread,
seed1701,0GPU/new trials. Inputs immutable completed QUAL001 and TIME001 with
all output/status hashes. Recompute timing summaries from all105 raw rows;
strict six-policy/five-source/three-repeat coverage, counts and component sums.
Join mean DCG/count and actual timing floors without changing thresholds.
Selection rule largest full-cohort window reduction, higher DCG, lexical name.
Save machine-readable joined result, CSV and Markdown table with exact units,
scope and input hashes. No claim of confirmatory noninferiority or novel sampler.
New report script leaves tested sampler/scorer/encoder source unchanged.
Config REPORT001.json; output method/frozen_probe/runtime/REPORT001_20261004_P002v1_s1701.
Exact command once CLI exists, before execution.

Exact command registered before execution:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/summarize_window_budget.py --config method/frozen_probe/configs/REPORT001.json --seed 1701 --run-id REPORT001_20261004_P002v1_s1701 --output-dir method/frozen_probe/runtime/REPORT001_20261004_P002v1_s1701 --quality-run method/frozen_probe/runtime/QUAL001_20261004_P002v1_s1701 --timing-run method/frozen_probe/runtime/TIME001_20261004_P002v1_s1701
```

Observed completed exit0, wall0.016999s, GPU0.000000000h.
Original outputs/status/snapshot immutable; durable copies retained.
