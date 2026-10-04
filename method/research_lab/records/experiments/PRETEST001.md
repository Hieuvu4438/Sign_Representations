# PRETEST001 — complete held-out source audit before H002 inference

Registered P003_v1/D009, CPU-only, seed1701, max1800s. No task score or GPU.
Run ID PRETEST001_20261004_H002_P003v1_741b0823_s1701.
Output method/frozen_probe/runtime/PRETEST001_20261004_H002_P003v1_741b0823_s1701.
Immutable freeze precedes first selected-test access. Every test row matching
fixed100 train classes must be retained; derived counts, exact hashes, complete
pixel/PTS decode, disjoint participant/files, no silent exclusions. Bounded near
screen candidates require source review. Estimate four real query recipes using
TIME001 p90 rates/safety1.5 and exact native counts, before a GPU launch.

Actual command:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-signrep/bin/python method/frozen_probe/scripts/prepare_confirmatory.py --config method/frozen_probe/configs/PRETEST001.json --freeze method/research_lab/records/protocols/P003_v1_freeze.json --seed 1701 --run-id PRETEST001_20261004_H002_P003v1_741b0823_s1701 --output-dir method/frozen_probe/runtime/PRETEST001_20261004_H002_P003v1_741b0823_s1701
```

Status is authoritative in the run output; completion not assumed by registration.
