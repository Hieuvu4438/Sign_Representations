# CONFAN002 — diagnosed serialization retry1 of frozen held-out analysis

CONFAN001 failed CPU-only on numpy.bool_ JSON serialization after computing in
memory; no analysis/metrics persisted or shown. Original run preserved, no GPU
inference invalidated or rerun. P003serializationerratum explicit after execution.
Only wrap unchanged cost comparison in Pythonbool, allscientificconfigs/metrics/
seeds/endpoints/thresholds unchanged;57strictJSON/regression checks pass. Exact
source diff checked; new technical gate links original pretest/warmupgates/SHA.
Same CONFAN001.json immutable config hash12d05a8e; new exclusive run/output.
CPU120s, seed1701, retry1ofmax2. Output actual metrics/errors/allgalleries pending.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/analyze_confirmatory.py --config method/frozen_probe/configs/CONFAN001.json --freeze method/research_lab/records/protocols/P003_v1_serialization_gate.json --prediction-run method/frozen_probe/runtime/CONFIRM001_20261004_H002_P003v1_7b94d089_s1701 --seed 1701 --run-id CONFAN002_20261004_H002_P003v1_12d05a8e_s1701 --output-dir method/frozen_probe/runtime/CONFAN002_20261004_H002_P003v1_12d05a8e_s1701
```
