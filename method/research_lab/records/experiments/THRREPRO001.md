# THRREPRO001 — close the original stdin driver replay gap

CPU-only max120s seed1701. THR001 actually completed before test access but its
recorded process argv was Python stdin. This versioned driver reconstructs exact
train-only quartile thresholds and representative TIME001 p90component rates,
requires byte-for-byte parity with both original registered output hashes. No
new thresholds, test access, fitting rule, config selection or primary changes.
New command is an engineering replay verification, not another scientific trial.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/rebuild_registration.py --config method/frozen_probe/configs/THRREPRO001.json --seed 1701 --run-id THRREPRO001_20261004_P003v1_s1701 --output-dir method/frozen_probe/runtime/THRREPRO001_20261004_P003v1_s1701
```
