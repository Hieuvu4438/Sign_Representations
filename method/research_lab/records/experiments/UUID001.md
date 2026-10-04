# UUID001 — registered diagnosed GPU UUID format fix

Registered2026-10-04 local before code mutation/execution. H001/P001_v3
software only,≤60s CPU/BLAS1thread,0 actualGPUh/HPO. GPU003 failed before model
load because nvidia-smi returns GPU-prefixed UUID while PyTorch _CUuuid string
omits that prefix, established by DEV001 live device-property report.
Normalize both via standard UUID parser; validate full128-bit UUID and retain
unique matching, never infer granted device by index or bypass scheduler.

Fixture: actual DEV001 identifiers match with/without GPU- prefix/case, other
valid UUID does not match, invalid/malformed input rejected. Required existing
38 regressions remain. Config UUID001.json seed1337; exclusive output
method/frozen_probe/runtime/UUID001_20261004_P001v3_s1337. Keep snapshots/status/
config/report/log. No model/parity/feature measurement; GPU retry separately
registered after this test. Previous snapshots/failure immutable.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/UUID001.json --seed 1337 --run-id UUID001_20261004_P001v3_s1337 --output-dir method/frozen_probe/runtime/UUID001_20261004_P001v3_s1337
```

## Observed terminal result

Status **completed**, exit0; wall0.439907s;
conservative actual GPU allowance0.000000000s.
Durable evidence [UUID001_status.json](UUID001_status.json), original config/code/environment/log retained.
