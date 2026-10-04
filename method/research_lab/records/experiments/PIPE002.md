# PIPE002 — resource and CLI gate regression

Registered before execution2026-10-04 local, H001/P001_v3 software only.
New extraction/parity/baseline CLIs are checked for argument parsing and
refusing missing gates on CPU. No CUDA initialization or resource waiting;
pure reserve-boundary fixture checks20% reserve logic. Include all existing
26 meaningful tests. Synthetic fixtures only, no corpus features/metrics.

Config PIPE002.json, seed1337,≤60s, one CPU/BLAS thread,0 GPUh/HPO.
Snapshots include all new Python dependencies; no skips permitted. Output
method/frozen_probe/runtime/PIPE002_20261004_P001v3_s1337, exclusive.
Exact command registered before execution:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/PIPE002.json --seed 1337 --run-id PIPE002_20261004_P001v3_s1337 --output-dir method/frozen_probe/runtime/PIPE002_20261004_P001v3_s1337
```
Success qualifies software interfaces only; GPU UUID/inference/VRAM/runtimes
are still unverified until GPU001. No job launched at registration.

Observed completed exit0;30 checks,0 skips,0.3674504s CPU wall,0 GPUh/HPO.
CLI help/refusal guards and reserve logic checked with CUDA hidden/mock device
observations; actual GPU device UUID/cap/parity remains pending.
