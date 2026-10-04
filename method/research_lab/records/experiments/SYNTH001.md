# SYNTH001 — frozen-result presentation

Registered CPU-only presentation after CONFAN002. Preserve all existing primary metrics, all four timing methods, all participant/slice denominators and all twenty predeclared cases. Map IDs to exact gloss labels; no visual/linguistic interpretation. No new inference or statistical selection.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=method/frozen_probe/runtime/mpl_config python3 method/frozen_probe/scripts/report_confirmatory.py --config method/frozen_probe/configs/SYNTH001.json --seed 1701 --run-id SYNTH001_20261004_P003v1_s1701 --output-dir method/frozen_probe/runtime/SYNTH001_20261004_P003v1_s1701
```
