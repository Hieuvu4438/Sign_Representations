# COVFIG001 — export source-only coverage curves

CPU-only max120s seed1701, completed COVER001 source rows/outputs verified by
SHA. EmpiricalCDFs for all fixed train/dev/test clips under three same-count
sparse policies, figure data CSV retained, SVG/PDF/PNG. No prediction/score
read, inferred quality or primary P003 changes. Standard installed matplotlib,
Agg renderer, own runtime config cache, no installs or external uploads.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=method/frozen_probe/runtime/mpl_config python3 method/frozen_probe/scripts/plot_input_coverage.py --config method/frozen_probe/configs/COVFIG001.json --seed 1701 --run-id COVFIG001_20261004_A001_s1701 --output-dir method/frozen_probe/runtime/COVFIG001_20261004_A001_s1701
```
