# COVFIG002 — expose the complete-coverage ECDF jump

Presentation-only successor to completed COVFIG001. Initial image visually
checked: ECDF jump at x=1 overlapped the right spine. Add2%axis margin and fixed
0..1ticks, same COVER001 data/curves/denominators; no new analysis or metrics.
COVFIG001 remains immutable/valid source-data evidence. CPU120s seed1701.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=method/frozen_probe/runtime/mpl_config python3 method/frozen_probe/scripts/plot_input_coverage.py --config method/frozen_probe/configs/COVFIG002.json --seed 1701 --run-id COVFIG002_20261004_A001_s1701 --output-dir method/frozen_probe/runtime/COVFIG002_20261004_A001_s1701
```
