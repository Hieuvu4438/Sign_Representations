# REPRO002 — add structural diagnostics and exact train-registration replay

CPU-only120s, no test scoring. Successor to immutable REPRO001, index17completed
runs, including THRREPRO001 byte-parity driver, COVTEST0013oraclechecks, COVER001
source-only exposure and both retained figure exports. Source/output/config hashes
and commands/environment verified. Live CONFIRM001 remains unreconciled, excluded
from completed evidence/cost. Primary P003 sources/configs remain unchanged.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/build_reproduction_index.py --config method/frozen_probe/configs/REPRO002.json --seed 1701 --run-id REPRO002_20261004_P003v1_s1701 --output-dir method/frozen_probe/runtime/REPRO002_20261004_P003v1_s1701
```
