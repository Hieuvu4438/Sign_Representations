# FACTEST001 — factorial arithmetic/axes/coverage oracles

Three necessary synthetic oracles: hand-computed distinct query/gallery effects and interaction with participant denominators; incomplete cell/fullclass permutation rejection; wrongidentity/signer/split/method/seed rejection. No real mixed outcomes. CPU120s.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/FACTEST001.json --seed 1337 --run-id FACTEST001_20261004_A002_s1337 --output-dir method/frozen_probe/runtime/FACTEST001_20261004_A002_s1337
```
