# COVER001 — structural frame/time coverage on full fixed source audits

A001 auxiliary registered after test metadata but before any quality score read.
CPU120s/seed1701, all2908 source audits, four actual selection recipes, no fitting
or scoring. Three coverage oracles required before real structural output. Source
SHA-native policy selection unchanged; original P003 screen unaffected.

Commands:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/COVTEST001.json --seed 1337 --run-id COVTEST001_20261004_A001_s1337 --output-dir method/frozen_probe/runtime/COVTEST001_20261004_A001_s1337
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/analyze_input_coverage.py --config method/frozen_probe/configs/COVER001.json --seed 1701 --run-id COVER001_20261004_A001_s1701 --output-dir method/frozen_probe/runtime/COVER001_20261004_A001_s1701
```

Full raw structural rows and summaries retained; union exposure is not quality,
linguistic ground truth, causality, novelty or a new primary confirmatory claim.
