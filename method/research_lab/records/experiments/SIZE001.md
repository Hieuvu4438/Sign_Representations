# SIZE001 — bounded near-screen and conditional sizing

All661train/dev sources, original fixedfingerprint thresholds/allpairs/no exclusions; use knownoldtest/devaggregate ratios to projectnewtest totals from newdevmetadata only. Original ratesp90/safety1.5/setup60 unchanged; conservative train-onlygallery extraction+fourqueryrecipe planning. No freshtestCSV/pixels/outcomes, no GPU, no launchgate. CPU120s.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/audit_replication_feasibility.py --config method/frozen_probe/configs/SIZE001.json --seed 1704 --run-id SIZE001_20261004_P004v0_s1704 --output-dir method/frozen_probe/runtime/SIZE001_20261004_P004v0_s1704
```
