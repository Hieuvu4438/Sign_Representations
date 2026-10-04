# SIZE003 — include dev-first native extraction plan

Same selectedsources/nearheuristic/rates/projection asSIZE002. Add all661native-source extraction sizing, no pooling/search cost for rawcache; this enables fixed5method qualitypilot on383dev before freshheldouttest. No newtest/HPO/GPU. Preserve SIZE002 direct-gallery planning evidence; originalSIZE001failurebound. CPU120s.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/audit_replication_feasibility.py --config method/frozen_probe/configs/SIZE003.json --seed 1704 --run-id SIZE003_20261004_P004v0_s1704 --output-dir method/frozen_probe/runtime/SIZE003_20261004_P004v0_s1704
```
