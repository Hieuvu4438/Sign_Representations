# PREDVERIFY001 — independently regenerate every held-out ranking before scoring

Engineering validation added while CONFIRM001 runs; no changes to the original
P003 scientific config/scorer/selection. Prerequisite: complete terminal prediction
status and all output/config hashes. CPU120s/seed1701, no GPU, no task-score compute.
Verify exact test schema/ID-label-signer bindings, every actual recipe/nativewindow
selection/count against source audit/SHA-only policy, pooledNPZ finitefloat64/L2
shape1217×768/methods, payloadSHA, immutabletrain-onlycachedgalleries and all15
fullrankmatrices independently regenerated from actualpooledqueries. No labels
used for evaluation before the frozen separate analysis. Failure is an integrity
incident; never reinterpret failure as a scientific negative or silently repair.

Three oracle checks for source/recipe completeness, mutated timing/schema/PTS and
pooleddtype/dimension/L2. Run real verification only after successful prediction:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/PREDTEST001.json --seed 1337 --run-id PREDTEST001_20261004_P003v1_s1337 --output-dir method/frozen_probe/runtime/PREDTEST001_20261004_P003v1_s1337
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/verify_confirmatory.py --config method/frozen_probe/configs/PREDVERIFY001.json --seed 1701 --run-id PREDVERIFY001_20261004_P003v1_s1701 --output-dir method/frozen_probe/runtime/PREDVERIFY001_20261004_P003v1_s1701
```

Additionalchecker is post-execution QA, outside the frozen scientific method.
Original scientific source SHA gate remains intact. No test-driven fit or threshold.
