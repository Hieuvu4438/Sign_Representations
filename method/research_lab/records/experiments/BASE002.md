# BASE002 — P001_v4 first failure pilot, pending real cache gate

Registered2026-10-04 local before launch. H001/P001_v4 BASE002.json seed1337,
same [BASE001](BASE001.md) statistical/control/head/label-budget plan. Only
protocol version/suballocation changes; no corpus performance/test feedback.
108 nine-control×three-seed×four-budget outputs, all378 dev IDs/100 classes,
alpha1, HPO0, CPU/BLAS1thread≤3600s. Pending
BASE002_20261004_P001v4_s1337 output method/frozen_probe/runtime/BASE002_20261004_P001v4_s1337.

Requires completed verified CACHE005 and Protocol Auditor VALIDATED gate bound
to this config/current real cache. No gate exists or run launched. Preserve
rankings/full schema/projection/pooling hashes, metadata/results/status. Analyze
with ANALYZE002.json, participant/bootstrap/selection limitations unchanged.
Exact command deferred until actual cache and gate exist.

## Launch gate satisfied before predictions

CACHE005 completed; VERIFY001 checks all1691 shards/61339 native windows.
P001_v4_launch_gate.json is VALIDATED for this adapted failure pilot only,
bound to actual cache/config hashes. Current core consumer hashes match
ALLOC00140-check snapshot; baseline scope literal change registered in VERIFY001.
Exact command registered before execution:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/run_baseline.py --config method/frozen_probe/configs/BASE002.json --seed 1337 --run-id BASE002_20261004_P001v4_s1337 --output-dir method/frozen_probe/runtime/BASE002_20261004_P001v4_s1337 --cache-manifest /dev/shm/signrep_frozen_probe_8bfd9b55453cd54f1d7b/manifest.json --launch-gate method/research_lab/records/protocols/P001_v4_launch_gate.json
```

## Observed completion

Completed exit0, wall5.493169s,0GPUh/HPO, test_access none.
Evidence [BASE002_status.json](BASE002_status.json).
All108 complete378×100 rankings retained in original runtime; lightweight
[summary](BASE002_summary.json) contains every metric and output hashes.
