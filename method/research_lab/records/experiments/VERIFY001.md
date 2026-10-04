# VERIFY001 — registered CPU gate for the completed real cache

Registered before implementation and execution, H001/P001_v4. Question: does
CACHE005 contain exactly the locked 1691 sources and 61339 native windows with
valid provenance and every shard intact? This is a pipeline prerequisite,
not a performance experiment. Seed1701, CPU/BLAS1thread, max120s, zero GPU/HPO.

Use existing strict CacheReader to check all hashes, source/split/class/signer
bindings, finite arrays, dtypes/shapes and exact native centers. Check terminal
extraction status/report/config, source snapshots, environment and agreement
between RAM and durable manifests. Any mismatch stops prediction launch.
No source exclusions, partial-cache acceptance, automatic retry or test access.

Config VERIFY001.json; planned run VERIFY001_20261004_P001v4_s1701,
exclusive output method/frozen_probe/runtime/VERIFY001_20261004_P001v4_s1701.
Command will be registered after the CLI exists and before execution. It must
wait for terminal successful CACHE005; no completed validation is claimed here.

Also correct run_baseline.py metadata to derive scope from config protocol:
BASE002 uses P001_v4, while its old literal says P001_v3. This affects reporting
only; pooling, heads, seeds, metrics and frozen extraction code are unchanged.
Existing ALLOC00140 checks cover those consumers; AST compilation suffices for
the literal change. Active CACHE005 imports neither new validation CLI nor
run_baseline.py; its original source snapshot remains immutable.

CLI now exists; AST and --help pass. Exact command registered for execution
**only after CACHE005 terminal success**, not executed yet:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/verify_cache.py --config method/frozen_probe/configs/VERIFY001.json --seed 1701 --run-id VERIFY001_20261004_P001v4_s1701 --output-dir method/frozen_probe/runtime/VERIFY001_20261004_P001v4_s1701 --extraction-run method/frozen_probe/runtime/CACHE005_20261004_P001v4_s1701
```

## Observed completion

Completed, exit0; wall0.470857s; GPU allowance0.000000000h.
Full1691 sources/61339 windows, no exclusions/test access.
Evidence [VERIFY001_report.json](VERIFY001_report.json), [VERIFY001_status.json](VERIFY001_status.json).
