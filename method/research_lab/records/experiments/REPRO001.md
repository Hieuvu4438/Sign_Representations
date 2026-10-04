# REPRO001 — rebuildable completed-run provenance index

CPU-only, max120s, seed1701, no test scoring/metric read. Eleven actual completed
scientific/pilot/validation runs; validate recorded config/output/source snapshot
hashes, preserve original protocol/warmup erratum and index exact historical argv.
AllGPU attempts included for cost, live CONFIRM001 explicitly unreconciled. Index
does not authorize replay into immutable completed directories or prove science.
New reporting code is outside locked inference/scorer; no changes to their
scientific config or source. Source snapshots included in this CPU run.

Registration incident: first config-discovery command included a launch.log in
its CACHE005 glob and failed its uniqueness assertion before writing any config.
A subsequent invocation therefore failed on missing config before RunContext;
no output/run/GPU was allocated. Directory-only discovery corrects registration,
with no scientific run invalidation or changes to completed evidence.

Command:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/build_reproduction_index.py --config method/frozen_probe/configs/REPRO001.json --seed 1701 --run-id REPRO001_20261004_P003v1_s1701 --output-dir method/frozen_probe/runtime/REPRO001_20261004_P003v1_s1701
```
