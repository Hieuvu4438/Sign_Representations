# PIPE001 — CPU cache-to-prediction contract fixture

Registered before implementation/execution, 2026-10-04 local. H001/P001_v3.
Scope: synthetic software validation only; no corpus embeddings, recognition
measurement, test access, tuning or GPU use.

Question: does the cache consumer reject identity/coverage/timeline corruption,
and do all six ridge plus three class-max controls produce complete ID-joined
rankings with identical train support access? Fixture has three classes,
separate train/dev participants, deterministic artificial window vectors.
Perfect synthetic separation is an oracle, not an empirical effect.

Cache contract binds cohort, video audit, checkpoint/upstream/preprocessing/
sampling/layer/context hashes; shard completion is atomic and content checked.
No filename-based reuse, partial coverage, silent exclusion, zero filling,
pickle loading or corpus normalization. Existing outputs are read-only;
resume means verify an already completed cache, never mutate a terminal run.

Tests: read/write round trip; wrong identity; missing/extra/duplicate IDs;
unsafe shard path; altered SHA; incorrect dtype/native timeline; train/dev
support substitution; support count/class mismatch; complete rankings for
every method; deterministic support-order handling. Existing contract tests
remain required because the data consumer uses pooling/probes/scorer.

Budget: one single-thread CPU software job, 60 seconds, 0 GPU-hours, 0 HPO,
temporary fixture bytes <1MiB. Config PIPE001.json. Run/output IDs and exact
command will be recorded once the CLI exists; exclusive output directory.
Keep snapshots/config/environment/log/report/status; no completed-run edits.
Pass iff all fixtures pass without skips; failures are software incidents,
not H001 refutation. Follow bounded diagnosed retries if necessary.

Run PIPE001_20261004_P001v3_s1337; output
`method/frozen_probe/runtime/PIPE001_20261004_P001v3_s1337`.
Exact command (registered before launch):

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/PIPE001.json --seed 1337 --run-id PIPE001_20261004_P001v3_s1337 --output-dir method/frozen_probe/runtime/PIPE001_20261004_P001v3_s1337
```

Observed completed exit0;26 checks,0 skips,0.0637633s CPU wall,0 GPUh/HPO.
Status/report/test log copied to records; original runtime snapshots retained.
Synthetic three-class fixture gives expected perfect separation for controls;
that known oracle is not an ASL Citizen performance result.
