# Run index — actual executions

Every attempt retained; CPU diagnostics and granted GPU prerequisites. No stochastic training/HPO trials.
Commands invoked from `/home/haipd/Sign_Representations`.
Status JSONs store observed start/end, host/PID, script/output hashes and exits.
Runtime source snapshots preserved and hashes checked for all attempts.

| Run | H/P | Status / exit | Job wall seconds | Evidence |
| --- | --- | --- | --- | --- |
| PREP001_20261003 | inventory / no scientific protocol | completed /0 | 1.3932 | PREP001_status.json; inventory/PREP001_inventory.json |
| VAL001_20261003 | H001/P001_v1 diagnostic | completed /0 | .1964 | VAL001_status.json; protocols/P001_v1_parity.json |
| LOAD001_20261003 | H001/H002/P001_v1 | failed /1, literal shape product parsing | .6976 | LOAD001_20261003_status.json |
| LOAD001_20261003_r1 | same | failed /1, timm import API | 1.2402 | LOAD001_20261003_r1_status.json |
| LOAD001_20261003_r2 | same | completed /0, strict load | 1.8028 | LOAD001_20261003_r2_status.json; LOAD001_coverage.json |
| DATA001_20261003 | H001/H002/P001_v1 diagnostic | completed /0; VFR flag invalidated | .4351 | DATA001_status.json; inventory/DATA001_video_audit.json |
| DATA001_20261003_r1 | same | completed /0, corrected jitter | .3955 | DATA001_r1_status.json; inventory/DATA001_r1_video_audit.json |
| FEAT001_20261003_P001v1_s1701 | H001/H002/P001_v1 diagnostic | completed /0, CPU preprocessing/features | 85.0751 | FEAT001_status.json; FEAT001_parity.json; FEAT001_environment.json |
| REL001_20261003 | H001/P001 prerequisite | completed /0, official metadata equality | 3.7050 | REL001_status.json; inventory/REL001_release.json |
| COHORT001_20261003_s1701 | H001/P001_v2 draft prerequisite | completed /0, candidate cohort | .9129 | COHORT001_status.json; inventory/COHORT001_cohort.json |
| VID002_20261003_P001v2_s1701 | H001/P001_v2 prerequisite | completed /0, all cohort pixel/timeline checks | 283.0515 | VID002_status.json; inventory/VID002_video_audit.json |
| NEAR001_20261004_P001v2_s1701 | H001/P001_v2 prerequisite | completed /0, heuristic pair screen | 1.0183 | NEAR001_status.json; inventory/NEAR001_screen.json |
| POOL001_20261004_P001v3_s1337 | H001/P001_v3 draft software | completed /0,20 checks | .0445 | POOL001_status.json; POOL001_report.json |
| POOL002_20261004_P001v3_s1337 | same, expanded controls | completed /0,21 checks | .0355 | POOL002_status.json; POOL002_report.json |
| PIPE001_20261004_P001v3_s1337 | H001/P001_v3 draft software | completed /0,26 checks | .0638 | PIPE001_status.json; PIPE001_report.json |
| PIPE002_20261004_P001v3_s1337 | same, resource/CLI checks | completed /0,30 checks | .3675 | PIPE002_status.json; PIPE002_report.json |
| AN001_20261004_P001v3_s1337 | same, analysis/uncertainty checks | completed /0,33 checks | .4629 | AN001_status.json; AN001_report.json |
| BUD001_20261004_P001v3_s1337 | same, cumulative budget/status | completed /0,37 checks | .4933 | BUD001_status.json; BUD001_report.json |
| BUD002_20261004_P001v3_s1337 | same, remaining walltime | completed /0,38 checks | .4649 | BUD002_status.json; BUD002_report.json |

Each audit uses new exclusive output dir, never overwrites a completed run.
Current job status is recorded in the latest dated update below. Failed jobs are engineering findings, not scientific
negative results. Runtime original logs/snapshots under `method/*/runtime/`
are ignored; small evidence above is tracked-capable and persists outside RAM.

## Exact commands that ran

```bash
python3 method/research_lab/scripts/inventory.py --run-id PREP001_20261003 --output-dir method/research_lab/runtime/PREP001_20261003
python3 method/frozen_probe/scripts/validate_scorer.py --run-id VAL001_20261003 --output-dir method/frozen_probe/runtime/VAL001_20261003
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 method/frozen_probe/scripts/audit_signrep.py --run-id LOAD001_20261003 --output-dir method/frozen_probe/runtime/LOAD001_20261003
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 method/frozen_probe/scripts/audit_signrep.py --run-id LOAD001_20261003_r1 --output-dir method/frozen_probe/runtime/LOAD001_20261003_r1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/haipd/miniconda3/envs/seds/bin/python method/frozen_probe/scripts/audit_signrep.py --run-id LOAD001_20261003_r2 --output-dir method/frozen_probe/runtime/LOAD001_20261003_r2
python3 method/frozen_probe/scripts/audit_video.py --run-id DATA001_20261003 --output-dir method/frozen_probe/runtime/DATA001_20261003
python3 method/frozen_probe/scripts/audit_video.py --run-id DATA001_20261003_r1 --output-dir method/frozen_probe/runtime/DATA001_20261003_r1
PYTHONPATH=method/frozen_probe python3 -m unittest discover -s method/frozen_probe/tests -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-signrep/bin/python method/frozen_probe/scripts/validate_features.py --config method/frozen_probe/configs/FEAT001.json --seed 1701 --run-id FEAT001_20261003_P001v1_s1701 --output-dir method/frozen_probe/runtime/FEAT001_20261003_P001v1_s1701
python3 method/research_lab/scripts/audit_release.py --run-id REL001_20261003 --output-dir method/research_lab/runtime/REL001_20261003
python3 method/frozen_probe/scripts/build_cohort.py --config method/frozen_probe/configs/COHORT001.json --seed 1701 --run-id COHORT001_20261003_s1701 --output-dir method/frozen_probe/runtime/COHORT001_20261003_s1701
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-signrep/bin/python method/frozen_probe/scripts/audit_cohort_video.py --config method/frozen_probe/configs/VID002.json --seed 1701 --run-id VID002_20261003_P001v2_s1701 --output-dir method/frozen_probe/runtime/VID002_20261003_P001v2_s1701
PYTHONDONTWRITEBYTECODE=1 python3 method/frozen_probe/scripts/screen_duplicates.py --input method/frozen_probe/runtime/VID002_20261003_P001v2_s1701/video_audit.json --seed 1701 --run-id NEAR001_20261004_P001v2_s1701 --output-dir method/frozen_probe/runtime/NEAR001_20261004_P001v2_s1701
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/POOL001.json --seed 1337 --run-id POOL001_20261004_P001v3_s1337 --output-dir method/frozen_probe/runtime/POOL001_20261004_P001v3_s1337
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/POOL002.json --seed 1337 --run-id POOL002_20261004_P001v3_s1337 --output-dir method/frozen_probe/runtime/POOL002_20261004_P001v3_s1337
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/PIPE001.json --seed 1337 --run-id PIPE001_20261004_P001v3_s1337 --output-dir method/frozen_probe/runtime/PIPE001_20261004_P001v3_s1337
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/PIPE002.json --seed 1337 --run-id PIPE002_20261004_P001v3_s1337 --output-dir method/frozen_probe/runtime/PIPE002_20261004_P001v3_s1337
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/AN001.json --seed 1337 --run-id AN001_20261004_P001v3_s1337 --output-dir method/frozen_probe/runtime/AN001_20261004_P001v3_s1337
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/BUD001.json --seed 1337 --run-id BUD001_20261004_P001v3_s1337 --output-dir method/frozen_probe/runtime/BUD001_20261004_P001v3_s1337
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/BUD002.json --seed 1337 --run-id BUD002_20261004_P001v3_s1337 --output-dir method/frozen_probe/runtime/BUD002_20261004_P001v3_s1337
```

Base Python3.13 torch2.11/timm0.4.5; seds Python3.10 torch2.3.1+cu121/timm1.0.29.
No package install/shared-env write. Code base Git SHA in PREP001; all new code
untracked/dirty, hashes/snapshots substitute commit provenance for audit runs.
GPU-hours 0/4; run walltime ~6.16 s excludes browsing, coding, metadata discovery
commands and tests. Not a full session compute-cost measurement.

Update1: FEAT uses existing `.venv-signrep` (versions in environment JSON),
all new jobs CPU; cumulative listed job walltime~95.9s. Separate source lookup
fetched bounded official GitHub release API JSON; digest matches local asset.
Test logs FEAT001_tests.txt9 checks, COHORT001_tests.txt11 checks. No HPO/GPU
usage. All completed/failed output directories retained; no active process.

Update2: four more completed CPU jobs284.1498s combined; latest21 checks pass.
VID002 data duration/cost is pipeline audit, not representation inference cost.
NEAR screen/POOL checks not performance evidence. No live owned process or
GPU job, no HPO. Current protocol P001_v3 still BLOCKED for science launch.

Update3: three more CPU software jobs.8940974s; latest33 checks pass.
GPU001/CACHE001/BASE001 are registered **DEFERRED**, not executions; no owned
GPU process/cache/recognition predictions. Analysis CLI also pending real
predictions. GPU-hours remain0/4, HPO0. All evidence copies/snapshots preserved.

Update4: BUD001/002 CPU.9581588s combined,38 checks pass. GPU002/CACHE002
supersede unexecuted v1 configs; all completed snapshots immutable. Printed
mock GPU statuses in lifecycle fixtures are not jobs/spend; real ledger0GPUh.

## Update5: GPU execution and live extraction

2026-10-03T19:36:49.616771+00:00 UTC. User confirms available GPU.

| Run | H/P | Status / exit | Job wall seconds | Evidence |
| --- | --- | --- | --- | --- |
| GPU003_20261004_P001v3_s1701 | H001/P001_v3 | failed /1 | 1.430090 | GPU003_status.json; [GPU003 card](GPU003.md) |
| DEV001_20261004_P001v3_s1701 | H001/P001_v3 | completed /0 | 0.681904 | DEV001_status.json; [DEV001 card](DEV001.md) |
| UUID001_20261004_P001v3_s1337 | H001/P001_v3_draft | completed /0 | 0.439907 | UUID001_status.json; [UUID001 card](UUID001.md) |
| GPU004_20261004_P001v3_s1701 | H001/P001_v3 | completed /0 | 11.891743 | GPU004_status.json; [GPU004 card](GPU004.md) |
| ALLOC001_20261004_P001v4_s1337 | H001/P001_v4_draft | completed /0 | 0.535377 | ALLOC001_status.json; [ALLOC001 card](ALLOC001.md) |

GPU003 failed a UUID-prefix guard before encoder execution. DEV001 diagnosed
the actual Torch UUID representation; UUID001 fixes parsing. GPU004 retry1
passes all five-source/ten-window numerical comparisons. ALLOC001 passes40
software checks. Prior terminal GPU allowance totals10.745844s (.002985h);
CACHE005 active cost is pending and must be reconciled before a new GPU job.

CACHE005_20261004_P001v4_s1701 is running, PID1614077, exec session21036,
full1691-source/61339-window plan. Only one owned GPU job. Resume same handle;
no recognition predictions or completed cache yet. [CACHE005 card](CACHE005.md)
records the exact command and scratch path is in STATE. P001_v4 H0012h lies
inside unchanged4h program. Shared-load cost is provisional.

Exact commands from actual GPU/software status records (history):

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/haipd/Sign_Representations/.venv-signrep/bin/python method/frozen_probe/scripts/gpu_smoke.py --config method/frozen_probe/configs/GPU003.json --seed 1701 --run-id GPU003_20261004_P001v3_s1701 --output-dir method/frozen_probe/runtime/GPU003_20261004_P001v3_s1701
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/haipd/Sign_Representations/.venv-signrep/bin/python method/frozen_probe/scripts/inspect_device.py --config method/frozen_probe/configs/DEV001.json --seed 1701 --run-id DEV001_20261004_P001v3_s1701 --output-dir method/frozen_probe/runtime/DEV001_20261004_P001v3_s1701
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/UUID001.json --seed 1337 --run-id UUID001_20261004_P001v3_s1337 --output-dir method/frozen_probe/runtime/UUID001_20261004_P001v3_s1337
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/haipd/Sign_Representations/.venv-signrep/bin/python method/frozen_probe/scripts/gpu_smoke.py --config method/frozen_probe/configs/GPU004.json --seed 1701 --run-id GPU004_20261004_P001v3_s1701 --output-dir method/frozen_probe/runtime/GPU004_20261004_P001v3_s1701
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/ALLOC001.json --seed 1337 --run-id ALLOC001_20261004_P001v4_s1337 --output-dir method/frozen_probe/runtime/ALLOC001_20261004_P001v4_s1337
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/haipd/Sign_Representations/.venv-signrep/bin/python method/frozen_probe/scripts/extract_cache.py --config method/frozen_probe/configs/CACHE005.json --seed 1701 --run-id CACHE005_20261004_P001v4_s1701 --output-dir method/frozen_probe/runtime/CACHE005_20261004_P001v4_s1701 --smoke-run method/frozen_probe/runtime/GPU004_20261004_P001v3_s1701
```

VERIFY001 is registered **pending**, no execution or cache-completion claim.

## Update6: real cache, integrity gate and first failure pilot completed

2026-10-03T20:17:39.065530+00:00 UTC. No owned GPU/CPU process remains; CACHE005 session21036 and
BASE002 session11872 terminal exit0.

| Run | H/P | Status / exit | Job wall seconds | Evidence |
| --- | --- | --- | --- | --- |
| CACHE005_20261004_P001v4_s1701 | H001/P001_v4 | completed /0 | 2477.870906 | [CACHE005 card](CACHE005.md), CACHE005_status.json |
| VERIFY001_20261004_P001v4_s1701 | H001/P001_v4 | completed /0 | 0.470857 | [VERIFY001 card](VERIFY001.md), VERIFY001_status.json |
| BASE002_20261004_P001v4_s1337 | H001/P001_v4 | completed /0 | 5.493169 | [BASE002 card](BASE002.md), BASE002_status.json |
| ANALYZE002_20261004_P001v4_s1701 | H001/P001_v4 | completed /0 | 0.169760 | [ANALYZE002 card](ANALYZE002.md), ANALYZE002_status.json |

All actual GPU allowance2487.344571s=.690929GPUh of4h, H0012h remainder
4712.655429s. HPO0/6, no test access. Cache190727048payload bytes in RAM;
complete source/window coverage and every array/hash/native center verified.
All108378×100 rankings retained, metrics recomputed. D007 stops scaling the
failed ordered-pooling recipe, H002 registration next. Selected dev participants
**five**, corrected by explicit protocol erratum; issued gate/snapshots preserved.

New actual commands (completed history):

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/verify_cache.py --config method/frozen_probe/configs/VERIFY001.json --seed 1701 --run-id VERIFY001_20261004_P001v4_s1701 --output-dir method/frozen_probe/runtime/VERIFY001_20261004_P001v4_s1701 --extraction-run method/frozen_probe/runtime/CACHE005_20261004_P001v4_s1701
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/run_baseline.py --config method/frozen_probe/configs/BASE002.json --seed 1337 --run-id BASE002_20261004_P001v4_s1337 --output-dir method/frozen_probe/runtime/BASE002_20261004_P001v4_s1337 --cache-manifest /dev/shm/signrep_frozen_probe_8bfd9b55453cd54f1d7b/manifest.json --launch-gate method/research_lab/records/protocols/P001_v4_launch_gate.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/analyze_baseline.py --config method/frozen_probe/configs/ANALYZE002.json --seed 1701 --run-id ANALYZE002_20261004_P001v4_s1701 --output-dir method/frozen_probe/runtime/ANALYZE002_20261004_P001v4_s1701 --prediction-run method/frozen_probe/runtime/BASE002_20261004_P001v4_s1337
```

## Update7: H002 quality and actual timing completed

2026-10-03T20:54:11.573945+00:00 UTC. No owned process remains; QUAL001 session55985 and TIME001
session30115 terminal exit0.

| Run | H/P | Status / exit | Job wall seconds | Evidence |
| --- | --- | --- | --- | --- |
| WIN001_20261004_P002v1_s1337 | H002/P002_v1 | completed /0 | 0.448695 | [WIN001 card](WIN001.md) |
| QUAL001_20261004_P002v1_s1701 | H002/P002_v1 | completed /0 | 2.317680 | [QUAL001 card](QUAL001.md) |
| TIME001_20261004_P002v1_s1701 | H002/P002_v1 | completed /0 | 59.397070 | [TIME001 card](TIME001.md) |
| REPORT001_20261004_P002v1_s1701 | H002/P002_v1 | completed /0 | 0.016999 | [REPORT001 card](REPORT001.md) |

Total actual GPU2545.048626s=0.706957952h/4h; H002 actual
57.704055s/3600s. Six policy trials consumed,0 gradient
HPO/training/test access. Uniform8 passes registered joint screen; D008 retains
analysis only because stride8 already appears in native SignRep ablations.
All24 rankings, seed metrics,105 raw timings and rebuilt table retained.

Completed exact commands (history):

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/WIN001.json --seed 1337 --run-id WIN001_20261004_P002v1_s1337 --output-dir method/frozen_probe/runtime/WIN001_20261004_P002v1_s1337
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/window_quality.py --config method/frozen_probe/configs/QUAL001.json --seed 1701 --run-id QUAL001_20261004_P002v1_s1701 --output-dir method/frozen_probe/runtime/QUAL001_20261004_P002v1_s1701 --cache-manifest /dev/shm/signrep_frozen_probe_8bfd9b55453cd54f1d7b/manifest.json --launch-gate method/research_lab/records/protocols/P002_v1_quality_gate.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/haipd/Sign_Representations/.venv-signrep/bin/python method/frozen_probe/scripts/window_timing.py --config method/frozen_probe/configs/TIME001.json --seed 1701 --run-id TIME001_20261004_P002v1_s1701 --output-dir method/frozen_probe/runtime/TIME001_20261004_P002v1_s1701 --cache-manifest /dev/shm/signrep_frozen_probe_8bfd9b55453cd54f1d7b/manifest.json --quality-run method/frozen_probe/runtime/QUAL001_20261004_P002v1_s1701 --launch-gate method/research_lab/records/protocols/P002_v1_timing_gate.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/summarize_window_budget.py --config method/frozen_probe/configs/REPORT001.json --seed 1701 --run-id REPORT001_20261004_P002v1_s1701 --output-dir method/frozen_probe/runtime/REPORT001_20261004_P002v1_s1701 --quality-run method/frozen_probe/runtime/QUAL001_20261004_P002v1_s1701 --timing-run method/frozen_probe/runtime/TIME001_20261004_P002v1_s1701
```

## Update8 — P003 pre-test registration

D009 reallocates unchanged4GPUh into1h/2h/1h. CONFTEST001 CPU49checks,0skips,
wall.533554s completed; THR001 CPU train-only thresholds/rates wall.422089s completed.
THR001 command Python stdin invokes snapshot confirmatory.train_thresholds with
THR001 config/immutable CACHE005 and rates from TIME001; exact formulas/rate
quantiles are recorded in P003/config/source snapshot. No test access there.
P003 immutable SHA freeze issued before PRETEST001, planned CPU full audit/cost
estimate. No confirmatory result/GPU job yet. See PRETEST001 card for actual command.

PRETEST001 completed147.330352s CPU; full1217query audit passes. Original freeze
precedes metadata by45.179009s. Warmup fixed source is dev, not train; code/config
registration erratum before test inference/scoring,50regression checks pass.
CONFIRM001 actual inference launch registered, estimate3588.438359s, config and
policy/scorer scientific plan unchanged. See PRETEST001/CONFIRM001 cards.

CONFIRM001 live GPU PID1887036/session65703/startUTC21:28:28.297846;
7098s deadline, actual selected inference. Only terminal cost is2545.048626s;
new live cost pending reconciliation. No confirmatory score yet. Poll same handle.


## Update9 — CPU reproduction/exposure artifacts during live CONFIRM001

REPRO001 completed.036570s; THRREPRO001 exact pre-test artifacts byte parity
completed.548757s; COVTEST0013manualunion/PTSchecks0skips completed.035419s;
COVER001 all2908sources/fourpolicies exposure completed.381107s;
COVFIG001 completed.816389s, COVFIG002 presentation margin correction.647956s,
figure_dataCSV unchangedSHA1230edf35ad4563c5b841810b797fff5dc61b1fb70361eb3d3ad3638b2032296;
REPRO00217completedrunhash/commandindex completed.051898s. All CPU-only; no new
policy/search/test scoring. A001 after-metadata/pre-outcome descriptive analysis,
original P003 screen unchanged. Source-only coverage testuniform.953717 vs
random.830741/.833046, no quality/causality interpretation. Paper dossier draft,
figure exports and replay index created; mainquality/cost/errors still pending.
CONFAN001 existing frozen command/card registered, not run until completed inputs.


## Update10 — H002 actual held-out confirmation completed

CONFIRM001 terminal2398.762307s wall/2397.640770s GPU, all1217queries/68295actual
encoderwindows/4868timingobservations. PREDTEST0013oracles.058932s CPU; PREDVERIFY001
3.973594s CPU independently regenerates15fullrankings/exactsource-recipes/pooled
payloadSHA5bfa8a0a027c1bd657651551a68cd7d40f4cac709ba87b167e2c853f45822a22.
CONFAN001 failed.150999s CPU on numpybool strictJSON, no metricartifact persisted;
explicitserializationerratum,57SERTEST001checks.406365s CPU, unchanged comparison
coercedPythonbool only. CONFAN002 diagnosedretry1completed.150851s CPU; sameoriginal
CONFAN001config, no new GPU/policy/seed/threshold. Originalerrors/snapshots retained.

DCGuniform.884978943/dense.886378665, loss.001399723, conditional11group95%CI
[-.004337290,.000968716]difference; everygallerylosswithin.02. Exact3.816549×
querywindow reduction, pairedmedianpipeline ratio.373602586, bothregisteredfloors
pass. All15metrics/slices/20seedednumericcases retained. D010continues scoped
analysis/synthesis, no new sampler or PAPER_READY. TotalownedterminalGPU4942.689396s
(1.372969277h) of4h, H002remaining4744.655175s. No ownedlivejobs/sessions.

## Update11 — terminal synthesis and reproduction (2026-10-04)

SYNTH001 CPU.615268s exported frozen results/all20cases; SYNTH002 CPU.474579s
clarified dense-activity margin label, identical case/data hashes. Rebuildable
SVG/PDF/PNG/CSV/report under paper/, no new science/inference/tuning. REPRO003
CPU.052367s verifies25completed run configs/output/source snapshots, historical
argv, both technical errata and failed CONFAN001 incident binding; terminal GPU
4942.689396s, no live owned attempt. Dossier/gates/H002/claims updated to D010.

## Update12 — attribution, primary-source refresh and prospective feasibility

FACTEST0013manualoracles pass; FACTOR001post-outcome all16pairs/48ranks/exact12diagonal
parity completeCPU9.436802s, allseed/participant negatives retained. LIT001bounded
supplementcapfailure retained; LIT002officialmain/supp retrieval complete31.787322s,
pagepreflightUNAVAILABLE/no installs; S03F/L04/L05 selectedtexts and matrix/dated
noveltyaudit updated. D011continues scopedanalysis, no sampler/causal novelty.
COHORT002278train/383dev disjoint vocabulary inventory completeCPU.456317s;
VID003full661nativevideoaudit complete80.882380s. SIZE001schemafailure retained;
SIZE002/.231428s andSIZE003/.175270s completeboundedheuristic/sizing. D012queues
P004fixedpolicydevfirst registration;4710.633481sconditionalprojected total versus
4744.655175sH002remaining,34.02sestimate slack; not a launchgate/measurement.

## Update13 — P004 prospective dev registration and execution

REPTEST001 four manual integration oracles pass CPU.079150s; REPREG001 gate
CPU.109202s issued23:33:32.194846UTC before new dev outcomes. CACHE006 one owned
GPU0 native extraction live; all661 fixed sources/24001 windows, no test access,
1500s ceiling/8GiB allocator cap. Running GPU_hours0 is unfinalized spend, not free
compute. Separate future held-out wrappers/protocol prepared with unchanged
scorer/timing/core and explicit budget1; REPCTEST001 two mutation/freeze oracles
CPU.073878s pass, but test freeze/access awaits positive fixed dev pilot. Original
P003 binding SHA checks pass; no new HPO/policy/seed/class selection.

## Update14 — positive dev pilot; fresh test native-domain failure

CACHE006terminalGPU703.502506s/wall704.989919s,661sources/24001windows/74115088B.
REPQUAL001CPU1.227352s fixed383query/100class/one-example/15rankdevscreen passes
(D013). REPFREEZE001CPU.148761s freezesbefore newselectedtest access. REPPRE001
failed51.602858s at10frame SPEND source, noGPU/testscore. CONTRACT001all1203source
censusCPU146.545589s confirmsone native-contract violation/0boundednearcandidates;
SHORTREL001CPU4.446711s officialmemberbyteequal/10.76MBranges. D014stopsP004test
scaling, no exclusion/padding/policychange; originalP003 unchanged. CurrentGPU
program5646.191902s, H002remaining4041.152670s; no ownedlivejobs. NextH003inputaudit.

## Update15 — H003 first raw-input audit

STREAM001 newstream_reliability namespace terminalCPU.028632s/0GPU, four fixed
train/dev source schemas/hashes/native counts and pinned upstream interfaces.
D015continues input/provenance/environment audit, model measurementheld:133point
coordinates and uncalibrated scores are not ready SHuBERT crop embeddings/body
features or naturalfailurelabels. Existing DINOface/hand assets found, no download
needed before provenance audit. ProgramGPUunchanged; no ownedlivejobs.

## Update: D016 H003 actual prerequisites and GPU software smoke

Eight actual attempts, seven completed/one HTTP404 retained. CPU/model/source audits are not scientific pilots.

| Record alias | Status | Wall seconds | Raw status GPU seconds |
| --- | --- | ---: | ---: |
| [ASSET001](ASSET001.md) | failed | 0.318064810 | 0.000000000 |
| [ASSET002](ASSET002.md) | completed | 1.190199684 | 0.000000000 |
| [SHUBERT_LOAD001](SHUBERT_LOAD001.md) | completed | 2.690883761 | 0.000000000 |
| [DINOSRC001](DINOSRC001.md) | completed | 4.625458309 | 0.000000000 |
| [STREAMDINO001](STREAMDINO001.md) | completed | 1.785063556 | 0.000000000 |
| [LANDMARK001](LANDMARK001.md) | completed | 43.395969011 | 0.000000000 |
| [GLAUDIT001](GLAUDIT001.md) | completed | 0.004626871 | 0.000000000 |
| [STREAMGPU001](STREAMGPU001.md) | completed | 3.325681433 | 2.900944370 |

LANDMARK001 raw zero is corrected once by GLAUDIT001: full43.395969011s for actual NVIDIA OpenGL context despite CPU delegate. Effective H00346.296913381s/program5692.488814915s; no owned live process. Use three-root effective GPU accounting for all future launches. STREAMBUDTEST001 four actual regressions pass, mocked fixture statuses are not jobs. SHUBERT_LOAD001 alias preserves the distinct full-ID SHuBERT run without colliding with old SignRep LOAD001 records. [D016](../decisions/D016.md) keeps crop/annotation/pilot gates held.

STREAMAUDIT001 CPU.059415706s completed: nine predecessors/code+config scientific bindings unchanged,31official sources/four raw artifact hashes/ASTs/effective budget; artifact_audit_20261004_r10 is a historical closure snapshot before final state pointer/card append.

## Update17 — H003 input/native CUDA contracts; failed cross-device gate retained

| Record | Status | Wall seconds | GPU seconds | Scope |
| --- | --- | ---: | ---: | --- |
| [CROP001](CROP001.md) | failed | 1.784334853 | 0 | NumPy metadata JSON incident, partial report only |
| [CROP002](CROP002.md) | completed | 1.541971405 | 0 | retry1 integer-only serialization,16fixtures/all396geometry rows |
| [SOFTGL001](SOFTGL001.md) | completed | 14.175596832 | 14.095760175 | conservative probe charge,128prefix CPU-renderer coordinates exact |
| [INPUTPARITY001](INPUTPARITY001.md) | completed | 10.332137229 | 0 | all396 encoded crop pixels/body exact native,6state fixtures |
| [STREAMFEAT001](STREAMFEAT001.md) | failed | 7.040762268 | 6.602126127 | original real CPU/GPU main gate fails first context |
| [NUMDIAG001](NUMDIAG001.md) | completed | 7.198581601 | 6.638495895 | preserved7layer-cell violations, original gate still failed |
| [NUMCPU001](NUMCPU001.md) | completed | 3.114718352 | 0 | mixed-precision CPU diagnostic, not ideal full64 oracle |
| [AUTHORGPU001](AUTHORGPU001.md) | completed | 4.501499658 | 3.883341263 | separate H003_v2 nativeCUDAbitwise all4/all396/12layers |

INPUTTEST0016state/error/alignment fixtures pass; raw detector geometry includes
BORROW14duplicate native hand assignments, preserved rather than repaired.
Color path validated decoded RGB/BGR/encoder roundtrip; mp4v lossy, unencoded
crops cannot silently replace native model input. Software Mesa llvmpipe worker
has no NVIDIA libraries/device handles; bounded128frame replay only.
All attempts preserve source/config/error/argv and hashes; diagnostic completion
does not accept old parity/quality, no new HPO/test access. [D017](../decisions/D017.md),
[D018](../decisions/D018.md), [D019](../decisions/D019.md). Effective H00377.516636841s
/program5723.708538375s; no live owned job. Scientific H003 still held.

STREAMAUDIT002 completesCPU0.084842006s/GPU0:17terminal predecessors,
245original frozen bindings unchanged,31DINO sources/fourrawstores,
20ASTs and independentall396native/reference array+mask/PTS/dupflag checks.
artifact_audit_20261004_r11 documenthashes are historicalclosure snapshot before
finalpointer/cardappend. No inference or oldgateacceptance.

## Update18 — D020 tracking annotation and confidence feasibility

| Record | Status | CPU wall seconds | GPU seconds | Scope |
| --- | --- | ---: | ---: | --- |
| [TRACKREL001](TRACKREL001.md) | completed | 4.204997995 | 0 | Official XML schema/provenance; exact local train/dev ID join0/0 |
| [TRACKJOIN001](TRACKJOIN001.md) | completed | 2.455494741 | 0 | Literal name normalization still0/0; 524288B archive prefix, no accepted image join |
| [CONFSEM001](CONFSEM001.md) | completed | 0.179117717 | 0 | Installed score semantics and all396 saved rows; handedness is not crop quality |

Actual configs/source snapshots/argv/status/report SHA retained in each card and
own runtime. No model inference, calibration, policy search or test quality.
Effective program5723.708538375s/H00377.516636841s unchanged; no live owned job.
[D020](../decisions/D020.md) continues annotation feasibility and holds H003 pilot.
The prospective intake document contains no received or produced annotations.

## Update19 — H004 deterministic completion feasibility

| Record | Status | CPU wall seconds | GPU seconds | Scope |
| --- | --- | ---: | ---: | --- |
| [CERTBOUND001](CERTBOUND001.md) | failed | unknown | 0 | Parser error before RunContext; command/config/source preserved |
| [CERTBOUND002](CERTBOUND002.md) | failed | 1.015409963 | 0 | Gallery seed-key type guard before bound measurement |
| [CERTBOUND003](CERTBOUND003.md) | completed | 5.712723394 | 0 | Component envelopes, all1149 train-gallery/dev-query pairs |
| [CERTPAIR001](CERTPAIR001.md) | completed | 3.145320844 | 0 | Shared-query pair envelopes, identical cohort/order/galleries |

CERTTEST001 four manual/enumerated oracles and CERTPAIRTEST001 three mathematical
oracles pass. Both completed recipes have0half-budget stops and fail practical
gate; no saved envelope/reference violations. Component simulated windows
41408/42093; paired40363/42093. No new encoding/quality/DCG/test or latency result.
[D022](../decisions/D022.md) kills GPU scaling of these recipes; original H002
science and all failed source/config snapshots retained. No live owned attempt.

## Update20 — H005 participant geometry diagnostic

[PARTPROBE001](PARTPROBE001.md), terminal completed exit0, session49778,
CPU2.227921573s/GPU0; no encoder calls/test/dev feature scoring. All1313train
sources evaluated once across two fixed disjoint-gloss directions,100 controls.
Balanced mean.1938131684 vs control median.0303761098, delta+.1634370586,
both candidate coverages>.995; fixed15pp/95% screening gates pass. Four fixture
groups and independent recount/digest verifier pass. Participant association
only: task-harm/retrieval improvement/nuisance/novelty unproved. Initial
registration-only historical overcheck incident preserved; one actual run.
[D023](../decisions/D023.md)/[D024](../decisions/D024.md), C063/F059. No GPU
allocation or policy search added. Original H002/H003/H004 gates unchanged.

## Update21 — H005_v1 task-sensitive removal falsifier

[PARTREMOVE001](PARTREMOVE001.md), completed exit0/session52963,CPU1.243262503s/
GPU0. All378queries × three galleries × five modes =5670full50class rankings.
Participant DCG.8110123124 vs raw.7941688076; generic residual PCA.8369124636
wins. Registered gate fails despite ID-probe drop; no rank/threshold/seed sweep.
Four oracle groups and independent count/digest/basis verification pass. D025/
[D026](../decisions/D026.md),C064/F060,SESSION_AUDIT_r16. All outcomes/trained
bases/native bindings and supervision caveats retained; no test/encoder/GPU.
Kill scaling this participant-specific recipe; generic geometry transfer only
a next question, not method promotion. Effective three-root GPU ledger unchanged.

## Update22 — H006 base-class geometry transfer/control diagnostic

[GEOMTRANSFER001](GEOMTRANSFER001.md) exit0/session68178 CPU1.584446220s/GPU0;
all383dev queries ×3fixed galleries ×6modes=6894full100class ranks. Fit1313old-class
train clips;100target class strings/IDs/source bytes disjoint. True-label residual
PCA DCG.8102504602 vsraw.7667061357; all other four controls beaten, fixed gate
passes. GEOMVERIFY001 exit0/session14505 CPU1.540257314s/GPU0 independently rebuilds
all6894ranks exactly and1149raw ranks match earlier REPQUAL001, covariance/digests/
metrics/access verified. D027/[D028](../decisions/D028.md), C065/F061/r17.
Known PCA family/extra base supervision/reused five same-corpus dev participants;
continue analysis and strong-baseline review, no new method/GPU/test promotion.

## Update23 — H006_v1 strong covariance falsifier and storage incident

[SHRINKTRANSFER001](SHRINKTRANSFER001.md), completed exit0/session42366/PID2923310,
CPU3.101969110s/GPU0; all383dev ×3galleries ×8modes=9192full100class ranks. Residual
LW cosine.829746 beats PCA.810250 at everygallery (raw.766706); fixed primary gate
fails. D029/[D030](../decisions/D030.md),C066/F062/r18; kill hardrank8 superiority/
scaling, retain standard geometry analysis. Three math/error fixtures pass;
SHRINKVERIFY001 exit0/session90347/PID2927854/CPU1.316439716s/GPU0 independently
rebuilds all6894new LW original-space ranks, fourth-moment fit covariance/coefficients,
all9192row/aggregate/gate/digest contracts. No retries/test/new encoder/HPO.
Measured durable40,750,758bytes exceeds predeclared20MiB cap; retained explicit
resource failure, not scientific-score failure. Prospective size planner rejects
current format; compact writer enforcement still pending before new measurement.

## Update24 — Checkpoint and public source-pool lineage

LINEAGE001 completed CPU1.735427503s/GPU0; LINEAGE002 technical header parser
failure CPU1.735323351s/GPU0 retained; LINEAGE003 retry1 completed CPU1.141348479s/
GPU0. Exact release matches/model-only327tensors;39197public sourceIDs include
all11096localYouTube-ASL IDs. LINEAGEVERIFY001 completed CPU0.1331578780664131s/
GPU0, independent first-record/count/byte/digest/prelaunch-cap verification passes.
D034/C067–C068/F063–F064/r19; no checkpoint training membership or encoder-unseen
claim, no task/new encoder/test-quality access. Compact writer next prerequisite.

## Update25 — Compact writer and full CPU-context validation

ARTIFACTTEST001 completed exit0/PID2987258/captured wall.206736575s/GPU0:
14fixed software checks, fullsize synthetic numerical archive9,406,478bytes<20MiB.
ARTIFACTCTX001 completed exit0/PID2997175/wall.102253448s/GPU0:5fixed integration
checks, actual bounded parent12files/43,493bytes below2MiB. All registered inputs/
config/output/manifestcaps verified, fullfixture independently recounted. No
new task outcome/original output conversion/retroactive budget repair. D035–37,
C069–C070/F065–F066/r20. Next H006_v2 axis/spectrum DRAFT, no run launched;
implement independent mathematical controls and lock future bounded wrapper.

## Update26 — H006_v2 fixed direction/spectrum factorial

AXISTEST0013analytic groups pass/exit0; AXISTRANSFER001 completed exit0/PID3020212/
wall1.569296160s/GPU0,4596fullranks. RR.829746 vsTR.769915 matched spectrum,
+.0598309112/all3positive passes fixedgate; RT.830849/TT.762639 secondary.
AXISVERIFY001 exit0/PID3023264/wall.534516770s/GPU0 exact all4596original-space
ranks/2298parentrows/rowmetrics/aggregates/intervals/gate/sourcefilecaps verified.
D038–D040/C071/F067/r21, same-dev computational analysis only; no new method.
Actual14,810,320B<20MiB, complete new writer plan15,821,902B, old capviolation
unchanged. Next cross-corpus MS-ASL metadata/native/source/signer feasibility,
not more same-dev search or P004test rescue. All terminal, no GPU/newencoder.

Update27 / D043: MSMETATEST001/002 two groups each pass; MSMETA001/002 and
MSMETAVERIFY001 completed exit0. Actual archive once2,022,053B;5byteequal members,
25513source/CSV rows independently checked.7212parents/4269raw candidates/2943missing;
signer_id disjoint/sourceparent3/2/0. No model quality/native decoding/GPU.
Five-run captured wall3.144955038s,7,169,390logicalB summed, each under registered
2/12/2/12/2MiB ceiling. One prelaunch configuration construction FileNotFoundError
preserved; fresh helper binds old manifest_audit. Original parser/narrow census
immutable; explicit URL syntax newversion. C072/F068/r22; next registered CPU
native/source/crop/PTS feasibility, no official full1000 benchmark or method promotion.

Update28 / D047: MSNATIVETEST0013groups pass; MSNATIVE001 failedserializer retained,
MSNATIVETEST0024groups pass; MSNATIVE002 actualfour297frames/120nativewindows
PTS<=1e-6/independentresizedRGBexact; MSPREPROC0013constants/64nativeframes/
9,633,792manualnormalizationscalars passfixed1e-6, max2.2373e-7. Fiveattempts
capturedwall7.342458018s/GPU0/15,343,368logicalB total, eachcap verified.
C073/F069/r23, no model/quality/contributionpromotion. NextnewboundedGPUcontext
and preregisteredfourwindowfeatureparity/cost beforeadaptedMSASLscience.


Update29 / D053: GPUCTXTEST001 three CPU simulation groups pass; MSFEATURECPU001
28realCPU forwards/four native windows strictload/directwrapper/repeats bitwise.
MSFEATUREGPU001 registerednotlaunched/resourcehold/H006120s. MSCOHORT001 prelaunch
file-bound rejected/nolaunch; MSCOHORT002 wrong-IDactual completed but invalid
preregistered executionidentity, retained; freshMSCOHORT003 exactlaunch completes.
All4980metadata rows verified;2677train/748val/all100classes, no nativecoverage or
quality. Fourterminalactualdirs4,442,797B/115.275464643capturedseconds/GPU0, each
capverified. D048–D053/C074–C075/F070–F071/r24. Next whole3425nativeeligibility;
no methodpromotion/PAPER_READY or same-dev tuning.


Update30 / D058: streaming native fixturesMSNATIVESCANTEST0012groups/0023groups
complete; firstMSNATIVECOST001failedreferencefield/earlyassert retainspartial
outputs; versionedparser/retention costretry1 MSNATIVECOST002 completes11clips/
1155exactffprobe native ticks/four oldRGB16/count/time references. Fourterminal
actualstores356263B/wall17.551602580s/GPU0, everycap/completemanifest verified.
MSNATIVECOVER001 live51449/PID3209972/full3425candidateaudit at1400rows/14incomplete
lastobserved. CPU7200s/20MiB/fixed4ea6cf6a. Resumehandle, finishallmanifest, inspect
incomplete actualcauses beforepromotion; not fullcoverage/quality result. D054–
D058/C076/F072/r25. Prelaunchscope/path/interpretererrors retained; no newmethod.


Update31 / D062: originalfullnativeaudit terminalfailedgate afterALL3425IDs,
3384eligible/38trainrawclockfail/3trainshort; val748valid/100classes bothsplits,
no scientificpromotion. DICTTEST0017syntheticgroups pass. FreshD0602GiBallocator
MSFEATUREGPU002 completes28forwards/all3fixedCPUcombinedparities/bitwiseGPU
direct/repeats, reserved560MiB/charged3.141068830s. MSPTSDIAG001 all38failures
rawprefix+selectedticks exactindependent ffprobe;30withinclip/7preroll/1boundary,
65backwardpairs, best_effort estimatedclock differs. No silentfix/quality/gallery.
Fouractualstores6028666B/capturedwall1630.122400269s (overlap), eachcapverified.
D059–D062/C077–C079/F073–F075/r26. Allterminalhandlesclosed, no ownedlivejob.
Nextregistered BOTH-clock/framecorrespondence audit, not blindfullrerun/error drop.


Update32 / D065: MSCLKTEST0014groups and0025groups pass; retainedMSCLKDIAG001 serializerfailure, freshMSCLKDIAG002 all38 dualclocks/entireRGB38exact. Wholeprefixraw37/38/estimated36/38 fails; twooutsidesuffix joins, selectedclock/RGB38exact is diagnosticposthoc. One train_11136 membership59->60. Allfourcaps7058505B/52.8234224220505s/GPU0 verified; nojob/quality/promotion. C080/F076/r27; next explicitestimatedclock adapter/manualfixtures/actual38cropaudit.


Update33 / D068: MSESTTEST0016groups and MSESTCOVER001 all3425 adapted input contract pass:2674train/748val eligible,3short unchanged,100classes each;38newdecode/3387inheritance. MSESTENV00143 input hashes/counts/clock/crops exact. MSDICT0013 fixed shared galleries/all748queries,289unique supports/1037extractionclips/36424windows frozen. Four stores9307506B/capturedsum35.48027317679953s/GPU0/caps and current bindings verified; allterminal/no ownedjob. D066–D068/C081–C083/F077–F079/r28. Next registered extractioncost wrapper+fixtures on metadata-locked7clips/396windows before fullquality budget/protocol. Original rawclock/prefix failures retained.
