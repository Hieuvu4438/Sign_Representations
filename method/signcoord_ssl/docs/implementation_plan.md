# SignCoord-SSL implementation plan

## Material Passport

- Objective: proposal9 spec v1.0, isolated implementation and real evidence for RQ1–RQ4.
- Authority: `docs/proposal9/SignCoord_SSL_End_to_End_Implementation_Spec_VI.md`.
- Mode: implementation + experiment execution; hypotheses supplied, no new scoping needed.
- Artifact root: `method/signcoord_ssl`; existing method code and third_party stay read-only.
- Research status: **not established**; engineering status: **in progress**.

## Dependency and acceptance matrix

| Milestone | Depends on | Required deliverables / acceptance | Current status |
|---|---|---|---|
| M0 | spec | source/checkpoint/schema lock; actual environment; typed configs; doctor; decisions and reproduction delta | Inventory/config/doctor and source/schema/SignRep/generic locks verified; complete CLI contract pending |
| M1 | M0 | real adapters + overlap/near-duplicate audit; PTS video sampling; canonical pose cache + extraction; affine; reliability; train-only stats; fixture; 50 real overlays | Production ASL-Citizen/WLASL/How2Sign adapters and frozen PTS/media indexes implemented; 50 overlays inspected; 100/20 How2Sign caches; verified-track bounded gap integration and pose/recipe-bound train statistics tested; selected-cohort perceptual audit reviewed; original extractor, production pose sharding and full-corpus leakage/tracking gates pending |
| M2 | M1 | official SignRep compatibility; generic upstream Hiera; T16/32/64 geometry; index tests; A0/A1; common dense decoder; geometry/velocity/projector/DDP tests | Both backends strict T16/T32 load/forward/index tested; T64 rebuild/index tested; A0 materialized and exported; official A1_ref wrapper parity and A1-inspired pooled control implemented; dense/backward and 2-process losses/gather verified; full scientific readout evaluation pending |
| M3 | M1,M2 | analytic phase; train calibration; lag/magnitude/A4/A8/A9; target-only gate; sign/swap/gradcheck/coverage evidence | A4/A5/A6/A7/A8/A9 kernels/controls tested; magnitude train statistics and global DDP compatible derangement implemented; real A5/A9 active finite smokes; full trajectory/collapse/loss-gradient diagnostics pending |
| M4 | M2,M3 | named AdamW/AMP/DDP/accumulation; deterministic paired plans; atomic checkpoints and exact boundary resume; finite fixture and real smoke; profiles/ledger | Single/2-rank CPU engine, exact rank-RNG/dropout/accumulation resume and centralized gradients verified; actual torchrun CLI completed 8 synthetic updates with active phase; known-boundary failure synchronization tested; GPU AMP/NCCL, selected-dev loop and stable GPU profiling pending |
| M5 | M4 | frozen full-video coverage feature extraction; instance retrieval/probes; paired clustered bootstrap; slices; CTC/SLT heads + CLIs/tests; RGB-only export parity/isolation | RGB-only features/retrieval/bootstrap and portable relocated generic T32 export verified; A0 real 50-class ASL-Citizen screen running in source capsule; probes/SLT/CTC CLI, comparison validation, large sharding and full real evaluation pending |
| M6 | M0–M5 | A0/A1 references; A2/A3/A6/A7 matched real pilot; preregistered ≥3 seeds; A5/A8/A9/context/extractor controls; 2 datasets; measured frozen metrics + compute + limitations | Pending; no SignCoord-SSL benchmark claim |

## Required interfaces

The CLI must eventually implement doctor, manifest, audit-data, extract-pose,
fit-target-stats, audit-targets, smoke, profile, train, extract-features,
eval-retrieval, eval-probe, eval-ctc, eval-translation, compare and export.
An absent command is unfinished, not a successful placeholder.

## Verification policy

Section 20 unit/integration/real gates and section 21 final handoff are the
completion contract. Actual reports must carry code/config/data hashes and
timestamps. `not_run` is not a passed gate. Synthetic results only validate
engineering. Confirmatory phase claims require matched A7−A3 and A7−A6 frozen
evaluation, seed variability and paired signer/source intervals. Neither GPU
availability nor existing SignRep scores prove this proposal works.

## Outstanding real inputs

Legacy local `.pkl` arrays have 133 nodes but unidentified coordinate scaling,
scores above 1, missing extractor/config/weight provenance and missing pose PTS.
Coordinate scaling is now supported by previous readers and 50 overlays;
scores use an explicitly versioned train-only heuristic. Original extractor and
true tracking identities remain unknown. A canonical schema's existence alone
does not prove original extractor provenance.
The generic MAE checkpoint and upstream backend are now pinned and verified
independently of the SignRep continued-pretraining backend. The currently busy GPU belongs to
another experiment and must not be interrupted.

## Current authoritative evidence

- `docs/evidence/tests_report.json`: 86 executed tests, zero failures, with case IDs and unchanged source during execution.
- `docs/evidence/adapter_report.json`: actual checkpoint T16/T32 forward and
  keep-mask/native indexing parity; T64 forward explicitly not run.
- `docs/evidence/generic_hiera_adapter_report.json`: upstream official generic MAE strict load, T16/T32 forward/index/full-keep parity; T64 rebuild/index only.
- `docs/evidence/signrep_reference_report.json`: strict full official-wrapper and transform parity, post-fc max error 4.77e-7.
- `docs/evidence/A5_real_controls_smoke_report.json` and A9 counterpart: real How2Sign 4-update finite/active smoke, coverage reports include weighted losses and skipped shuffles.
- `docs/evidence/real_rgb_smoke_features.npz.json` and official-reference counterpart: one real 1.28s interval, RGB-only extraction, no benchmark metric.
- `docs/evidence/smoke_report.json`: actual current CLI fixture A3/A6/A7 pipeline,
  synthetic metrics explicitly labeled; short CPU profiles not budget estimates.
- `docs/evidence/real_smoke_report.json`: final real How2Sign 4 updates with active
  phase supervision and finite backward; no real frozen representation metric.
- `docs/evidence/real_target_audit_bounds_v4.json`: final source-bounded validity.
- `docs/evidence/real_data_overlap_report.json`: exact/source/session check only;
  near duplicate and signer verification remain unfinished.
- `docs/evidence/pose_trace_summary.json`: 50 visually inspected middle frames.
- `docs/evidence/package_report.json`: isolated wheel installation/doctor test.
- `docs/evidence/ddp_cli_smoke_report.json`: actual launcher/CLI, two CPU ranks, 8 updates, stats batch4/optimizer batch8, active phase and full rank RNG.
- `docs/evidence/attention_kernel_report.json`: exact same-math layout parity; fused float32 drift and the failed initial absolute gate disclosed; GPU pending.
- `docs/evidence/portable_export_report.json`: actual generic Hiera T32 relocated fresh-process output parity, no training/pose imports.
- `docs/evidence/real_target_statistics_recipe_bound_v8_report.json`: current train-only statistics bound to NPZ contents, gap policy and geometry/spectral recipe.
- `docs/evidence/screen_overlap_perceptual_reviewed_v1.json`: selected cohort only, three candidates reviewed with hashes, signers still unverified.
- `docs/evidence/wlasl_gallery_query_source_filtered_report.json`: 197/86 selected records, no known media/source collision, 34 shared known signers explicitly disclosed.

Short-gap interpolation is integrated before affine and tested against verified,
changed, unknown and out-of-frame tracks. Legacy unknown tracks remain disabled;
ambiguous-person frames are invalid, never made observed by guessing identity.
