# Continuation state — 2026-10-07

Goal **active**, turn **progress**. Original M0–M6 objective remains incomplete;
research efficacy unproven. All changes stay inside method/signcoord_ssl; other
methods, original data/checkpoints and vendor source remain untouched. Root has
unrelated pre-existing dirty/untracked files; do not commit them.

## Current verified implementation

Implementation SHA256:
542b3605a2a84171bb7e9ec109b39395589d25669047fbb9295c0c204f939eef.
86 tests pass on .venv-signrep/bin/python, CPU, two threads, unchanged source
during final isolated validation. See docs/evidence/tests_report.json. Earlier
failed/intermediate runs remain under runs/validation; final pass is authoritative.
The last pre-pass failure was a test regex mismatch, not a missing mutation check.
Wheel installed into an owned target, no shared venv mutation, matching source
hash; SHA d6434d60138ebedcc67873781300260b5c9665842c646793c4689f61d0b044cf.
Root-anchored ignore rules now expose src/signcoord/data to Git/rg.

DDP trainer: global weighted means/differentiable statistics gather, disjoint
global seeded sample plan, no_sync accumulation, rank RNG, atomic rank0 save,
exact same-world/source/config/input optimizer-boundary resume. Dropout+accum
resume is bitwise exact. Full FP32 vs centralized same-batch gradients:
max2.38e-6/relativeL2 1.40e-5; analytic double checks remain strict. Empty local
geometry works if global support exists; all-global-invalid rejects. Known-boundary
loader/prediction/gradient/rank0-I/O failure synchronization works. Hard process
or communication failures require launcher timeout; not elastic fault recovery.
CPU snapshots never query CUDA; CUDA snapshots save only this rank's device RNG.
Bare cuda resolves to cuda:0 before setup. GPU/NCCL/AMP execution pending.

Actual torchrun CLI: runs/fixture_ddp_recipe_bound_v8, two Gloo CPU ranks,
8 updates, stats batch4/optimizer batch8/accum2, active phase, full checkpoint.
Synthetic engineering only. A9 gathers detached targets/reliability/support/
times/IDs globally and deranges compatible groups; one-sample ranks exchange.
Singletons skip. Masks derive per sample from seed/update/ID independent of rank
partition/module RNG; clip-shared jitter also uses its own RNG. Unsupported
config recipes reject. A1-inspired/A2/A3/A6 need no phase calibration.

Both SignRep and generic Hiera strict loaders/temporal-only adaptation work at
T16/T32; T64 rebuild/index verified, full forward pending. Source/schema/weights
locks in docs/source_lock.json. Generic source b12b842542ee5c757fcfec8c41f6b56fcbe89b65,
official video MAE SHA dd80be81ca35a575511e7b5204cac07368c3e497c79dbe0da9fe339fb86f6dea.
Scoped zero-class-token 4D SDPA preserves max/tie gradients, masks and weights.
Same-math full-model parity exact; fused masked max2.7737e-4 fails initial3e-5
gate, relativeRMS2.04e-6 disclosed separately. GPU/AMP parity pending. A1_ref
retains vendor_5d/native T16 stride2/direct resize/strict fc; earlier official
post-fc parity4.77e-7. Video post-fc averaging is an extension. A1-inspired pooled
shared-2D MLP is not the unavailable exact original prior reproduction.

Portable native export ships deployment modules/vendor licenses/hashes/relative
paths/dependency versions. Actual generic T32 FP32 relocated fresh-process
outputs match exactly with forbidden training/pose imports. See
docs/evidence/portable_export_report.json and
runs/portable_export_generic_T32_v1/relocated_bundle/encoder.pt. Export report
has an earlier full source hash; relevant deployment modules have not changed.
Current tests also verify relocation/isolation. ONNX/TorchScript unattempted.

## Data and actual evidence

Reader tracing plus50 inspected overlays support x/W,y/H conversion. Original
pose estimator/config/weights, independent pose PTS, tracking identity and SSL
signers remain unknown. Train-only body/face/L/R q90 over100 How2Sign clips
defines clip(raw/q90,0,1) quality heuristic, not probability; raw scores retained.
Zero/multi-person/out-of-camera observations are invalid. Unknown tracks stay
unfilled. Verified equal integer track IDs permit at most2 missing source frames
before affine with .5 endpoint quality; tests cover changed/unknown/out-of-frame.

Current100 train/20 source-disjoint dev caches: data/how2sign_pilot_v1.
Use stats/target_stats_recipe_bound_v8.json and phase_calibration_recipe_bound_v8.json.
Statistics bind NPZ contents, gap policy, geometry/spatial/repeat recipe,
manifest/schema/timing. Old bounds_v4/controls_v5/source_gaps_v6/pose_bound_v7
fail current validation. Fitting crop seed is recorded/shared across matched
arms. Positive calibration count remains92 per pair/bin. Ready engineering
config configs/real_smoke_recipe_bound_v8.yaml; choose a fresh run_dir.

New real smoke runs/A7_how2sign_real_smoke_recipe_bound_v8:4 CPU FP32 updates,
SignRep external initialization, microbatch2, shortened relation schedule,
active supervision at updates3/4, finite gradients, final checkpoint. Reports,
coverage and ledger under docs/evidence/A7_real_smoke_recipe_bound_v8_*.
Not the generic-init matched10k pilot. Earlier A5/A9 controls_v5/other smokes
remain diagnostics and cannot exact-resume across source/statistics changes.

Production manifest adapters ASL-Citizen/WLASL2000/How2Sign/configured_source
freeze full PTS decode, constant geometry/media hashes, reusable owned indexes,
separate labels and availability/exclusion ledgers. WLASL metadata/preprocessing
pinned ac00e6be631c1a2a486621b65f202219f3964d6b; fetch_source_metadata.py reproduces
owned files. Original frame intervals are provenance, never reapply to trimmed
MP4s.12 local IDs absent upstream fail if selected, no guessed annotation.

ASL-Citizen screen200 train gallery/100 dev,50 classes,23/5 signers,0 failures/
known source/media/signer overlap. WLASL original200/86, filtered197/86 all50
classes,3 shared sources removed; no remaining known media/source collision,
34 known shared signers. Not signer-held-out. These are subset screens only.
Selected SSL/eval pHash yields3 How2Sign candidates, visually different content
with hashed keyframe sheets/explicit decisions. Bounded screen is not proof of
all-reuse absence or identity. Full downstream dev/test audit still required.

## Live work — verify before any launch

A0 frozen ASL-Citizen screen LIVE: exec session68498, parentPID2746172,
gallery workerPID2746326 at last observation. Run
runs/A0_ASLCitizen_screen50_capsule_v1. Immutable source capsule hash
94c2b72eabc1cd842df6b0a1dec5374c4f35f3d9ed065712f5102dc02e294d9b,
generic T32 CPU FP32, zero sign SSL updates,200 gallery/100 dev. Do not edit
capsule, restart on tool timeout or kill. Poll session68498/live_step.json;
collect gallery/query/retrieval/screen_report when done. Main code can evolve
without invalidating this run. No real frozen metric/objective comparison yet.

GPU remains occupied by unrelated p4_signssl PID2693096/H012b_X1, last22614MiB/
49140MiB100% utilization. Previous PID4158611 is stale. Do not terminate other
experiments; verify live state before GPU work. Other CPU jobs heavy. All own
transient checks above completed; only frozen A0 remains live.

## Next toward the full original objective

1. Collect A0 screen; execute disclosed A1_ref. Complete full primary/secondary
   manifests/full leakage audit and production pose sharding/pinned fresh
   extractor/tracking/quality sensitivity. Subset pass is not full strict gate.
2. Add selected-dev checkpoint evaluation while preserving fixed-budget final,
   diagnostics and compare validation. Verify GPU/NCCL/AMP/attention parity and
   useful GPU/end-to-end profiles. DDP long rank0 eval needs collective-safe wait.
3. Implement large feature sharding/optional sequence storage, probes/kNN/label
   fractions, slices, CTC training/eval CLI, translation head/CLI, matched
   comparison/bootstrap provenance. Avoid storing unnecessary full sequences
   for every retrieval arm with only ~73GB disk free.
4. Run sufficiently large matched generic-init A2/A3/A6/A7 pilot10k updates with
   common inputs/masks/decoder/stats/optimizer/readout and equal dev tuning.
   100 clips/4-update smokes/50-class screens do not replace this protocol.
5. Follow preregistered_plan.yaml: A3/A6/A7 seeds[3407,2027,42], two datasets,
   locked final test, A5/A8/A9/context/extractor controls, compute/limitations.
   Null results valid; no phase-gain claim without matched frozen A7−A3/A7−A6.

Independent engineering remains; GPU contention is not an impasse. Keep goal
active. See implementation_plan.md for full M0–M6/CLI acceptance contract.
