# Decisions

## I01 — implementation isolation

All new code, configs, manifests, caches, reports and runs live under
`method/signcoord_ssl`. Vendor imports are read-only and independent of other
method packages. D01–D12 from the spec remain the defaults.

## I02 — use the installed environment for validation

CPU tests use `.venv-signrep/bin/python` (PyTorch 2.11.0+cu128 observed). No shared
venv package mutation. Record a local dependency lock; fresh environments may
install the package normally. GPU allocation is deferred while an unrelated
live training process owns the device; CPU work continues.

## I03 — legacy pose caches are not canonical inputs yet

Initial samples contain `[F,1,133,2]` keypoints and `[F,1,133] scores, normalized-
looking coordinate values and scores above 1. Their semantics/extractor/PTS
are unverified. Do not silently multiply by image size or clip all logits to
1 and call the result confidence-calibrated. This is a data provenance issue,
not evidence against the method. Canonical NPZ inputs require documented
pixel coordinates, [0,1] quality scores, semantic schema hash and matching PTS.

## I04 — compatibility backend

The existing SignRep commit is exactly the commit in the proposal. Build the
Hiera core directly with the public video architecture to rebuild T32; omit
training heads under an explicit prefix allowlist; interpolate separable
temporal positions only; load every encoder key strictly. This backend is
`signrep_pretrained` (continued adaptation), not generic K400 MAE. Uniform
encoder pooling differs from the official `fc`/activity readout and must be
evaluated as a separate protocol.

## I05 — source PTS quantization and bounded nearest mapping

The initial 25-Hz-grid half-interval tolerance incorrectly discarded valid
nearest frames from 24-fps sources. Use half the median source PTS spacing.
Keep the 10% repeat limit. Distinguish randomized crop-grid origin from the
manifest's allowed interval beginning so a nearest earlier frame can be used
within the approved source interval. Preserve v1/v2/v3 diagnostics and fit new
statistics; do not resume across source or calibration changes.

## I06 — legacy quality, source visibility and tracking

User instructed tracing other method folders and deciding from evidence. The
previous readers plus 50 train overlays support x/W,y/H conversion. The original
extractor remains unidentified. Fit body/face/L/R score q90 on 100 SSL-train
clips, use clipped score/q90 as a disclosed quality heuristic, and keep raw
scores in cache. No probability claim. Invalidate zero/multiple-person frames
and retain unknown tracking identity. A joint outside the source camera image
stays invalid even if letterboxing moves it into the output padding. Final
stats carry `source_and_transformed_frame_bounds_v1`; older stats fail validation.

## I07 — fixture duplicate audit

The first full CLI fixture pipeline correctly failed exact-media overlap:
periodic synthetic parameters produced three train/dev duplicate videos.
Add instance-specific constant background values and regenerate under a new
run ID. The overlap gate is not bypassed. Final fixture pipeline uses the
current generator and source visibility policy.

## I08 — engineering evidence scope

Short real CPU smoke uses 4 updates, microbatch2, FP32 and an explicitly shortened
relation schedule. This exercises real gradients but is not the 10k-update
matched pilot. Short CPU profile uses full relation weight and predecoded fixed
inputs for a kernel/optimizer timing check; it is not stable GPU or end-to-end
throughput. No research result is inferred from either.

## I09 — pinned generic Hiera and frozen A0

Fetch the exact upstream revision b12b842542ee5c757fcfec8c41f6b56fcbe89b65
and official video MAE K400 weights (SHA256 dd80be81ca35a575511e7b5204cac07368c3e497c79dbe0da9fe339fb86f6dea)
inside this project. Build the core with q_pool2 and separable temporal positions;
strict-load every encoder key, allowlist known MAE training keys, and adapt
temporal positions only. Final-stage pre-fusion/pre-norm readout is common to
this study. A0 has no pose targets/optimizer/sign SSL exposure. T16/T32 forward
and index parity pass; T64 remains rebuild/index only. Source code is Apache2.0,
weights CC BY-NC4.0, recorded separately in the source lock.

## I10 — official reference and pooled mechanism control

A1_ref uses official transform, native-frame T16/stride2, and strict fc loading.
Full-wrapper parity is measured, not inferred. Video averaging after fc is an
explicit downstream extension; the reference is not matched continued training.
A1-inspired uses one pooled visible-memory vector and an MLP to reconstruct the
shared 2D geometry sequence, with common geometry/regularization but no velocity.
Original angle/distance/3D prior reproduction remains unavailable.

## I11 — magnitude and shuffled controls

A5 averages reliability-weighted per-anchor auto-power on the same mean-centered
Hann windows. Train-only log mean/std (std floor .05), auto-energy q10 (floor1e-6)
and clipped auto/reliability gate define normalized log-power squared error;
no cross-spectrum/phase participates. Loss scale differs from unit phase error,
so log weighted contributions and gradient norms before interpreting a negative
A5 result. A9 deranges relational targets inside exact basic-valid support and
microsecond interval-timing groups using a deterministic nonzero cyclic rotation.
Geometry/velocity targets stay correct; target reliability/gate moves with the
shuffled signal. Singleton groups skip the relation term, with counts/target IDs
logged. The original policy was single-process. I16 extends the same compatible
derangement to gathered detached targets over the global DDP microbatch.
Real A5/A9 each completed 4 CPU updates with active supervision and finite
gradients. These are engineering checks, not matched efficacy experiments.

## I12 — validation artifact isolation

Validation writes per-execution JUnit/log/report artifacts under runs/validation
and promotes them only when implementation and test source remain unchanged
during execution. This prevents overlapping validation runs from mixing reports.

## I13 — production manifests and conservative WLASL sources

Select downstream screen IDs/classes by metadata-only seeded hashes, before
any representation score. Preserve separate labels and full decoded-PTS/content
verification. WLASL original URLs define conservative source groups; remove
three gallery sources shared with the selected dev cohort into a new manifest.
Preserve original clip frame annotations as provenance, never double-trim.
Full-corpus and signer-held-out claims are not inferred from these subset audits.

## I14 — attention layout and numerical backend

Use instance-scoped 4D SDPA for zero-class-token Hiera, keeping source weights,
pooling max/tie gradients, masks and architecture intact. Same math backend gives
exact full-model dense/masked parity. The initial fused absolute 3e-5 gate fails
on masked tokens (max2.7737e-4); fused relative RMS is 2.04e-6 and is reported
separately. Do not hide this failure or claim bitwise fused parity. The official
A1_ref explicitly retains vendor_5d. GPU/AMP parity remains to execute.

## I15 — bounded perceptual candidate review

Five-keyframe pHash is a candidate screen, not duplicate truth. Three candidates
are reviewed as different content with contact sheets/frame hashes. Review never
establishes signer identity. Unknown How2Sign signers remain unverified. Full
downstream audit remains a separate required gate.

## I16 — complete DDP batch/optimizer boundaries

Use one global seeded permutation, disjoint rank slices, equal local sizes and
drop an incomplete global microbatch. Statistics batch is local batch×world;
optimizer batch additionally multiplies accumulation. no_sync spans forward and
backward on earlier accumulation microsteps, while weighted loss/gather
collectives remain per microstep. A9 gathers detached target signal/reliability,
basic support/timing and IDs, derives one deterministic global derangement, then
selects this rank's targets. One-sample ranks can therefore exchange targets.

Save only on rank0 at optimizer boundaries, retaining all rank RNG states and
the global sample plan. Exact resume requires unchanged source/config/input
hashes and world size. Loader/input/prediction/nonfinite-gradient and rank0 I/O
errors synchronize at known boundaries. Hard process/communication failures
still require the launcher/timeout; this is not fault-tolerant elastic training.
Two-rank dropout+accumulation resume is bitwise exact. Full FP32 engine vs a
centralized same-stats-batch gradient differs by max2.38e-6/relativeL2 1.40e-5;
the initial 2e-6 element bound fails and the documented 5e-6/3e-5 aggregate
numerical bounds pass. Analytic double-precision loss/gather checks remain strict.

## I17 — paired stochastic plans and pose artifact freeze

Per-sample masks derive from seed/update/sample ID, independent of rank partition
and module/global RNG; clip-shared photometric factors use a local seeded RNG.
This keeps matched arms' inputs/masks comparable despite different heads.
Train statistics additionally bind all canonical NPZ contents. Read-time stable
fingerprints and hashes reject changed pose content; v7 statistics are refit on
the same train cohort. v8 additionally binds normalization/spatial size/repeat
eligibility so changed target recipes cannot reuse stale statistics. Crop fitting
seed is recorded and shared across matched arms. Track-unverified legacy gaps
remain unfilled. CPU checkpoint RNG never queries CUDA; CUDA checkpoints save
only this rank's device RNG. Bare cuda resolves to cuda:0 before device setup.

## I18 — portable encoder-only native bundle

Export deployment modules and pinned vendor source/license alongside encoder
state, dependency versions and source hashes, with relative vendor paths.
Fresh relocated-process inference requires neither original vendor paths nor
training data/targets/pose/decoder/projector imports. Actual generic T32 FP32
parity is exact for sequence, times, validity, raw and normalized clip outputs.
ONNX/TorchScript remain optional/unattempted; no efficacy result follows.

## I19 — ignore boundaries

Anchor data/runs/vendor/build ignore rules at the method root. The previous
unanchored data/ rule accidentally hid src/signcoord/data from Git and rg;
source modules are now visible while owned artifacts remain ignored.
