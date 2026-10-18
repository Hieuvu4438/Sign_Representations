# Staged design and implementation review

**Historical record:** user requested retirement on2026-10-17. Runtime evidence,
weights, logs and smoke artifacts were deleted; the systemd service was removed.
The source/config/tests remain. References to runtime evidence below describe
the verified setup at launch and are no longer available locally.

## Material Passport

Date:2026-10-16. Scope:user-authorized new method and persistent full-training setup. Material:local PLctx/Pose-JEPA/S1/S2 code, official split files, existing full-train cache and SHA-pinned full160PL weights. Implementer self-review plus adversarial/integration/execution tests; no independent human or external-model review. No dataset upload. The verification status below concerns code execution and setup, not a claim of lexical improvement.

## Stage 1: design and input provenance

The method isolates a frozen complete PLctxfull160 encoder and a new body/relations graph branch. This preserves learned handshape context while allowing geometry to contribute through learned bounded per-group gates. The old Pose-JEPA30k/full weights no longer exist: the graph starts fresh and reuses its architectural ideas, not its learned feature space. There is no claim that geometric probe dimensions identify a pure detachable handshape/body representation.

Raw inputs reuse the existing103-joint,3.33GiB uncompressed full-trainNPZ; no RGB/pose cache is duplicated. A ZIP_STORED memory-map loader reopens arrays by offset in each spawned worker, avoiding serializing3.4GB per worker. Tests inspect actual real clips and require a pickled cache handle under500bytes plus worker0/worker2 data equality.

Preflight verifies ordered full40154training IDs, official val10304/test32941 IDs, train/val/test signer/video disjointness, cache byte hash, PL checkpoint byte hash, split file hashes, keypoint stat inventory and environment. The stat inventory is explicitly not a full content-hash audit of every original detector/video byte.

Design tradeoffs:the new branch uses core torso and two arm-chain GCNs plus wrist/body/head/wrist relations, not Pose-JEPA's full four-part hand/face architecture. Training windows remain64samples, so geometry is aligned with frozen PL features. This is a two-branch method adaptation in body units rather than rebuilding deleted Pose weights. ExistingPL160 pretraining budget must be counted before the additional160fusion epochs.

## Stage 2: leakage, graph support and fusion

Student raw masking inherits reviewed interpolation support, anatomical ownership, wrist/nose duplicate closure and torso tubes. Graph messages are frame-local, with explicit node validity; graph attention keys are removed if any declared supporting source is forbidden. Explicit relation tokens require all raw sources allowed and corresponding context channels visible. Teacher geometry uses the student's shoulder frame/affine; contextual teacher retains its own affine view.

Review findings fixed:

1. The original PL shoulder-unavailable fallback used all observed joints, which could allow hidden points into a coordinate reference. The frozen-weight feature implementation now accepts the safe shoulder-only reference from the reviewed pipeline; absence fallback uses a fixed origin/scale. This is a preprocessing correction, so parity is stated only for identical input/reference, not falsely universal parity with every legacy missing-data case.
2. Stochastic hand dropout must not become a masked geometric target. Auxiliary weights now use explicit `forbidden` joint support and true target ownership rather than `t_obs & ~s_obs`.
3. A parameter named `type` collided with `nn.Module.type`; renamed to `part_type` before any real optimizer steps.
4. Optional graph diagnostic output is extracted separately; the primary output remains D/N/face768D. Gate0 parity is checked both at token level and against the real original pretrained extractor, allowing only its fp16 storage quantization tolerance.

Evidence:raw hidden-coordinate/confidence perturbations across five seeds in each near/far view leave encoded outputs exactly unchanged. Missing shoulders and arbitrary missing-coordinate payloads cannot change outputs. FrozenPL state remains bitwise identical after training; graph/fusion/gate get nonzero gradients after the opening schedule; teacher receives no gradients. Gate coefficients are bounded at.3; LayerNorm/residual tensors themselves are learned, so this is not a mathematical guarantee of constant Euclidean feature distance.

## Stage 3: optimizer, checkpoints and validation transactions

The optimizer excludes frozenPL and teacher weights. Full state includes student/teacher graph+fusion, frozen reference, compatible initialized predictor/context head, geometry heads/stats, optimizer, center, RNG and consumed-update cursor. Fixed training permutations and `(seed,step,draw)` views make DataLoader prefetch irrelevant to the saved cursor.

Review findings fixed:

1. Restoring a validation checkpoint must not use a serialization byte hash that changes upon each save as the feature-cache identity. Validation cache identity now hashes teacher tensor contents plus step/provenance/stride, enabling partial extraction reuse.
2. Validation must not perturb training RNG at a committed checkpoint. RNG is restored before best/last commit and again in the validation finally clause. DataLoader generators are separate from training RNG.
3. Falling back from corrupt last to a readable previous must not overwrite the good previous with corrupt last on the next save. The corrupt last is quarantined before rotation. Source/config/data mismatches never trigger fallback/fresh training.
4. OOM can happen before the inner training loop. Train/evaluation entry points also return the resource-requeue code for initialization OOM. Partial optimizer state is not saved after OOM; recovery starts from the last fully committed checkpoint.

Tests compare uninterrupted and restored CPU losses/model/auxiliary state exactly, exercise corrupt-latest fallback and changed-provenance rejection, and signal the actual CLI subprocess during validation. The latter saves `pending_validation=0`, exits75, resumes validation before any optimizer step, then completes the planned steps with validated0/1/2.

The full run validates at0,2000,every5000 and200640 against all official validation clips, reporting fusion and PLgate0 with identical raw-dictionary ranking. Best selection is raw validationRec1 with earlier checkpoint on ties. Test is not in the training loader or selection loop. Sparse stage checkpoints supplement last/previous/best; validation clip caches are discarded only after metric/checkpoint commit to limit disk use.

## Stage 4: inference and final evaluation

Selected-checkpoint extraction is resumable per clip and checkpoint-bound. Raw window mean then videoL2 is the primary dictionary representation; final windows stride2, validation stride32. Both fusion and PLgate0 have the same768D output. Graph-only vectors are retained as diagnostics without test-optimal primary selection.

The actual ranking implementation uses cosine, per-gloss max and pessimistic ties. Invalid/missing-positive queries have rankinfinity and stay in the denominator; invalid gallery vectors are excluded. Independent known-rank fixtures verify exact ties/invalid behavior. Final secondary whitening fits gallery pooled features only and is labeled separately from historical window whitening. A frozen linear head uses train-only standardization, selects on official validation and then scores test; encoder weights remain fixed. Its cache and incomplete-epoch continuation are tested.

An integration fixture runs the real final orchestration through raw/white512 val/test ranking, selected frozen probes and cached completion. It uses synthetic vectors to test wiring; real pose/extractor behavior is tested separately. No synthetic score is presented as real method performance. Full official test has not yet been scored for this new method.

## Stage 5: resource queue and persistent service

`plctx-pose-fusion-full.service` uses an enabled lingering user service, immutable scientific code/config snapshot, single-supervisor lock, spawn workers, RAM24/32GiB high/max and CPU8core quota/Nice5. It does not stop or inspect unrelated jobs as a prerequisite. At least10240MiB freeVRAM allows joining concurrent GPU work; otherwise it polls15seconds. A CUDA OOM/SIGKILL resource exit requeues from committed state after60seconds. Nonresource code/data failures are terminal and diagnostic rather than retried blindly. Minimum8GiB disk free gates phase launch.

A real subprocess test with an isolated mockGPU query proves that the supervisor stays alive without launching a child under100MiB free, refuses a second supervisor, and starts its short isolated child once free memory rises to20000MiB. It does not launch actual full training in this test. This complements the productionGPU smoke, not a replacement for live service inspection.

Live launch review found Python symlink resolution selected the baseConda executable rather than the explicit `.venv-signrep` entrypoint. The service was gracefully stopped in initial validation atstep0, keeping checkpoint and partial receipts. `setup` now preserves the absolute venv path without resolving its symlink; a regression assertion checks it. Only the ops snapshot/entrypoint was repaired, not scientific code/config/checkpoint. Audit:`runtime/review/venv_entrypoint_repair.json`. The continued process was verified to invoke `.venv-signrep/bin/python`; source/environment proof still matches the checkpoint. This was an operations repair, not an optimizer restart.

## Execution evidence and completion requirements

- Test suite currently has17 meaningful cases, covering the mechanisms above; JUnit and final counts at `runtime/review/tests.xml` and `tests.log`.
- ProductionGPU real-data smoke:batch32,bf16,PL6×256/graph8×256,predictor4×128. Graph/gate gradients active, PL frozen, EMAteacher changed, all scalars finite. ExactCUDA checkpoint continuation produces equal losses and all model state; peakPyTorch allocation2,381,001,216bytes (~2.22GiB). Evidence:`runtime/review/cuda_smoke.json`.
- Final-source actual trainingCLI smoke completed3steps with validation at0/1/2/3, production dimensions/batch32 and spawnworkers2, real data and fitted geometry stats. Smoke flags remain explicit. Evidence:`runtime/review/smoke_verified/train/complete.json` and validation records.
- `scripts/verify_setup.py` rechecks these artifacts, current source hashes, full corpus and pretrained checkpoint provenance, schedule and all14 pinned original source files; `runtime/verification.json` recordsPASS. Package `compileall` is also run.
- The real full service must additionally be inspected for enabled/live state, frozen snapshot integrity, actual child command, current checkpoint/progress and successful validation/optimizer updates before setup is claimed complete. `runtime/full160/` retains the live evidence. Completion means persistent training/evaluation is installed and genuinely underway (or verified live in a resource queue), not that160epochs have already finished.

Scientific limits:one seed; extraPL pretraining budget; fresh graph; no current proof of lexical gain. Gate retention and validation protect baseline comparisons but do not guarantee improvement. Final full-test and frozen-probe reports are separate future outputs automatically chained after training.
