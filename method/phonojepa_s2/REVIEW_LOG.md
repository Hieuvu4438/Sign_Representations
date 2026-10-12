# S2 staged implementation review

## Material Passport

- Date:2026-10-12; scope:proposal12 §7 S2 and the eleven items in `CONTRACT.md`.
- Artifact: code correctness and experiment setup review. This is implementer self-review with mathematical oracles, adversarial tests and actual execution checks; it is not an independent human review.
- Inputs:local proposal, reviewed S1 implementation, audited primary-paper/official-code bundle. No unpublished data uploaded and no external-model review invoked.
- Claim boundary: implementation/setup only. No S2 lexical accuracy, convergence or improvement result is established.

## Stage 1: path mathematics, timestamps and observation support

Reviewed `intervals.py`, dataset resampling/export, target layout and dimensional conventions against proposal §6.1/§7. Fifteen explicit weighted2D paths, scales8/16/32 and stride half the interval give375 slots. The descriptor arrays and metadata share a scale→path→start ordering. Physical differences use actual source-time deltas, independent of output sample count. Lags unavailable within an interval are masked per dimension. Start/end level and duration remain alongside translation-invariant path targets.

Findings addressed:

1. A naive all-joint linear combination can propagate `0*NaN` from an unrelated missing joint to every path. Unobserved/nonfinite joint coordinates are made benign before the weighted sum; each path still has its own strict observation mask. Regression tests perturb an unrelated hand joint toNaN and require every descriptor to remain identical.
2. `t_obs & ~s_obs` conflates raw training masks with stochastic hand dropout. The dataset now exports the exact OR of forbidden raw interpolation supports separately, plus relative source timestamps. Tests reconstruct interpolation supports at15/25/29.97/60fps and check source-time increments.
3. Filling isolated gaps with zeros before taking differences would create artificial motion. A whole interval is gated before path estimation, and all contiguous samples must exist even for lag4/8. Tests cover overlapping scales, internal gaps and benign invalid outputs.

Validation: independent NumPy full-tensor Chen product/logarithm versus Torch free-Lie coefficients in2D/4D/6D; known square area and straight line; translation, reversal, subdivision and affine signed-area transforms; standard lead–lag ordering/dimension10; physical speed versus frame-difference controls. No signature library or third-party implementation was copied.

## Stage 2: masking, queries, model and objectives

Reviewed raw joint ownership, duplicate closure, per-bin query selection and optional token attention masks before interpreting training losses. The primary input is exactly the eight S1 feature channels. An explicit equality test compares S2 token features/presence against the current reviewed S1 source. Targets use clean teacher geometry in the **student affine/reference frame**; their values never enter primary input or query metadata.

Findings addressed:

1. A hidden endpoint anywhere in the window is insufficient to justify supervision at an unrelated masked cell. Query selectors intersect the same interval, same bin, corresponding channel ownership and actual forbidden endpoint. A test places the masked query far from the hidden endpoint and requires zero selectors.
2. Artificial test flags exposed that selector ownership alone did not defensively close duplicate flags. Selectors now explicitly close wrist/nose duplicate flags as well as their owners. Dataset closure already happens before interpolation; tests check both paths.
3. Padding/mode exclusion before normalization would allow a nonzero normalization center to reintroduce unused signature dimensions. Optional input fields are masked **after** normalization. Motion/signature/both tokenizers have identical slots, padded width and parameter count.
4. Hidden optional descriptor content could leak through attention even with a valid base-token mask. Whole-interval validity gates each extra key. Tests perturb invalid descriptor payloads and every raw-hidden coordinate/confidence after a fixed dominance decision; visible encoded tokens and CLS remain exactly unchanged.
5. Unconstrained latent MSE without a collapse control is not a safe contextual-target comparison. The latent interval configuration requires the inherited contextual CE anti-collapse branch. Pure deterministic motion remains a separate control.

Validation: all seven combinations covering plain/lead–lag, motion/latent/dense reconstruction and optional input modes have finite CPU backward passes. Teacher gradients remain absent, relevant heads/tokenizers receive gradients, and EMA changes teacher parameters. Empty target support produces a differentiable zero loss. Group-balance tests ensure confidence/count in one path cannot drown another group. All primary configs have identical total model/head capacity; optional input modes match each other. The active supervised output dimensions differ by objective and are documented.

## Stage 3: normalization, optimizer, resume and real execution

Reviewed train-only fit selection, robust statistics, per-dimension masks, optimizer ownership, checkpoint state, worker sampling and CLI/config validation. Median/IQR parameters and counts are stored per path/dimension. The fixed seed123 sampling is restricted to the verified training cache, with no validation/query fitting. Checkpoints contain interval heads/stats along with EMA, CE center, optimizer and RNG state. Existing worker-independent `(seed,step,draw)` window/batch sampling is retained.

Validation:

- Full-size robust-stat fit on **640 unique real training clips**, lead–lag plus both-input configuration:PASS,7.24s. Every one of15 paths has observations for motion, signature, start/end and dense-motion targets. Student/teacher descriptor normalization buffers match. Evidence:`runtime/review/stats640.json` includes selected training IDs, counts and source/cache provenance.
- Exact CPU continuation after checkpoint, including fitted statistics, logged losses and all interval state:PASS in tests. Worker0 versus two spawn workers yields identical data.
- Final-source CUDA bf16 smoke, **production encoder6×256/heads4, predictor4×128, batch32**, two spawn workers, two optimizer steps, real corpus:PASS for primary/plain and both-input/lead–lag branches. All losses/gradients finite and both views have active interval queries. Peak PyTorch CUDA allocation:2,202,849,792 bytes (~2.05GiB) primary;3,887,478,784 bytes (~3.62GiB) optional input. These exclude allocator reserves and other jobs.
- Additional real-data batch32 CUDA bf16 checkpoint continuation:PASS. From the same step2 checkpoint, uninterrupted versus restored step3 has **exactly equal logged metrics, encoder/teacher/predictor/auxiliary/interval state and CE center**. Evidence:`runtime/review/cuda_resume.json` and reproducible script `scripts/smoke_cuda_resume.py`.

Only the above short verification steps ran. No4k/16k S2 research experiment was launched.

## Stage 4: frozen extraction and evaluation protocol

Reviewed encoder-only loading, descriptor preprocessing/stat buffers, quarter-window readout, gallery-only projection and every guard inherited from S1. Extraction uses64-frame windows at speed1, source timestamps, no augmentation and the checkpoint's chosen weights. Auxiliary heads/predictor are not loaded for inference. Static D/N/face/torso and four ordered temporal means produce2048 raw dimensions; every S2 arm uses this readout and the same768-dimensional gallery projection.

Findings addressed:

1. A checkpoint's source identity could differ from the code currently computing targets/input descriptors. Extraction now rejects mismatched package hashes and rejects protocol checkpoints without source provenance. Resume already rejects scientific-config/corpus/source mismatches. Synthetic rejection tests cover the extraction guard.
2. The static+temporal readout changes the inference representation relative to the earlier S1 run. `ctx_only_4k` has the identical S2 readout and projection, providing the required internal control. README explicitly states that comparison with S1 alone cannot isolate target effects.
3. Toy/subset feature outputs must not become research metric tables. Existing guards reject smoke/partial extractions, mixed split provenance, unprojected primary width, changed rows/order and malformed offsets. The final verifier actually calls the smoke guard, rather than merely checking metadata exists.

Validation:

- Real CUDA/spawn-worker CLI extraction on both val and ASLC300 smoke subsets for **both** primary and optional descriptor branches:PASS. Each branch produced39 val windows and44 ASLC300 windows at width2048. Both sets share checkpoint/source/config identities.
- Gallery-only CLI PCA16 for these small smoke subsets:PASS for both branches. The verifier checks transformed arrays against the saved projection, exact gallery-ID provenance and smoke-evaluation rejection. Production **2048→768** PCA is separately tested with enough gallery rows; the tiny smoke PCA16 is not a primary performance artifact.
- The actual unchanged `method/obs_jepa/oj/evaluate.py` runs retrieval, true linear probe, five geometric probe families, signer/geometry and paired bootstrap in integration tests. Real prior loading is checked separately. Synthetic metric fixtures test protocol wiring, not real accuracy.
- Dataset preflight:PASS; exact3,000 training IDs/order; disjoint validation/query videos;10304 val clips/six signers;3000 gallery/1500 query/300 classes; required keypoint/prior files present. Original evaluator SHA256 remains `77d14bf94a7fb906a884f72d4cea7fc2b26ac4ef2739f6dd0b8f860d7d74ef22`.

Historical LOSO whitening and probe standardization conventions remain documented as transductive. S2's gallery-only PCA does not change that evaluation limitation.

## Completion audit against CONTRACT.md

| Item | Inspected implementation | Authoritative evidence |
|---|---|---|
| 1:unchanged primary input | `features.py`, `train.view_loss`, configs | Exact S1/S2 feature equality test; no descriptor module in primary encoder |
| 2:motion/time controls | `intervals.motion`, exported `times` | Known-velocity/dt tests; frame/single/multilag configs; four-fps support tests |
| 3:selected anatomical paths | immutable `PATHS` supports |15 explicit path specs and real stats coverage for each |
| 4:intervals/gaps | layout/unfold/all-sample observation gate |375-slot shape tests; internal gaps and overlapping-support perturbations |
| 5:degree-two math | `logsig2`, `lead_lag` | Independent tensor oracle + geometric/metamorphic tests |
| 6:static/timing/query metadata | level branch; duration/start/scale/path metadata | Start/end gradients; metadata-only query implementation reviewed |
| 7:raw support/closure | dataset forbidden export; selectors; descriptor validity | Exact support reconstruction; duplicate/same-bin tests; raw-hidden perturbation invariance |
| 8:loss/stats/controls | interval heads/trainer;16 configs | Robust per-path stats tests;640-real-clip fit; gradients, balancing and all target types |
| 9:optional input | descriptor tokenizer + appended attention slots | All modes train; capacity/post-normalization/key exclusion tests; production CUDA both/lead–lag |
| 10:pose inference/evaluation | extractor/project/evaluate | Both real extraction/projection branches;2048→768 test; unchanged evaluator and guards |
| 11:review and execution | staged review; tests/scripts/artifacts |55 passed,0 failures/errors/skips; CPU/CUDA smokes; exact CPU/CUDA continuation; source/protocol verifier |

Final test evidence:`runtime/review/tests.xml`, **55 passed**,12.37s. `compileall` passes. All17 S1 source files pinned in `UPSTREAM.json` retain their original hashes. `scripts/verify_setup.py` rechecks current hashes, real checkpoints,640-clip stats, config matrix, finite extracted/projected arrays, split identity and smoke guard, writing `runtime/verification.json`.

Remaining scientific work is the full equal-budget/multiple-seed training and frozen evaluation, followed by deciding whether motion/signatures help lexical metrics. This is separate from the completed implementation/setup objective. Strict contiguous support, uncalibrated estimator confidence, inherited dominance/anchor side information and potential jitter remain explicit experiment limitations.

## Authorized full-run follow-up,2026-10-12

After the implementation/setup review above, the user requested full S2 training and evaluation. Ran `signature_4k.json` from scratch in tmux `sign_s2`, run `S2_FULL_4k`: contextual CE + motion(.1) + plain2D degree-two log-signature/start-end(.1), unchanged eight-channel input, batch32,seed0,4000 steps. No audited package source was changed.

All five phases exited0:train677s including initialization, ASLC300 extraction65s, val extraction114s, projection7s, evaluation115s; total979s (~16.3min). Training itself logged666s. Full features contain4500 ASLC300 clips/6056 windows and10304 val clips/13490 windows, all projected to768 with gallery-only PCA. Final checkpoint is `runtime/runs/S2_FULL_4k/final.pt`.

Full artifact verification is PASS at `runtime/eval/S2_FULL_4k_verification.json`. Results:LOSO R@1=7.20%;ASLC300 R@1=15.73%,R@5=35.60%,DCG=.3993,LP macro=36.13%. Versus S1_FULL_4k, differences are−3.29pp LOSO,−7.60pp ASLC300 retrieval and−7.87pp LP; signer-cluster CI95 excludes0 for all three conditional comparisons. One seed and differing readout/auxiliary objectives prevent attributing the change specifically to signatures. Movement probe R² remains negative (−.07390); no movement/lexical benefit is established by this run.

See `runtime/eval/S2_FULL_4k_REPORT_VI.md`, full JSON and summary CSV for every metric family, references, paired confidence intervals and limitations. Multiple-seed and matched context-only/motion-only ablations remain separate scientific work, not part of the completed single-run request.
