# S1 staged review

## Material Passport
- Artifact: implementation/reproducibility review, 2026-10-11
- Scope: proposal12 S1; ASLC002 3,000 training clips; no claims of accuracy improvement
- Inputs: local PhonoJEPA source and audited primary sources in proposal12 research bundle

## Stage 1 — inputs and raw-support masking

Decision: shoulders are explicitly always-visible side information. They enter the stable window reference frame and the TORSO level/delta input, but are never counted as masked auxiliary body targets. Hidden torso content is body nose and hips. This chooses the always-visible-anchor option expressly permitted by S1; it does not implement shoulder reconstruction.

Review findings addressed:
1. Baseline no-shoulder fallback read all observed points: replaced by a deterministic origin/100-pixel scale/zero angle. Hidden points cannot influence it.
2. Body wrist/hand wrist and body nose/face nose are duplicate observations. Added bidirectional raw scrubbing closure and propagated context drops through channel dependencies.
3. Cross-bin temporal differences can reveal forbidden sources: torso input differences stay inside the two-frame bin; both observations must be valid. Temporal context is handled by the Transformer.
4. Mirroring without shoulders previously read a full-body centroid. Changed the fallback mirror center to zero. Whole-clip handedness decision remains declared side information, as in baseline.

Validation: `python -m pytest tests/test_data.py -q`: 5 passed. Covers 40 mask/perturbation cases, feature invariance to arbitrary hidden source coordinates/scores, duplicate closure, absent-shoulder reference, missingness, torso feature dimensions and real clips from the 3,000-video cache. Later stages must also check encoded outputs and gradient ownership.

## Stage 2 — model, explicit targets, loss, optimizer and checkpoints

Review findings addressed:
1. An independently augmented teacher frame is not a well-defined coordinate target for student-view geometry. Auxiliary targets use the shared allowed shoulder reference and the **student affine**; contextual JEPA targets retain the baseline teacher augmentation.
2. A relation head on an unrelated masked query can simply copy visible coordinates. Loss dimensions now require an endpoint hidden for D_LOC/N_LOC/TORSO query ownership. Body heads exclude shoulders, respect observation/confidence masks and require both observations for displacement targets.
3. Baseline worker RNG/shuffle state was not enough to reproduce an interrupted batch stream. New sampling uses (seed, optimizer step, draw) and a deterministic per-step batch sampler; checkpoint contains full optimizer, auxiliary stats, center, model and RNG states. Resume rejects changed corpus, source hashes and scientific config.
4. Input-only/capacity/shuffled controls reject nonzero auxiliary weights. Capacity keeps exactly the same model/slots while replacing torso evidence with a constant. Shuffled torso uses a minibatch derangement during training; inference donors come from a fixed hash-selected training clip.
5. Review of interpolation found `0 * NaN` could invalidate a frame whose missing neighbor had zero interpolation weight. S1 now excludes the unused neighbor explicitly; added a regression test.

Validation before interpolation follow-up: 10 tests passed; CPU 3-step real-data training completed with finite losses and nonzero body/relation query counts. CUDA bf16 smoke with production 6x256 encoder / 4x128 predictor completed 2 steps on real clips (batch2, 2 workers). Teacher has no gradients, torso tokenizer and auxiliary heads receive gradients, EMA changes teacher weights, and checkpoint continuation matches uninterrupted CPU parameters/losses exactly. These are execution checks, not accuracy evidence. Final-stage audit will rerun tests after all fixes.

## Stage 3 — protocol, frozen extraction, projection, evaluation and setup

Review findings addressed:
1. Adding torso makes raw group pooling 1024-dimensional versus baseline768. Added a separate PCA projection fitted on ASLC300 gallery **train** windows only and applied unchanged to all validation/query windows. Projection parameters/hash are saved; evaluator refuses an unprojected primary readout.
2. Smoke subsets can accidentally produce misleading comparison tables. Feature validation rejects smoke checkpoints, partial extractions, changed split rows/order and malformed offsets. Actual shared evaluator functions still run unchanged; only feature/cache roots are redirected.
3. Independent sampling with replacement across training batches would introduce another recipe difference. Replaced it with epoch permutations and drop-last; per-step/draw view seeds and batch-stream resume remain deterministic.
4. Kept the historical LOSO transductive whitening and probe standardization conventions expressly documented. S1 projection itself is gallery-only; this does not turn the inherited protocol into a fully inductive one.

Validation to this point: 16 tests passed, including actual evaluation function dispatch through every metric family, gallery projection insensitive to arbitrary query replacement, real keypoint-prior loading and real frozen extraction for real/capacity/shuffle inputs. Setup preflight confirms 3,000 training clips, disjoint val/query, 10,304 val clips/6 signers, 3,000 gallery +1,500 query/300 classes, all keypoint files present and prior stats available. Real CPU CLI extraction succeeded on val and ASLC300 smoke subsets; CLI gallery-only projection is checked in the final audit. Full performance evaluation remains a subsequent experiment after full training.

## Stage 4 — final implementation/setup audit

Final follow-up fixes:
- Feature artifacts now include checkpoint and extraction-source SHA256. Projection checks both splits before creating outputs and rejects mixed checkpoints/weights/config/source; evaluator repeats the identity check.
- Python 3.13 warned about forking a threaded process during the first worker-parity test. Training/extraction now use explicit `spawn` with picklable worker functions. Final test run has no warnings; worker0/worker2 data are identical.
- Added a production-size 1024→768 PCA test, mixed-checkpoint rejection, input-only/control optimizer tests, and actual paired-bootstrap evaluator dispatch (self-pair gives exactly zero differences and intervals).

Final validation:
- `python -m pytest tests -q`: **22 passed, 0 failed, 0 skipped**, 6.50s; JUnit at `runtime/tests.xml`.
- `python -m s1.prepare`: **PASS**, all required data/protocol assets present, same 3,000 training IDs/order, no training/evaluation video overlap, full evaluation split counts verified.
- `python -m compileall -q s1`: passed.
- Final-source CUDA bf16 smoke: production encoder6×256/heads4, predictor4×128, **batch32**, 2 workers using spawn, 2 optimizer steps; finite losses/gradients, both auxiliary groups active; checkpoint/manifest complete. Peak PyTorch CUDA allocation: 2,081,866,752 bytes (~1.94 GiB); this excludes other processes/CUDA allocator reserves and is not a throughput benchmark.
- Final-source real keypoint frozen extraction CLI succeeded for both val and ASLC300 smoke subsets using CUDA +2 spawn workers; raw grouped width1024. Gallery-only projection CLI succeeded (dim4 for the tiny smoke subset). Full primary dim768 is covered separately by the production-size projection test. No synthetic labels are used in real training/extraction checks.
- The unchanged evaluator ran retrieval, true linear probe, all five geometric probe groups, signer probe, feature geometry and paired bootstrap in integration tests. Synthetic metric fixtures verify wiring; real prior loading is checked separately. This is not a full research evaluation or a performance improvement claim.
- `scripts/verify_setup.py` verifies the requirement matrix, final source hashes, manifests/checkpoint, real extraction/projection artifacts and smoke-evaluation rejection, writing `runtime/verification.json`.
- All 13 original source files in `UPSTREAM.json` retain their original hashes. Existing baseline scheduler process remains untouched.

Implementation and setup are complete. Full 4k/16k training and full frozen performance evaluation have not been launched; the six configs and README commands are ready. No claim that S1 beats PLctx16k follows from these execution checks.
