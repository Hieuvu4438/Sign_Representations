# PhonoJEPA S1 — torso input and explicit geometric targets

Implementation of [proposal12 S1](../../docs/proposal12/PhonoJEPA_Strategies_Deep_Research_VI.md#s1--giữ-conf-thêm-torso-và-explicit-relations-với-thay-đổi-nhỏ). This directory is independent of existing training jobs. All generated artifacts go under its `runtime/`; datasets and other methods are read only. S1 is an experimental hypothesis, not an established accuracy improvement.

## Setup and data

Run from the repository root. The existing `.venv-signrep` has the tested dependencies; no installation or modification of that environment is required:

```bash
export PYTHONPATH="$PWD/method/phonojepa_s1${PYTHONPATH:+:$PYTHONPATH}"
.venv-signrep/bin/python -m s1.prepare
.venv-signrep/bin/python -m pytest method/phonojepa_s1/tests -q
```

For a separate environment, install `method/phonojepa_s1` with its `[test]` extra and a suitable PyTorch build. `requirements-tested.txt` records the actual tested versions, including CUDA 12.8 PyTorch. This package relies on sibling evaluator modules and the local dataset layout; it is not a standalone dataset distribution.

The default cache is `method/phonojepa/runtime/cache/aslc002_train.npz`. It has exactly the same 3,000 ASL-Citizen training videos, in the same order as `method/jepa_baseline/runtime/ASLC002/corpus/pretrain.jsonl`. Training rejects a changed ID list, clip count or order. Labels/signers are absent from the cache. Preflight checks that the ASLC300 gallery is exactly this training set and that validation/query videos are disjoint. It also checks keypoint files, prior stats and the evaluator hash. If rebuilding is necessary, `python -m s1.data.cache` writes a **new S1-local cache**; pass its path with `--cache` to prepare/train. Existing cache files are not modified.

## Exact input and modeling contract

- Raw: 103 selected COCO-WholeBody 2D keypoints (9 body, 52 face, 21+21 hands), quality scores and observation flags. Normalized image coordinates are converted back to pixels before geometry. No RGB/text/gloss is input to S1.
- Seven inherited channels: D_CONF/D_LOC/N_CONF/N_LOC/F_UP/F_MO/F_HD. `CONF` means configuration, including orientation. Eight-channel S1 adds TORSO: body nose, shoulders, hips; per-point 2D level and displacement within the two-frame bin, with confidence. Its input is 50-dimensional per bin: 2 frames × (10 level + 10 displacement + 5 scores). Missing points are zeroed; scores/validity distinguish absence. No subtraction of the entire static torso pose.
- Canonical dominant-hand mirroring follows baseline. The whole-clip handedness bit is declared side information and is **not** claimed to be independent of hidden content. Window start/speed/masks are content independent. Windows contain 64 samples at min(fps,25), training speed 0.8–1.25, exactly two frames concatenated per token.
- **Anchor policy:** shoulders remain always visible, including their time series. Their window median center/width/angle defines the body reference, shared between teacher/student. They may appear in TORSO inputs but are excluded from hidden explicit body targets. Nose/hips are scrubbed for torso tubes. Missing-shoulder fallback is fixed (origin,100-pixel width,zero angle), never an all-joint centroid. This implements the proposal's declared always-visible-side-information option, not shoulder reconstruction.
- Raw masks are applied before interpolation/features. Duplicate body/hand wrists and body/face nose are scrubbed together; dependency closure drops any visible token that reads forbidden sources. Torso differences never read across bin boundaries. Existing PL near/far masks are retained with a 25–50% temporal torso tube in each view. Added torso targets and duplicate closure change realized visibility; logs expose hidden, visible and context-drop fractions among valid slots. This is an explicit S1 adaptation, not bit-for-bit baseline masking.
- Tokenizer MLP → d256; 32 bins × 8 channels = 256 slots plus CLS. Encoder: 6 Transformer blocks, 4 heads, FFN1024, temporal RoPE. Predictor: 4 blocks d128. EMA teacher has the same encoder, no gradient. Scratch training; no external pretrained weights. Hidden slots are excluded as keys, and predictor queries contain channel/time metadata only.

## Objectives

`L = L_ctx_near + L_ctx_far + λ_body(L_body_near+L_body_far) + λ_relations(L_rel_near+L_rel_far)`.

- Contextual targets average normalized outputs of the last four teacher layers. Centered CE uses student temperature .1, teacher .04→.06, center EMA .9. Group weights are hands1/face.5/torso1, each reduced independently.
- Body targets: masked nose/hip levels and within-bin displacements, 12 dimensions. Shoulders are not auxiliary targets. Displacement requires both observations.
- Relation targets: D wrist–hip midpoint, N wrist–hip midpoint, D wrist–body nose, N wrist–body nose, D–N wrist. Each has a 2D vector plus distance: 15 dimensions. These are geometric location proxies, not linguistic labels or 3D contact estimates. Arm-angle targets are deferred; elbows remain in LOC inputs.
- Auxiliary heads consume predictor outputs for D_LOC/N_LOC/TORSO query slots. A supervised relation dimension requires a hidden endpoint associated with that query; unrelated visible geometry is not scored as a masked prediction. Observation/confidence masks gate dimensions. Smooth-L1 beta .5, dimension mean then valid-query mean; zero valid targets give differentiable zero loss.
- Each auxiliary group has training-only per-dimension mean/std, fit on the same deterministic 640-clip subset, with std floor .05 and normalized target clamp ±8. Geometry targets use the **student** affine and the shared shoulder frame, avoiding an unknown teacher affine in coordinate supervision. Fixed λ=.1 are initial experimental choices, not paper-validated optima.

## Training and controls

```bash
.venv-signrep/bin/python -m s1.train \
  --config method/phonojepa_s1/configs/s1_full_4k.json \
  --out method/phonojepa_s1/runtime/runs/S1_FULL_4k
```

Use `--resume` with the same command/output to continue an interrupted checkpoint. Resume rejects scientific config/data/source changes and restores optimizer, center, normalization, EMA and RNG state. Worker-independent data seeds use (seed, optimizer step, draw); each epoch uses a seeded permutation with drop-last, like the comparison protocol. Checkpoints are atomically replaced. Nonfinite loss/gradient aborts without silently skipping an optimizer step.

AdamW lr3e-4, betas(.9,.95), wd.05 except biases/norms/embeddings, 10% warmup then cosine to 1%, grad clip1, CUDA bf16, EMA .996→.9999. 4k×32 clips for screening; 16k×32 for comparison with PLctx16k. Both use the same 3,000 unique videos, not 128k/512k unique training samples. Configurations:

| Config | Torso evidence | Auxiliary targets | Purpose |
|---|---|---|---|
| `s1_input_4k.json` | real | off | input effect |
| `s1_full_4k.json` | real | body + relations | additional supervision effect |
| `s1_capacity_4k.json` | constant | off | same slots/architecture, no additional torso content |
| `s1_shuffle_4k.json` | mismatched donor | off | whether paired torso content matters |
| `s1_input_16k.json`, `s1_full_16k.json` | real | off / on | 16k budget confirmation |

Capacity/shuffle controls must use λ_body=λ_relations=0. Shuffle uses a minibatch cyclic derangement of normalized torso tokens in training; inference uses a deterministic hash-selected training donor, without labels. It is a mismatch control, not a causal importance estimator. Run matched seeds and budgets for comparisons; old PLctx checkpoints remain external references. Warm-start is deliberately not supported in the primary scratch S1 implementation.

A small diagnostic run:

```bash
.venv-signrep/bin/python -m s1.train --smoke --device cpu \
  --out method/phonojepa_s1/runtime/smoke/example \
  --d 32 --depth 2 --dp 16 --pdepth 1 --batch 2 --steps 3 --stats-clips 4 --workers 0
```

Smoke runs still read the verified 3,000-video corpus but have shortened compute/normalization and are marked non-protocol. Their features are rejected by the reporting CLI.

## Frozen extraction and matched evaluation

```bash
.venv-signrep/bin/python -m s1.extract --name S1_FULL_4k \
  --ckpt method/phonojepa_s1/runtime/runs/S1_FULL_4k/final.pt --set aslc300
.venv-signrep/bin/python -m s1.extract --name S1_FULL_4k \
  --ckpt method/phonojepa_s1/runtime/runs/S1_FULL_4k/final.pt --set val
.venv-signrep/bin/python -m s1.project --name S1_FULL_4k
.venv-signrep/bin/python -m s1.evaluate --names S1_FULL_4k_grp --protocol both \
  --out method/phonojepa_s1/runtime/eval/S1_FULL_4k.json
```

Teacher weights are used by default, no train-time masking/augmentation, T64/stride32, same source-span convention as baseline. Extraction produces `_grpraw` (D/N/face/torso concat1024), `_mean`256, `_cls`256. Projection **fits only gallery train windows**, transforms both sets to `_grp`768, and saves mean/components plus hashes. It does not whiten. Primary comparison uses `_grp`768 to match baseline dimensions; evaluator whitening stays unchanged.

The implementation imports the actual `method/obs_jepa/oj/evaluate.py`, redirects only output/input feature roots to S1, and reuses its prior/probe code. Metrics:

- ASL-Citizen val: all 10,304 clips / 6 signers; LOSO retrieval R@1/R@5/MRR/DCG and macro metrics, both historical white512 and mean readouts.
- ASLC300: 3,000 train gallery / 1,500 val queries / 300 classes. Gallery-fitted white512 retrieval and true logistic linear probe with gallery-signer holdout choice of C; macro and instance Top-1.
- Five ridge probe groups: handshape, body, location/relations, non-manual, movement; signer probe and feature geometry (mean cosine/effective rank).
- Paired signer-cluster bootstrap vs available reference models, including PLctx16k.

Validation rejects partial extraction, smoke training, changed split rows/order, malformed offsets, unprojected group readouts, or mismatched primary dimensions. Existing evaluation conventions are preserved for comparability: LOSO whitening fits on evaluated val windows (label-free transductive), and historical probe X standardization uses the evaluated windows before signer folds. They are **not** claimed to be a strictly inductive train-only evaluation; changing those conventions would require rerunning every baseline. No test split is accessed.

## Source reuse and review evidence

`UPSTREAM.json` hashes the local baseline source files at fork time. `SOURCES.md` distinguishes design inspiration from code reuse. `REVIEW_LOG.md` records stage findings/fixes and validation. `runtime/setup_audit.json`, smoke manifests/logs/checkpoints, and `runtime/verification.json` record execution evidence. Tests cover hidden raw perturbation, duplicate closure, absent anchors, interpolation, actual train cache, encoder visibility, auxiliary validity, gradient/EMA ownership, exact checkpoint continuation, projection leakage, real extraction/priors and all evaluator metric families.

Full research training has not been launched by setup. Smoke losses/metrics do not establish better sign representations; improvement requires the matched full runs, controls and multiple seeds.

To recheck the delivered setup evidence after running the supplied diagnostics:

```bash
.venv-signrep/bin/python method/phonojepa_s1/scripts/verify_setup.py
```

This audit intentionally requires the shipped real-data smoke artifacts and final-source test report. It checks implementation/setup completion, not experiment accuracy. Production batch32 CUDA smoke used about 1.94 GiB of PyTorch allocations; allow additional CUDA/runtime memory. Workers use `spawn` to avoid CUDA/threaded-process fork hazards.
