# PhonoJEPA S2

Implementation of proposal12 §7 S2: contextual EMA JEPA with multiscale motion supervision, then optional degree-two path log-signature supervision. This is an explicitly defined adaptation of the research ideas, **not an official Path-JEPA/MAMP/MASA reproduction**. Package, runtime and outputs are isolated from S1. Full scratch signature4k training and full evaluation completed on2026-10-12; see [the Vietnamese result report](runtime/eval/S2_FULL_4k_REPORT_VI.md).

## Material Passport

- Artifact: experiment implementation and setup; 2026-10-12.
- Specification: [Vietnamese proposal](../../docs/proposal12/PhonoJEPA_Strategies_Deep_Research_VI.md), S2; [implementation contract](CONTRACT.md).
- Design/code provenance: [SOURCES.md](SOURCES.md), S1 source snapshots in [UPSTREAM.json](UPSTREAM.json).
- Validation and limitations: [REVIEW_LOG.md](REVIEW_LOG.md); machine evidence under `runtime/review/` and `runtime/verification.json`.
- Accuracy status: `S2_FULL_4k` completed all five full-protocol phases. LOSO R@1=7.20%, ASLC300 retrieval R@1=15.73%, LP macro=36.13%; this configuration underperforms S1_FULL_4k. Smoke outputs remain rejected by full-protocol evaluation.

## Input, model and targets

The primary student input is **exactly S1's eight channels**: D_CONF, D_LOC, N_CONF, N_LOC, F_UP, F_MO, F_HD, TORSO. A sample has 64 resampled frames, two frames per bin, 256 base tokens plus CLS. The scratch encoder is six Transformer blocks, width256, four heads. The masked predictor is four blocks, width128. No pretrained weights are used. The primary encoder receives no motion/signature target values.

There are 15 selected 2D paths, with D/N side convention inherited from S1:

| Group | Paths |
|---|---|
| Absolute hand | Body wrists and index tips, both sides |
| Arm | Wrist minus elbow; elbow minus shoulder; elbow minus midpoint(shoulder,wrist), both sides |
| Shape proxy | Index tip minus hand wrist, both sides |
| Relative placement | Wrist minus head, both sides; dominant wrist minus nondominant wrist |

The chain-bend residual consumes all three arm joints; it is a 2D derived trajectory, not a claim to reproduce Path-JEPA's chain-path construction. Index extension/orientation is a limited dynamic shape proxy, not a replacement for the full CONF channels.

Each path has intervals of **8/16/32 samples**, with half-interval stride: 15+7+3=25 intervals per path, **375 slots**. An interval requires every supporting joint at every sample, including intermediate samples of lagged differences. No path connects separated observations or zero-filled gaps. Student descriptors additionally require no forbidden raw source anywhere in the interval. The exact raw interpolation dependency mask is exported by the dataset before augmentation; stochastic observation dropout is separate.

Motion target width is fixed at12:

1. For lags1/4/8: mean signed 2D displacement/Δt and mean displacement magnitude/Δt (nine slots).
2. Total path length, mean step speed and mean absolute radial change/Δt (three slots).

`Δt = lag * resampling_speed / min(source_fps,25)` in source seconds. The frame-difference control omits division by Δt. Single-lag controls zero and mask the unused slots without changing head shape. Lag8 is invalid for an eight-sample interval. Radial change has an extension interpretation on the shape paths; for absolute paths it measures distance change relative to the body origin.

`logsig2` implements the exact degree-two piecewise-linear log-signature in the free Lie basis: first-level increments, then signed areas in lexicographic pairs i<j. Plain2D has **3 coefficients**, padded to10 with seven inactive slots. The standard axis-aligned lead–lag lift `(x0,x0),(x1,x0),(x1,x1),…` is 4D with **10 coefficients**. There is no scalar coefficient. Formula and tensor-algebra oracle are independently implemented and tested. Start/end coordinates have a separate four-dimensional head so translation-invariant signatures do not erase static placement. Duration is query metadata, not a target to be trivially copied.

Queries pool only masked base-predictor cells whose channel owns a forbidden endpoint **in the same bin and interval**, including duplicate wrist/nose closure. They add learned path/scale embeddings and metadata `(start/64, length/64, log(duration))`. Target coordinates do not enter queries. Teacher geometry uses the student's affine transform and the shared allowed shoulder reference; contextual teacher features retain teacher augmentation.

## Objectives and controls

Default objective:

```text
λctx Lctx + λmotion Lmotion
  + λsignature (Llogsignature + 0.25 Lstart_end)
  + optional λbody Lbody + λrelations Lrelations
```

Defaults: λctx=1, λmotion=.1, λsignature=0, λbody=λrelations=0. Proposed weights are starting points, not tuned findings. Contextual centered CE, temperature, center update, EMA and gradient clipping inherit S1. Latent-interval supervision requires contextual CE to retain its anti-collapse mechanism. Pure deterministic motion is a separate permitted control.

Geometry targets use train-only median/IQR normalization per path and dimension, IQR/1.349 scale, floor .01 for signature and .05 otherwise, then clipping to ±8. Buffers store observed training counts; empty dimensions use center0/scale1 and remain inactive where invalid. Normalization samples 640 reproducible clean training windows, transformed into the student view frame. No validation fitting occurs. Huber beta.5 is used for motion/signature/start-end; losses average dimensions → confidence-weighted intervals → paths → four equally weighted path groups.

The `latent` interval control predicts normalized pooled clean contextual teacher features. The `mae` control predicts **all first differences** in the interval with MSE, padded to62 output slots and masked beyond the true length. It is a matched-input motion reconstruction control inspired by MAMP, not its official architecture or a literal raw-coordinate MAE. It uses the same train-only normalization and query validity as the deterministic target arm.

All primary controls allocate the same interval trunk and heads and use the same base input, mask, optimizer, batch, inference readout and total parameter count. The active output head/target dimension differs by objective. Compare against `ctx_only`, not only the older S1 run, because S2's temporal readout is itself a change.

| Config (`configs/<name>.json`) | Factor |
|---|---|
| `ctx_only_4k` | Contextual baseline, interval weights0 |
| `motion_4k` | Primary: lags1/4/8 with physical Δt |
| `raw_diff_4k` | Lag1, frame differences |
| `single_lag_dt_4k` | Lag1 with physical Δt |
| `multilag_frame_4k` | Lags1/4/8 without Δt division |
| `signature_4k` | Primary + plain2D log-signature/start-end |
| `leadlag_4k` | Primary + lead–lag4D log-signature/start-end |
| `latent_interval_4k` | Contextual interval targets |
| `motion_mae_4k` | Dense motion reconstruction |
| `intrinsic_motion_4k` | Deterministic motion, λctx=0 |
| `motion_input_4k`, `signature_input_4k`, `both_input_4k` | Optional descriptor-input factor; identical signature supervision |
| `motion_s1_aux_4k` | Primary + S1 body/relations auxiliaries |
| `motion_16k`, `signature_16k` | Explicit larger budget |

The optional descriptor input adds 375 interval tokens with a shared width27 tokenizer: motion12 + signature10 + start/end4 + duration1. Motion-only/signature-only/both input arms have identical token slots, parameters and padded width; unused fields are zeroed **after** normalization. Static endpoints and timing remain in all three. Forbidden or unobserved intervals are zeroed and excluded as attention keys. The teacher may use clean descriptor tokens. This is an additional input factor and must be compared separately from the unchanged-input primary arms. There is no duplicated global descriptor broadcast to visible local tokens.

## Run and verify

From `/home/haipd/Sign_Representations`, use the tested `.venv-signrep` environment. No extra signature dependency or compiled extension is required.

```bash
PYTHONPATH=method/phonojepa_s2 .venv-signrep/bin/python -m s2.prepare
PYTHONPATH=method/phonojepa_s2 .venv-signrep/bin/python -m pytest method/phonojepa_s2/tests -q
PYTHONPATH=method/phonojepa_s2 .venv-signrep/bin/python method/phonojepa_s2/scripts/verify_setup.py
```

Primary motion-only experiment command, prepared as a separate ablation; the completed full run instead used `signature_4k.json` with output `runtime/runs/S2_FULL_4k`:

```bash
PYTHONPATH=method/phonojepa_s2 .venv-signrep/bin/python -m s2.train \
  --config method/phonojepa_s2/configs/motion_4k.json \
  --out method/phonojepa_s2/runtime/S2_MOTION_4k
```

Use another config and a fresh output directory for each arm. `--resume` continues the same directory at the saved step, restoring optimizer, EMA, robust statistics, CE center, RNG and deterministic data stream. It rejects changes to scientific config, corpus/cache or package source. Changing4k→16k is a new experiment, not resume. All datasets use exactly the verified ASLC002 3,000-video unlabeled corpus; gloss/signer labels are not read during training.

Frozen inference uses only the chosen teacher/student encoder, including descriptor tokenizer when configured; predictor and auxiliary heads are discarded. Readout concatenates static D/N/face/torso means with four chronological quarter-window means, **8d=2048** at production width. The gallery-only projection maps this to768; every S2 control uses this same readout. Mean and CLS outputs are also saved.

```bash
PYTHONPATH=method/phonojepa_s2 .venv-signrep/bin/python -m s2.extract \
  --ckpt method/phonojepa_s2/runtime/S2_MOTION_4k/final.pt \
  --name S2_MOTION_4k --set aslc300
PYTHONPATH=method/phonojepa_s2 .venv-signrep/bin/python -m s2.extract \
  --ckpt method/phonojepa_s2/runtime/S2_MOTION_4k/final.pt \
  --name S2_MOTION_4k --set val
PYTHONPATH=method/phonojepa_s2 .venv-signrep/bin/python -m s2.project --name S2_MOTION_4k
PYTHONPATH=method/phonojepa_s2 .venv-signrep/bin/python -m s2.evaluate \
  --names S2_MOTION_4k_grp --protocol both \
  --out method/phonojepa_s2/runtime/eval/S2_MOTION_4k.json
```

Extractor requires the checkpoint's audited package source hashes. Feature metadata records checkpoint, source, weights and config; projection/evaluator reject mixed identities and partial/smoke runs. Evaluate both protocols together on a fresh feature name, or use `--force` when widening an existing evaluator cache.

The evaluator itself remains `method/obs_jepa/oj/evaluate.py`: LOSO retrieval, ASLC300 retrieval/true LP, five geometric probe groups, signer/geometry and paired bootstrap. ASLC300 gallery3000/query1500; val10304/six signers. Its historical LOSO whitening uses unlabeled validation windows and its geometric-probe standardization predates fold splitting; these conventions remain explicitly transductive. The additional S2 PCA is gallery-only and does not make the inherited evaluation fully inductive.

## Practical limits

Strict contiguous support can reduce interval coverage, especially at scale32 or under occlusion; logs report active query counts and statistics preserve per-path fitting counts. Shoulder anchors and whole-clip dominance are declared side information inherited from S1. Signatures are translation-invariant but are not universally invariant to frame subsampling, pose jitter, changed interpolation or time augmentation. Static/timing branches reduce information loss without proving lexical improvement. Smokes are execution checks, not accuracy, convergence or throughput estimates.
