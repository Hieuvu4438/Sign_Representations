# SignRep reproduction contract

The requested directory name is `jepa_baseline`; the implemented method is **SignRep**,
not a JEPA adaptation. Source authority: ICCV 2025 main paper §§3–8 and supplementary
A1–A8, then the public SignRep checkpoint/code. No substitute 2-D pose targets.

## Scope and acceptance evidence

| Stage | Required deliverables | Before implementation | After implementation |
|---|---|---|---|
| S0 | Source pins, scope, uncertainty register | Paper/code comparison | File hashes, dependency audit |
| S1 | All upstream architecture modules; checkpoint-safe wrapper; encoder tokens; 64-frame inflation | State-dict/schema review | Strict released-checkpoint load; direct upstream output parity; inflation oracle |
| S2 | Original 3-D teacher; frame-aligned extraction; dataset bone statistics; FK reconstruction; nine priors + activity; target cache | Skeleton/angle/confidence review | Real teacher load/extraction; FK and geometry invariants; missingness tests |
| S3 | Smooth-L1 reconstruction; variance/covariance; Gram style; discriminator; adversarial objective | Equation/gradient review | Independent numerical and gradient oracles; adversarial direction and EMA tests |
| S4 | Manifests/dedup; paired video sampling; coherent augmentation; optimizer/scheduler; pretraining CLI; resumable state; validation selection | Data/recipe/resume review | Actual paired-data training/backward/resume smoke; schedule and leakage tests |
| S5 | Frozen feature extraction, weighted retrieval/metrics, recognition/class-distribution, translation feature adapters | Protocol review | Manual ranking/metric oracles; recognition training smoke; upstream integration/schema checks |
| S6 | CLI/config/docs; provenance; complete pre/post review record | Requirement audit | End-to-end smoke and requirement-by-requirement completion audit |

Review is performed inline, with source inspection and written findings before each
stage. Tests cannot prove historical equivalence where information is absent.
Actual benchmark-score reproduction and full 500k-step training must be distinguished
from module correctness. Neither synthetic tests nor a checkpoint load proves scores.

## Paper-defined invariants

- Video Hiera-B: 16 consecutive input frames at 224×224; embedding 768; drop path 0.1.
- Decoder: pooled tokens → LayerNorm → bias-free linear 768 → Conv1d 768→512,
  GELU, ConvTranspose1d 512→384 with kernel 16 → per-frame linear projections.
- Body target: 61×3 (19 body/face + 21 right + 21 left). Wrist-centred hand targets:
  21×3 each. Angle targets: body 22×2, each hand 41×2, sin/cos of teacher Euler
  parameters, **not arbitrary three-point bend angles**.
- Distance targets: hand 5×11×3; interaction 12×22×3, with A1/Fig.2 indices.
- Standard bone lengths are corpus medians on pretraining data; reconstruct pose
  using teacher FK. Do not use the demo's per-video median as corpus statistics.
- Activity: low position relative to body indices [0,3,6,7] AND sum of temporal
  standard deviations of visible hand coordinates <0.26 denotes inactive.
- Valid keypoints must have confidence >0.5. Missing targets contribute no loss.
- Reconstruction uses SmoothL1 for all priors including activity. Table 9 weights:
  angles/keypoints 10, distances 20, activity 0.2. Hand keypoint targets scale 2;
  hand distance targets scale 4; other target scales 1.
- Variance: sample std (N−1), sum hinge max(0,1−std). Covariance: denominator N,
  sum squared off-diagonal entries. Weights: 1, 0.004, adversarial 2.
- Discriminator: style×100 → two linear layers (hidden/output 768), GELU and
  LayerNorm; concatenate embedding → four linear layers, hidden 768, output 1,
  spectral normalization, GELU except last; BCE. Matched/unmatched EMA momentum 0.1.
  Encoder adversarial objective is equations 7–9, gated on matched EMA > unmatched.
- Pretrain: 500000 optimizer steps, batch 20, mask 0.8; AdamW LR 1e−4, betas
  (0.9,0.95), weight decay 0.5, clip norm 1; warmup 50000 then cosine, layer decay
  0.85. Discriminator Adam LR 1e−4, betas (0.5,0.9), weight decay 1e−3.
- Augment: Planckian jitter, random-resized crop 256→224, Gaussian blur, grayscale.
  Two independently sampled 16-frame sequences per video; paired items in each batch.
- Exclude WLASL video IDs from pretraining. Select best pretrain checkpoint using
  WLASL validation retrieval every 25000 steps; never select on test.
- Retrieval: windows 16, stride 2 (Table 5 ablations use 8); activity-weighted or
  unweighted pooling, L2 normalization, closest train-dictionary video by cosine.
- Recognition: 64 frames; patch kernel (7,7,7), stride (8,4,4), padding (3,3,3);
  transfer old temporal weights [0,1,2] to new [1,3,5], zero elsewhere. Linear
  classifier, CE smoothing 0.1 + class-distribution KL κ=0.2; 100 epochs, batch 8,
  1000 warmup iterations, AdamW schedule as above. A5 temperature search [0.001,0.1]
  per class, choose max probability closest to but below 0.5.
- Translation: frozen window features stride 2; Sign2GPT Phoenix14T/CSL-Daily,
  omit pseudo-gloss pretraining; Tarrés et al. How2Sign. Preserve their training
  recipes; do not claim feature export alone reproduces translation metrics.

## Information that must remain explicit

| ID | Unresolved historical detail | Implementation requirement |
|---|---|---|
| U01 | Pose weight identity/version used by SignRep | Pin public teacher hashes; native body angles match demo; camera-root alternative explicit; historical identity unverified |
| U02 | Original filtered YT-SL manifest and median bone table | Build from supplied manifest; record hash, exclusion IDs and actual sample count |
| U03 | Per-axis distance source code is absent | Default signed source−destination, supported by released-head consistency; opposite/absolute ablations explicit |
| U04 | Gram normalization, detach/update ordering | Document selected convention, test gradient directions, retain configurable normalization |
| U05 | Angular confidence aggregation; missing activity; loss reduction/SmoothL1 beta | Angle masks use teacher observation flags (optional all-point visibility); complete observed activity, element mean, beta=1; record engineering choices |
| U06 | Augmentation probabilities/crop range, fps and short-video policy | Configured explicitly; no guessed historical values described as author defaults |
| U07 | Batch 20 counts views versus video pairs | Default 20 total views (10 pairs); record and allow alternative |
| U08 | Raw activity head extrapolation in weighted retrieval; zero weight | Preserve raw head by default; fail on zero/nonfinite denominator; alternatives explicit |
| U09 | Class similarity aggregation/self similarities; temperature step | Explicit policy and grid; tests for selection and unattainable bound |
| U10 | Original environment/seeds and translation integration revisions | Record installed versions/source hashes; do not promise exact numerical reproduction |

These are source limitations, not permission gates. Implementation proceeds with
named, documented choices, while historical-equivalence claims stay unverified.

## Sources

- https://github.com/ryanwongsa/SignRep/tree/06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2
- https://github.com/ivashmak/skeleton_3d_uplift/tree/6885e4865bc8993ae41a4d88287a5b03281d7d91
- https://arxiv.org/html/2503.08529v1 (cross-check only; final PDFs authoritative)
- Local final PDFs and hashes: `evidence/sources.json`.
