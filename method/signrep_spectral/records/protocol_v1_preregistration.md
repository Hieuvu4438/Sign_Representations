# Pilot protocol v1 — pre-registration (written before any quality measurement)

Date: 2026-10-05. Status: **locked before B0/B1/M1 features were scored**.
Method name: *SignRep-initialized canonical continuation + spectral auxiliary*. This
is not a SignRep reproduction and the metric is not an official benchmark.

## Question
Does adding the masked multiscale log-magnitude spectral loss (M1) to a framewise
geometric reconstruction continuation (B1) improve **frozen** RGB representations?

## Data
- Continuation training: ASL Citizen local copy, official `train` split, eligible
  videos only (audit `runtime/data_audit_001`), **labels never used by the objective**.
- Evaluation: same corpus. Gallery / probe-train = all `train` videos;
  development queries = `val` (6 signers); final queries = `test` (11 signers),
  scored once after this protocol and the arm configs are frozen.
- Splits are signer-disjoint (verified: no participant overlap).
- Pretraining membership of the SignRep checkpoint w.r.t. ASL Citizen: **unknown**.
  No zero-shot / unseen claims.

## Arms (one seed each in this pilot = exploratory screen, not confirmatory)
| ID | Training | Aux strength |
|---|---|---|
| B0 | none (published checkpoint, encoder-only extraction) | – |
| B1 | new decoder + L_geom (lambda=0, spectral never computed) | – |
| M1 | B1 + spectral log-magnitude L1 | calibrated to encoder grad ratio 0.1 |
| B2 | B1 + velocity L1 | calibrated to ratio 0.1 |
| B4 | B1 + window-time L1 (same windows/Hann/centering) | calibrated to ratio 0.1 |

B3 (acceleration) is optional and only run if budget remains; its calibrated lambda
is still recorded. All trained arms: same initial weights (hash-checked), same
sample schedule, same explicit Hiera masks per batch index, same optimizer/LR
schedule/updates, bf16 autocast for the network and FP32 losses.

Calibration rule (fixed): lambda_aux = 0.1 / median_b(||grad_enc L_aux|| / ||grad_enc L_geom||)
over 8 fixed train minibatches at the shared initial state; spectral warmup 100 updates.

## Feature extraction (identical for all arms)
16-frame windows, stride **S** (set once from measured throughput before scoring and
recorded in `decisions.md`; identical for every arm), mean of `features` windows in
float64 then L2. bf16 autocast + `sdpa_4d_reshape` attention kernel for every arm.

## Metrics
- Primary: class-ranking **MRR** on val queries, gallery = train, class score =
  max cosine, ties by class id (adapted protocol `asl_citizen_frozen_class_retrieval_v1`).
- Secondary: Recall@1/5/10, macro MRR; frozen linear probe top-1 (Linear(768,C),
  AdamW lr 1e-3, wd 1e-4, 60 epochs, batch 1024, seed 0, epoch selected on val).
- Uncertainty: paired bootstrap of per-query deltas (video-level and signer-cluster).

## Decision rule (stop / go for the spectral auxiliary)
- **GO (worth a confirmatory multi-seed study):** M1 − B1 val MRR ≥ +0.005 with the
  video-level 95% CI excluding 0, AND M1 ≥ the best of the matched controls (B2, B4)
  on val MRR, AND linear-probe val top-1 not lower than B1 by more than 0.005.
- **HOLD:** M1 − B1 positive but CI includes 0, or M1 does not beat the controls
  (gain explained by generic temporal supervision).
- **STOP (no evidence of representation benefit):** M1 − B1 ≤ 0.
- Decoder reconstruction improvements alone never count as representation benefit.
- Test results are reported once for the frozen arms and never used to change any choice.

## Amendment A1 (2026-10-05 ~10:59, after seeing only B0 dev numbers; before any trained arm was scored)
The locked probe (v1: raw L2 features, lr 1e-3, 60 epochs) had not converged on B0
(val top-1 still rising at the last epoch: 0.473 -> selected epoch 60), so it measures
optimisation speed rather than representation quality. Probe **v2** is added and becomes
the secondary probe metric for every arm: features standardized with train-split
mean/std, Linear(768,C), AdamW lr 1e-3, wd 1e-4, 40 epochs, batch 1024, seed 0, epoch
selected on val. Chosen on B0 dev between two candidates (`runtime/probe_select_B0_001`,
`_002`): v2 converges by ~epoch 8 (B0 val top-1 0.573). v1 numbers are still reported.
Retrieval (primary metric) is unchanged.

## Amendment A2 (2026-10-05 11:45, after B0 and B1 dev scores, before any M1 score)
Observation: B1 (geometric continuation) collapsed the frozen representation
(val MRR 0.510 -> 0.147; mean pairwise cosine of pooled features 0.32 -> 0.93;
effective rank 77 -> 24) although weights moved only 1.2%. The primary M1-vs-B1
comparison stays as registered. Added an exploratory matched pair that keeps the
same objective plus the spec's explicit variance regularizer (`var_cov_v1`,
w_var = 1.0, gamma = 1, w_cov = 0; B0 features already have per-dim std ~0.93 so the
hinge is ~inactive at init): **B1r** (lambda=0) and **M1r** (spectral, same
calibrated lambda 0.383). Weights fixed before running, not tuned. Run order changed
to M1, B1r, M1r, B4, B2. The same GO/HOLD/STOP rule is applied to M1r-vs-B1r and
reported separately (exploratory).
