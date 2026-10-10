# Pilot 2 results (dev = ASL Citizen val) — 2026-10-05

Exploratory screen: one seed per arm, 3,000 updates x batch 16 (48k clips, ~1.2 passes over
train videos), adapted protocol `asl_citizen_frozen_class_retrieval_v1` (not an official
benchmark). Queries: 10,304 val videos, 6 signers; gallery / probe-train: 40,153 train videos.
Run directories: `runtime/pilot2_<ARM>_{train,export,features,retrieval_val,probe_val,probe2_val,recon_val}`.

## Frozen representation metrics (val)

| Arm | Objective | MRR | R@1 | R@5 | macro MRR | MRR (latent) | probe v2 top-1 | probe v1 top-1 |
|---|---|---|---|---|---|---|---|---|
| B0 | published SignRep, no continuation | **0.510** | **0.375** | **0.677** | **0.515** | 0.468 | **0.573** | 0.484 |
| B1 | geometric continuation | 0.147 | 0.084 | 0.199 | 0.148 | 0.136 | 0.342 | 0.054 |
| M1 | B1 + spectral (lambda 0.383) | 0.135 | 0.076 | 0.183 | 0.135 | 0.125 | 0.330 | 0.046 |
| B1r | B1 + variance reg (A2) | 0.123 | 0.070 | 0.164 | 0.124 | 0.113 | 0.342 | 0.121 |
| M1r | M1 + variance reg (A2) | 0.115 | 0.066 | 0.152 | 0.115 | 0.105 | 0.329 | 0.112 |

Probe v2 for the trained arms selected the last or near-last epoch (38-40); B0 converged at
epoch 8. Probe v1 had not converged for any arm (amendment A1).

## Paired comparisons (per-query, bootstrap 2,000; `runtime/pilot2_compare_*`)

| Comparison | Metric | Delta | 95% CI (videos) | 95% CI (signer cluster) | signers > 0 |
|---|---|---|---|---|---|
| M1 - B1 | RR (MRR) | -0.0122 | [-0.0139, -0.0105] | [-0.0162, -0.0077] | 0/6 |
| M1 - B1 | R@1 | -0.0079 | [-0.0104, -0.0053] | [-0.0130, -0.0036] | 0/6 |
| M1 - B1 | probe v2 acc | -0.0123 | [-0.0165, -0.0084] | [-0.0209, -0.0041] | 0/6 |
| B0 - B1 | RR | +0.363 | [+0.355, +0.371] | [+0.290, +0.407] | 6/6 |
| B0 - B1 | probe v2 acc | +0.231 | [+0.220, +0.241] | [+0.159, +0.265] | 6/6 |
| M1r - B1r | RR | -0.0087 | [-0.0107, -0.0066] | [-0.0111, -0.0059] | 0/6 |
| M1r - B1r | probe v2 acc | -0.0125 | [-0.0171, -0.0079] | [-0.0189, -0.0037] | 1/6 |

## Decoder reconstruction (held-out val clips, 1,024; objective-level only)
B1: geom 0.1857, spectral 0.0588, velocity 0.2264. M1: geom 0.1870, spectral 0.0569, velocity 0.2254.
The spectral term improves spectral reconstruction by ~3% without improving representations.

## Feature geometry (val pooled `features`)
| Arm | norm of mean direction | effective rank | residual variance |
|---|---|---|---|
| B0 | 0.565 | 76.9 | 0.681 |
| B1 | 0.966 | 24.1 | 0.067 |
| M1 | 0.969 | 23.6 | 0.062 |
| B1r | 0.415 | 11.9 | 0.828 |

## Decision (pre-registered rule)
- **Primary pair M1 vs B1: STOP.** M1 - B1 < 0 with the CI excluding 0, on every signer.
- **Exploratory pair M1r vs B1r: STOP** (same direction, same strength).
- **Against SignRep (B0):** every continuation arm is far worse (MRR -0.36). The canonical
  continuation itself, a single 768-d vector decoded into 2-D hand trajectories, removes
  sign-discriminative information. Without the regularizer the features collapse toward a
  shared mean direction. With the variance hinge they collapse into a ~12-dimensional
  subspace instead. The spectral auxiliary makes this slightly worse, probably because it pushes the
  encoder further toward hand-motion statistics.
- Controls B2/B4/B3 were not run (B4 training was aborted at start): they are only needed to
  explain a positive M1 effect. Test split was not extracted or scored: the dev verdict is
  decisive and no arm was promoted.

## Cost
~3.6 GPU-hours of logged train/extract/calibration time on a GPU shared with another job,
plus the aborted pilot1 B0 (~9 min) and B4 (<1 min). Runtime outputs 5.7 GB.

## Limits of this conclusion
One seed and a short schedule. Hand-only 2-D targets (no z/body/face) from an unknown pose
backend. Stride-16 bf16 extraction. Adapted metric. SignRep pretraining overlap unknown.
The result says this **continuation design** plus spectral loss does not help. It does not rule out a
spectral term added to SignRep's native multi-prior objective (`native_signrep_extension`),
which is blocked: the public source has no trainer or target builder.

## Next actions if the direction is continued
1. Keep the representation anchored. Add a feature-distillation term to B0 (frozen teacher
   features) in every arm, then re-test M1 vs B1 on the same protocol. Cheapest informative test:
   ~1 h GPU per pair.
2. Or train only a decoder on frozen B0 features (spectral vs no spectral), to test whether
   spectral targets are linearly decodable. This is diagnostic, not a representation claim.
3. Do not scale or run test until a pair shows M1 > B1 on dev.
