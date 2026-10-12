# Implementation notes (proposal12 v2.1 → code)

**Status (2026-10-11):**
- Implemented. 40 tests pass.
- Two independent code-review rounds:
  - Round 1: 1 critical, 1 major and 6 minor issues, all fixed with regression tests.
  - Round 2: verified all 8 fixes and raised 6 new items (N1–N6), all fixed with tests for N1 and N2.
- GPU smoke runs (60, 200 and 100 steps) completed without NaNs, at ~0.12 s/step.
- **Nothing in this folder is a result yet.**

## Spec → code

| Spec (proposal12 v2.1) | Code |
| --- | --- |
| §6.3 joint layout (103), 7 channels, ownership, closure F_HD ⇒ F_UP / F_MO, DEPENDS, mirroring | `pj/joints.py` |
| §6.3 observation (thresholds of obs_jepa B) | `pj/data/cache.py::gate` |
| §6.3 steps 2–4: handedness, content-independent window placement, linear-interpolation resampling with support | `pj/data/window.py` |
| §6.6 PLCM (R0–R4, budget, context dropout, curriculum) | `pj/masks.py` |
| §6.3 steps 6–10: hidden source frames → scrub (owned joints and R1 joints) → propagation through support and DEPENDS → teacher/student views | `pj/data/dataset.py::build_pair` |
| §6.4 features (CONF / LOC / UP / MO / HD), tokens, ABSENT | `pj/features.py` |
| §6.7 intrinsic targets, reliability | `pj/features.py` |
| §6.5 encoder (RoPE, CLS, visible-only keys), predictor (query slot per target cell), heads, EMA | `pj/model.py` |
| §6.7–§6.8 losses: centred CE (S-JEPA), proto, L1 + SIGReg, intrinsic smooth-L1, KoLeo, loss balancer | `pj/losses.py` |
| §6.10 augmentation (body-frame affine, IK morphology, noise, hand dropout, speed) | `pj/data/window.py`, `pj/features.py::to_body_frame` |
| §6.11–§6.13 training step, gates, arms; §9 factorial and baselines | `pj/train.py` (flags) |
| §6.14 readouts (_grp primary, _mean, _cls); B0 descriptor | `pj/extract.py` |
| §8 evaluation (proposal11 protocol, unchanged code) | `pj/evaluate.py` → `method/obs_jepa/oj/evaluate.py` |
| §12–§13 sequential plan P1–P3, run manifest, abort on unfinished runs | `scripts/schedule.py`, `pj/train.py` manifest |

## Comparison protocol (fixed by the user's 2026-10-11 decision)

The protocol is identical to obs_jepa A/B and EX16:
- **Data and budget:** ASLC002 corpus (3,000 videos), 4,000 steps × 32 clips, windows of 64 samples at min(fps, 25), 1 seed.
- **Optimizer:** the obs_jepa B recipe.
- **Evaluation:** `oj/evaluate.py` unchanged, with reference models copied read-only for the paired statistics.

Per step and per clip, PhonoJEPA uses two masked views (near, far) against two teacher views. This is the multi-mask amortisation of data2vec 2.0 and V-JEPA. The number of clip samples is identical to B; the compute per step is higher, and the run manifest records it.

## Deviations (declared)

| # | Deviation | Why |
| --- | --- | --- |
| 1 | Body joints are gated by their joint threshold only. B used a part-mean gate. | Measured on 300 clips: B's body part gate rejected 36% of frames, while the wrist passed its joint threshold in 89% of hand-valid frames. A part mean over several channels would also leak hidden scores (review finding #7). |
| 2 | Hand part gate uses palm joints only; the face gate is computed per face channel. | Finding #7: no gate may span a mask boundary. |
| 3 | The handedness decision is content-based (declared 1-bit leak). | Worst-case swap-in flip rate is 0.20 (PL), 0.25 (R4) and 0.33 (R0) (`test_handedness_flip_rate_swap_in`). Arm A8(g) `--handedness random` removes the leak. |
| 4 | Near and far views draw independent windows (start, speed). §6.13 pseudocode shares one crop per (teacher, student) pair. | Views stay paired with their own teacher; independent windows only matter for the G1 CLS cross-view term. |
| 5 | EMA momentum ends at 0.9999, not 1. | Same value as obs_jepa B. |
| 6 | The 25% minimum-visible check runs before propagation. | Propagation can push a short clip slightly below 25%. This is logged through the hidden fractions. |
| 7 | The student encoder runs all 224 slots under a key mask; it does not gather the visible tokens. | Simpler and verifiably leak-free (test). Costs about 2× compute, which at this scale is still about 8 min per run. |
| 8 | The far-view budget is U(0.3, 0.6) × 2 × n_valid "hand slots" for **every** family (PL / R4 / R0), and R3 never hides both channels of one hand in a bin. Measured hidden fractions: near 0.57 / 0.57 / 0.60, far 0.13 / 0.13 / 0.17 of all cells. | Round 1 #2: the v2.1 text let the far view collapse into "both hands hidden". Round 2 N1: the three factorial rows must differ only in mask family, so they share one budget. |
| 9 | PLfull (gate G2): the R4 articulator span is uncapped, U(0.4, 1) × n_valid, and the R3 budget is added on top of it. | Round 2 N2: without this, R4 spans were cut to ~0.22 T and R3 contributed no hand targets. |
| 10 | Intrinsic targets are computed from the teacher view **without** its camera affine. The contextual teacher keeps the affine. | Round 2 N5: the student cannot know the teacher's rotation/scale, which would put an irreducible noise floor (~0.1 z) on absolute targets. |
| 11 | The centring update is committed only after a finite optimizer step. SIGReg uses 1024 slices. The collapse alarm fires at CLS mean-cos > 0.99 after 10% of the steps. A missing target-stats file now raises a warning. | Round 2 N6 |

## Verification

Tests (`tests/test_core.py`):

| Area | Tests |
| --- | --- |
| Layout | `test_ownership_and_mirror`, `test_mirror_roundtrip` |
| Masks | `test_sampler_properties` (5 families × content-independence, closure, padding, R3 hand-only, 25% context), `test_pl_far_keeps_other_channel_of_same_hand` |
| Leakage (data) | `test_e2e_hidden_source_perturbation` (5 configs × near/far: overwrite every hidden source entry with 1e6 / random conf → student arrays bit-identical), `test_perturbing_visible_content_changes_student` (non-vacuous), `test_propagation_with_speed`, `test_observation_gate_is_channel_local` |
| Leakage (model) | `test_student_outputs_ignore_hidden_tokens` (visible outputs, CLS and predictor queries are bit-identical when non-visible inputs change) |
| Numerics | `test_features_identical_under_autocast`, `test_feature_dims` |
| Training | `test_teacher_no_grad_and_ema`, `test_arms_one_step` (7 arms), `test_absent_targets_get_zero_weight`, `test_param_groups_exclude_predictor_channel_embedding` |
| Handedness | `test_handedness_flip_rate_swap_in` (measurement of the declared leak) |

Pipeline checks:
- **CPU smoke, extract → `oj/evaluate` end to end:** passed for the model readouts and for B0.
- **GPU smoke, 200 steps:**
  - ctx loss: 18.4 → 2.6
  - int loss: 0.66 → 0.50
  - teacher entropy: 0.21 → 0.39
  - gradient norm: 27 → 1.0
  - no non-finite steps

## Not implemented / open

- Phonology probe (ASL-LEX, lexically disjoint) and E3b (signer probe on the target spaces). These are planned for the evaluation stage; the current evaluation is the proposal11 protocol only.
- A4 ablation of palm-frame vs body-frame CONF. It is not a flag; the D7′ diagnostic decided body frame.
- Model M / Tier A (P4) and the final test (P5), which need new approval.
