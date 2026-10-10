# Confirmatory analysis (T06) — EX05-v1

Protocol `protocols/EX05-v1.md`, LOCKED 2026-10-06 before test extraction; single test access logged in `method/p4_signssl/runtime/logs/TEST_ACCESS.log` (one engineering re-run of the ASL Citizen chamfer step after a CUDA OOM, same code path and readouts). Raw outputs: `method/p4_signssl/runtime/eval/EX05_result.json`, rank files in `runtime/eval/EX05_*`.

## Results (frozen SignRep, released checkpoint, no training of any kind)

| Dataset (test) | Readout | R@1 | MRR | DCG | R@5 |
| --- | --- | --- | --- | --- | --- |
| ASL Citizen (32,941 q, 11 signers) | R0 paper (activity-weighted mean) | 0.4902 | 0.6234 | 0.7042 | 0.7929 |
| | R1 bins2 (single 1536-d vector) | 0.5493 | 0.6709 | 0.7418 | 0.8239 |
| | R2 chamfer@20 (window sets) | 0.5842 | 0.6982 | 0.7624 | 0.8421 |
| WLASL2000 (2,879 q) | R0 | 0.2942 | 0.4558 | 0.5713 | 0.6547 |
| | R1 | 0.3143 | 0.4752 | 0.5871 | 0.6801 |
| | R2 | 0.3057 | 0.4719 | 0.5845 | 0.6763 |

Paired differences (R@1; bootstrap 10,000; ASL Citizen clustered by signer, WLASL by class):

| Comparison | ASL Citizen | WLASL2000 |
| --- | --- | --- |
| R1 − R0 | +5.90 [+3.87, +7.53], 10/11 signers, Holm p = 0.0002 | +2.01 [+0.94, +3.11] |
| R2 − R0 | +9.40 [+8.15, +10.60], 11/11 signers, Holm p = 0.0002 | +1.15 [−0.24, +2.48] |

Verdicts by the locked rules: R1 SUPPORTED_WITHIN_SCOPE; R2 SUPPORTED_WITHIN_SCOPE (rule requires positive WLASL sign; its CI includes 0).

Baseline fidelity: R0 on ASL Citizen test reproduces SignRep v1 Table 4 (0.4902 vs 49.95 R@1; 70.42 vs 71.21 DCG) → G1 = EVAL_REPRODUCED.

## Uncertainty and limits

- Sample uncertainty only; the encoder is fixed (no training seeds involved). Signer clusters: 11 (ASL Citizen).
- Effect size is dataset-dependent (ASL Citizen ≫ WLASL). Not explained by sign duration (within ASL Citizen, short active spans gain most; within WLASL, long spans gain most — val analysis) and only weakly by gallery size (exploratory, ASL Citizen val gallery subsampled to 3/7/15 exemplars per class: bins2 +4.3/+5.5/+6.1, chamfer +7.9/+9.5/+9.6 R@1; `scripts/explore_gallery_size.py`). Signer overlap tested (exploratory, official WLASL metadata `WLASL_v0.3.json`, val queries; `scripts/explore_wlasl_signer_disjoint.py`): 99.9% of WLASL val/test signers appear in the train gallery; removing the query signer's own gallery clips gives bins2 +1.73 / chamfer +2.81 (vs +1.51 / +1.89 official) → signer overlap explains little of the ASL Citizen–WLASL difference. Remaining candidates: pre-trimmed 256-px 25-fps clips, recording style, SignRep's checkpoint selection on WLASL val.
- Side finding (exploratory, nuisance): removing same-signer clips from the WLASL gallery **raises** the paper readout from 0.3036 to 0.3276 R@1 — same-signer clips of other glosses act as distractors, i.e. the averaged SignRep embedding carries a substantial signer/appearance component.
- One encoder only for the confirmatory claim. SHuBERT (author extraction, FFN-branch outputs) is at floor for frozen kNN (≤ 4.6% R@1), so generality to a second encoder is untested; a residual-stream extraction (EX06b) is running.
- ASL Citizen test had been scored by unrelated projects in this repository before; none of their results were used here, but the split is not pristine at the repository level.
- Readout choices (span threshold, K = 2, rerank depth 20) were fixed on val before the lock; no test-time tuning.
