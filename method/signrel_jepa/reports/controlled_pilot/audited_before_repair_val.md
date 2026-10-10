# Controlled frozen evaluation pilot

Split: val. Same pretrained encoder, data, optimizer, batch, steps and random masks.

| Variant | R@1 | R@5 | Instance mAP | Phonology macro-F1 | Signer accuracy |
|---|---:|---:|---:|---:|---:|
| A_generic | 0.0375 ± 0.0125 | 0.2000 ± 0.0331 | 0.0630 ± 0.0019 | 0.2134 ± 0.0212 | 0.9683 ± 0.0137 |
| B_articulator | 0.0417 ± 0.0072 | 0.2208 ± 0.0072 | 0.0669 ± 0.0010 | 0.2329 ± 0.0568 | 0.9603 ± 0.0137 |
| C_relation | 0.0250 ± 0.0125 | 0.1792 ± 0.0402 | 0.0651 ± 0.0042 | 0.2667 ± 0.0397 | 0.9524 ± 0.0000 |
| D_combined | 0.0458 ± 0.0260 | 0.1875 ± 0.0125 | 0.0682 ± 0.0042 | 0.2324 ± 0.0264 | 0.9603 ± 0.0137 |

Decision: **relation_benefit_not_supported_by_this_pilot**.

40-gloss local pilot; descriptive training-seed variation; no significance, full benchmark or SOTA claim

Both generic and articulator controls, complete paired seeds and matching coverage must be checked. Broader validation is required before multiscale/scale-up.

## Paired relation comparisons

| Comparison | Δ R@1 | Δ phonology F1 | Δ signer accuracy | Seeds |
|---|---:|---:|---:|---:|
| C_relation_vs_A_generic | -0.0125 | +0.0532 | -0.0159 | 3 |
| D_combined_vs_A_generic | +0.0083 | +0.0189 | -0.0079 | 3 |
| C_relation_vs_B_articulator | -0.0167 | +0.0338 | -0.0079 | 3 |
| D_combined_vs_B_articulator | +0.0042 | -0.0005 | +0.0000 | 3 |

A smaller signer score is beneficial only when linguistic task quality is maintained.

Verified runs: 12. Provenance reporting warnings: 24. Original feature/evaluation artifacts are preserved.
