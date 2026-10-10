# Controlled frozen evaluation pilot

Split: test. Same pretrained encoder, data, optimizer, batch, steps and random masks.

| Variant | R@1 | R@5 | Instance mAP | Phonology macro-F1 | Signer accuracy |
|---|---:|---:|---:|---:|---:|
| A_generic | 0.0208 ± 0.0361 | 0.1208 ± 0.0382 | 0.0550 ± 0.0130 | 0.2093 ± 0.0398 | 0.9459 ± 0.0000 |
| B_articulator | 0.0292 ± 0.0191 | 0.1458 ± 0.0591 | 0.0600 ± 0.0105 | 0.1820 ± 0.0062 | 0.9459 ± 0.0000 |
| C_relation | 0.0167 ± 0.0072 | 0.1958 ± 0.0260 | 0.0601 ± 0.0022 | 0.2329 ± 0.0399 | 0.9279 ± 0.0156 |
| D_combined | 0.0208 ± 0.0191 | 0.1792 ± 0.0382 | 0.0610 ± 0.0068 | 0.1983 ± 0.0079 | 0.9279 ± 0.0156 |

Decision: **relation_benefit_not_supported_by_this_pilot**.

40-gloss local pilot; descriptive training-seed variation; no significance, full benchmark or SOTA claim

Both generic and articulator controls, complete paired seeds and matching coverage must be checked. Broader validation is required before multiscale/scale-up.

## Paired relation comparisons

| Comparison | Δ R@1 | Δ phonology F1 | Δ signer accuracy | Seeds |
|---|---:|---:|---:|---:|
| C_relation_vs_A_generic | -0.0042 | +0.0236 | -0.0180 | 3 |
| D_combined_vs_A_generic | +0.0000 | -0.0110 | -0.0180 | 3 |
| C_relation_vs_B_articulator | -0.0125 | +0.0509 | -0.0180 | 3 |
| D_combined_vs_B_articulator | -0.0083 | +0.0164 | -0.0180 | 3 |

A smaller signer score is beneficial only when linguistic task quality is maintained.

Verified runs: 12. Provenance reporting warnings: 24. Original feature/evaluation artifacts are preserved.
