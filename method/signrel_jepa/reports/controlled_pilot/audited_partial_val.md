# Controlled frozen evaluation pilot

Split: val. Same pretrained encoder, data, optimizer, batch, steps and random masks.

| Variant | R@1 | R@5 | Instance mAP | Phonology macro-F1 | Signer accuracy |
|---|---:|---:|---:|---:|---:|
| A_generic | 0.0375 ± 0.0177 | 0.2125 ± 0.0354 | 0.0628 ± 0.0026 | 0.2015 ± 0.0064 | 0.9762 ± 0.0000 |
| B_articulator | 0.0437 ± 0.0088 | 0.2188 ± 0.0088 | 0.0670 ± 0.0013 | 0.2064 ± 0.0473 | 0.9643 ± 0.0168 |
| C_relation | 0.0188 ± 0.0088 | 0.1875 ± 0.0530 | 0.0645 ± 0.0058 | 0.2533 ± 0.0455 | 0.9524 ± 0.0000 |
| D_combined | 0.0375 | 0.2000 | 0.0658 | 0.2056 | 0.9524 |

Decision: **insufficient_evidence**.

40-gloss local pilot; descriptive training-seed variation; no significance, full benchmark or SOTA claim

Both generic and articulator controls, complete paired seeds and matching coverage must be checked. Broader validation is required before multiscale/scale-up.

## Paired relation comparisons

| Comparison | Δ R@1 | Δ phonology F1 | Δ signer accuracy | Seeds |
|---|---:|---:|---:|---:|
| C_relation_vs_A_generic | -0.0188 | +0.0518 | -0.0238 | 2 |
| D_combined_vs_A_generic | -0.0125 | -0.0004 | -0.0238 | 1 |
| C_relation_vs_B_articulator | -0.0250 | +0.0469 | -0.0119 | 2 |
| D_combined_vs_B_articulator | -0.0125 | -0.0342 | +0.0000 | 1 |

A smaller signer score is beneficial only when linguistic task quality is maintained.

Verified runs: 7. Provenance reporting warnings: 14. Original feature/evaluation artifacts are preserved.
