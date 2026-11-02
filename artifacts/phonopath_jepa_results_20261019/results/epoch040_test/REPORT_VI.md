# A1 epoch40 — ASL Citizen official test

Gallery train40.154 / query test32.941;2.731 gloss. Fixed EMA checkpoint epoch40. User requested early test; no checkpoint/readout selection on test.

| Readout | R@1 (%) | R@5 (%) | R@10 (%) | MRR | DCG | Macro R@1 (%) |
|---|---:|---:|---:|---:|---:|---:|
| grp | 11.305 | 27.613 | 37.303 | 0.19924 | 0.33538 | 11.432 |
| grp+P | 8.567 | 22.510 | 31.438 | 0.16168 | 0.30015 | 8.643 |
| mw | 11.059 | 28.317 | 38.709 | 0.20102 | 0.33960 | 11.134 |
| ord2 | 9.365 | 23.803 | 33.077 | 0.17206 | 0.31008 | 9.450 |

PCA reused from the train gallery of VAL-DICT. This is epoch40, not the final160 model; account for different training budgets in baseline comparisons.
