# A1 full160 — official ASL Citizen test

EMA teacher final160,200.640updates. Gallery train40.154/query test32.941,2.731gloss. Cosine class-max/pessimistic ties. PCA fit gallery train only; no selection on test.

| Readout | R@1 (%) | R@5 (%) | R@10 (%) | MRR | DCG | Macro R@1 (%) |
|---|---:|---:|---:|---:|---:|---:|
| grp | 11.511 | 28.005 | 37.100 | 0.20052 | 0.33476 | 11.633 |
| grp+P | 6.928 | 18.305 | 25.640 | 0.13244 | 0.26706 | 6.997 |
| mw | 9.872 | 25.078 | 34.298 | 0.17893 | 0.31463 | 9.953 |
| ord2 | 7.340 | 19.259 | 26.769 | 0.13859 | 0.27286 | 7.404 |

## Same-readout change versus epoch40

| Readout | Epoch40 R@1 (%) | Epoch160 R@1 (%) | Delta (points) |
|---|---:|---:|---:|
| grp | 11.305 | 11.511 | +0.206 |
| grp+P | 8.567 | 6.928 | -1.639 |
| mw | 11.059 | 9.872 | -1.187 |
| ord2 | 9.365 | 7.340 | -2.025 |

Primary readout remains grp+P. Earlier epoch40 test was explicitly requested and recorded; no test score was used to stop or choose this final160 checkpoint.
