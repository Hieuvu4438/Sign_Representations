# A1 epoch40 — ASL Citizen VAL-DICT

Gallery official train: 40.154 clip. Query official val: 10.304 clip; 2.731 gloss. EMA teacher, cosine class-max, ties bi quan. Không dùng official test.

| Readout | R@1 (%) | R@5 (%) | R@10 (%) | MRR | DCG | Macro R@1 (%) |
|---|---:|---:|---:|---:|---:|---:|
| grp | 15.450 | 35.316 | 46.244 | 0.25450 | 0.38814 | 15.692 |
| grp+P | 12.238 | 30.939 | 41.809 | 0.21853 | 0.35610 | 12.196 |
| mw | 16.033 | 37.267 | 48.952 | 0.26592 | 0.40019 | 16.145 |
| ord2 | 13.888 | 33.220 | 43.643 | 0.23697 | 0.37251 | 13.900 |

`grp+P` là readout chính, PCA768 chỉ fit gallery train. `grp` không PCA dùng cho so sánh cùng chiều với A0. Không so trực tiếp VAL-DICT này với R@1 test PLctx full160.
