# Post-outcome query/gallery factorial (descriptive)

All1217queries/100classes/11participants; mean DCG across the same three fixed eight-per-class train galleries.
Rows=query recipe; columns=gallery recipe. All12original activity-method diagonal rankings match exactly.

| Query / gallery | full_activity | uniform8 | random8_s2718 | random8_s31415 |
| --- | ---: | ---: | ---: | ---: |
| full_activity | 0.886378665 | 0.885947526 | 0.871312807 | 0.873333177 |
| uniform8 | 0.885773539 | 0.884978943 | 0.871438998 | 0.868381332 |
| random8_s2718 | 0.875667106 | 0.875973421 | 0.857259423 | 0.852633130 |
| random8_s31415 | 0.872231296 | 0.870220073 | 0.850093366 | 0.850541570 |

| Sparse recipe | Query-only DCG delta | Gallery-only DCG delta | Joint delta | Interaction |
| --- | ---: | ---: | ---: | ---: |
| uniform8 | -0.000605126 | -0.000431140 | -0.001399723 | -0.000363457 |
| random8_s2718 | -0.010711560 | -0.015065858 | -0.029119242 | -0.003341824 |
| random8_s31415 | -0.014147369 | -0.013045489 | -0.035837096 | -0.008644238 |

Post-outcome exploratory; no new policy selection, primary metric or causal/linguistic claim.
All seed metrics, participant denominators/deltas and Top1 contrasts retained in summary/CSV.
These mixed query/gallery pairs have no measured mixed-pair query latency.
