| Method | DCG | Top1 (%) | Query p50 (s) | Query p95 (s) | Windows reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_mean | 0.842867024 | 72.8294 | — | — | — |
| full_activity | 0.886378665 | 80.3889 | 0.878318 | 1.769789 | 1.000000 |
| uniform8 | 0.884978943 | 80.1150 | 0.323220 | 0.636664 | 3.816549 |
| random8_s2718 | 0.857259423 | 74.4727 | 0.325376 | 0.632040 | 3.816549 |
| random8_s31415 | 0.850541570 | 73.1854 | 0.324232 | 0.635374 | 3.816549 |

N=1217, mean over all three fixed train galleries; DCG/MRR fraction, Top1 percent.
Full mean shares the dense encoder pass for quality; no separate latency claim.
One independent measured pipeline per query/recipe, cyclic counterbalance, shared eligible GPU, uncontrolled OS cache.
Gallery extraction/setup excluded from query latency; conditional group uncertainty, no population equivalence or full-benchmark claim.
