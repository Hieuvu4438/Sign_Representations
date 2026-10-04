# H002 first pilot — adapted dev cost/quality

DCG fractions; windows over1691 sources. Latency covers5 fixed sources×3 repeats;
model/gallery setup separate, shared GPU and uncontrolled OS cache. No test/novel
method/confirmatory noninferiority claim.

| Policy | DCG | DCG loss | Windows | Reduction | p50/p95 s | Paired ratio | Joint floor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_activity | 0.934446 | 0.000000 | 61339 | 1.000× | 0.790/1.674 | 1.000 | reference |
| uniform4 | 0.933093 | 0.001352 | 31091 | 1.973× | 0.466/0.961 | 0.583 | fail |
| uniform8 | 0.933839 | 0.000607 | 15976 | 3.839× | 0.305/0.618 | 0.377 | pass |
| random4_s2718 | 0.929084 | 0.005361 | 31091 | 1.973× | 0.464/0.979 | 0.586 | fail |
| random4_s31415 | 0.930191 | 0.004254 | 31091 | 1.973× | 0.488/0.988 | 0.596 | fail |
| random8_s2718 | 0.917034 | 0.017411 | 15976 | 3.839× | 0.311/0.605 | 0.381 | pass |
| random8_s31415 | 0.915378 | 0.019068 | 15976 | 3.839× | 0.300/0.621 | 0.377 | pass |

Selected by frozen rule: **uniform8**. All support-seed metrics and
conditional participant intervals remain in QUAL001; all105 raw observations
remain in TIME001. Successful screen permits further scoped analysis; it is
not a paper direction/novelty lock.
