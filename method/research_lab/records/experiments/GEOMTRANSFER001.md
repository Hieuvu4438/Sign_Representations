# GEOMTRANSFER001 — H006 base-class geometry transfer

Completed exploratory diagnostic, H006_v0/D027 → [D028](../decisions/D028.md).
Run GEOMTRANSFER001_20261004_H006_v0_9ea77572_s1701; configSHA
9ea77572a4607061041a851bcb315908a1d7aade24855a951176411eb6e4d273.
Terminal exit0/session68178/PID2886986/hostptitiec, CPU1.5844462199602276s/GPU0,
no retries/new encoder calls/test access. Rank8 fixed, no HPO. Fit1313train sources
from100old classes, all383dev queries of100class-disjoint new classes, three
fixed100source one-example train galleries. Same corpus and five reused val
participants, extra fully gloss-labeled base access; not globally one-shot-only.
Base-val and unused target-train feature arrays not loaded or fitted.

| Fixed mode | Mean DCG across383 queries and3 galleries |
| --- | ---: |
| Raw | .7667061357428586 |
| Base centering | .7671060501134552 |
| Random rank8 removal | .7676774557084557 |
| Total PCA rank8 removal | .7592248983884282 |
| Shuffled-label residual PCA | .7580830289418462 |
| True-label residual PCA | .8102504602148016 |

Registered diagnostic gate passes: residual-minus-raw+.04354432447; differences
against center/random/total/shuffled+.04314441010/+.04257300451/+.05102556183/
+.05216743127, all3 galleries positive against raw. Conditional paired five-group
95%interval against raw[2.6424586293,7.5856533547] in100×DCG units. All five
control intervals, secondary metrics and full6894 complete100class ranking rows
retained in [report](GEOMTRANSFER001_report.json). Shared supports, prior dev reuse
and five groups disallow population/causal/independent-confirmation claims.

GEOMTEST001 initial3 groups; GEOMTEST002 final4 risk oracles and CLI smoke pass.
Post-outcome verification preregistration GEOMVERIFY001_registration.json locks
exact ranking agreement and numeric1e-12/orthogonality1e-10/covariance1e-8 tolerances.
GEOMVERIFY001_20261004_H006_v0_963633df_s1701 exit0/session14505/PID2904406,
CPU1.5402573139872402s/GPU0, no retries. Independent arithmetic implementation
rebuilds all6894 rankings via direct Euclidean distances exactly; raw1149 ranks
match earlier REPQUAL001. Recounts all row/aggregate metrics/primary gates,
class/source/ID/split/support/load contracts, labels/shuffle counts, all registered
input and output digests. Independent covariance sum formula verifies saved
bases/singular equations, relative errors≤4.824e-15; means/orthogonality/random
QR also pass. [Verification](GEOMTRANSFER001_verification.json) is same-agent
software verification, not independent scientific review. No extra task trial.

Native producer/checkpoint/full_activity pool/scorer and H002 locks unchanged.
Fitted geometry, fit/support metadata, predictions, config/env/status/source
snapshot and logs remain in the immutable runtime directory. Output SHAs remain
those in status and verification. Peak RSS uninstrumented;64MiB was an estimate,
not measured memory. Two instrumented attempts totalCPU3.124703533947468s/GPU0.

Decision: continue class-conditioned geometry analysis and closest-baseline audit;
known NAP/PCA family, no novel method, signer-removal causality or paper promotion.
