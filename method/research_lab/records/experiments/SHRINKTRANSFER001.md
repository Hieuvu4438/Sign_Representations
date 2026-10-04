# SHRINKTRANSFER001 — H006_v1 standard covariance falsifier

Completed exploratory diagnostic H006_v1/D029 → D030, registered primary gate
**fails**. Run SHRINKTRANSFER001_20261004_H006_v1_b010cafb_s1701,
configSHAb010cafba13bcbebf2959bbec02398764f22bdde1cc2fde222cab123246e53d3.
Terminal exit0/session42366/PID2923310/hostptitiec,
CPU3.101969109964557s/GPU0/no retries/test/new encoder/HPO. Fixed rank8 prior
recipe versus six standard base-derived LW controls; all1313old-class base train,
383dev/100local class-disjoint target classes/three fixed one-example galleries.
Same fully labeled base budget/native producer/input/cache/scorer; no target fit,
baseval or unused targettrain array loading. Same-corpus reused five dev people.

| Mode | Mean DCG |
| --- | ---: |
| Original raw | .7667061357428586 |
| Original rank8 residual PCA | .8102504602148016 |
| Residual LW cosine | .8297463854564660 |
| Residual LW Mahalanobis | .8046075388724049 |
| Total LW cosine | .7626392085011816 |
| Total LW Mahalanobis | .7074060946519143 |
| Shuffled-label LW cosine | .7621715720449861 |
| Shuffled-label LW Mahalanobis | .7050348767538016 |

PCA-minus-residual-LW-cos−.01949592524, all three gallery losses. PCA-minus-LW-maha
+.00564292134 but negative seed1701 difference; primary requires both margins
and all galleries nonnegative, so fails. Conditional five-group95%intervals
PCA-minus-LW-cos[-4.1147676535,-.9973794207], PCA-minus-LW-maha[-.8634949506,
1.4537319068]100×DCG units, not population/causal/selection-adjusted significance.
Secondary registered true-label vs total/shuffled LW gains retained for both
metrics: cosine+.06710717696/+.06757481341, Mahalanobis+.09720144422/+.09957266212.
This supports a bounded label-conditioned geometry observation, not new method.

Train-derived LW coefficients residual.05555678719233484,total.035573129576730804,
shuffled.03560319443735448. Fixed existing sklearn1.8.0, not online docs1.9.1;
pooled residual/shared-signer dependence precludes optimality/IID guarantees.
All fitted covariance/whitening/eigenvalues, means/labels/supports, complete9192
100class ranks, per-seed/secondary metrics/conditional intervals, argv/env/source
snapshot/config/status/log/SHAs retained. No Simple CNAPS/LFD/FROST reproduction.

Three SHRINKTEST001 groups/CLI pass. All9192measured rows exactly match direct
transformed-space Euclidean rankings; original2298raw/PCA ranks match v0. Parent
SHA bindings all unchanged. SHRINKVERIFY001_20261004_H006_v1_451ad2ba_s1701,
configSHA451ad2ba60ab6b9f5052d0c41bcf5c5478178bae2bf9dadffd2742945a5420c4,
exit0/session90347/PID2927854/hostptitiec,
CPU1.316439716028981s/GPU0/no retries. Independent original-space inverse-covariance
metrics regenerate all6894new LW ranks exactly; all9192rows and aggregates/primary
gate recounted. Fourth-moment oracle coefficients agree, max covariance error
4.337e-19; input/output digests and parent fit/support contracts pass. Same-agent
software arithmetic check, not independent review. Verification/report common
copies retain identical bytes; terminal status/source snapshots authoritative.

Resource incident: actual measurement durable logical bytes40,750,758 (~38.863MiB)
**exceed the registered20MiB ceiling**. Covariance+whitening archive28,338,802bytes
and JSON predictions11,777,579bytes dominate. This is an output-estimation/control
failure, not a task-score or scientific gate rescue. Root bulk reserve still fails;
no retroactive ceiling increase/deletion/overwrite of completed outputs. See
H006_STORAGE_PREFLIGHT_INCIDENT.json. Future large writes require full array-nbytes
plus bounded serialization/snapshot estimates before registration. No large copy
of matrices/predictions made to common records; only small report/verification.
Peak RAM uninstrumented;128MiB was planned, not measured. Total instrumented CPU
4.418408825993538s/GPU0 for measurement+verification; GPU ledger unchanged.

[D030](../decisions/D030.md)/C066/F062 kills fixed rank8 PCA superiority/scaling.
Retain standard covariance-aware baselines and class-transfer analysis question;
contribution and valid independent confirmation still open. No method promotion.
