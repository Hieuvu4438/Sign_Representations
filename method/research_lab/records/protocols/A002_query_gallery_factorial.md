# A002 — post-outcome query/gallery attribution audit

Registered UTC 2026-10-03T22:51:05.405494+00:00. Original P003 diagonal results already
known. **Exploratory post-outcome**, not an original confirmatory endpoint,
additional sampler search or population/causal proof. Primary P003 unchanged.

Question: within the measured fixed-checkpoint setting, how do query-only,
gallery-only and joint policy changes affect rankings? Existing diagonal
comparisons cannot attribute random/uniform differences solely to query exposure.

Use exactly four original activity-pooling recipes (dense,uniform8,random8seed2718,
random8seed31415), all16query×gallery combinations/all3fixed800clip train galleries,
all1217test queries/100classes/11participant IDs. Original full_mean changes pooling
and is outside this sampling-only factorial; original quality result retained.
No fit, interpolation, new seed/method, support/label selection or GPU inference.
Reuse verified5method float64/L2 query pooled payload and train-native CACHE005;
fixed original supportIDs, class-max cosine/search/ties/scorer/source mappings.

Regenerate all48complete ranking matrices; all12activity-method diagonal matrices
must exactly equal CONFIRM001, otherwise abort and diagnose before interpretation.
Report DCG/MRR/Top1/5/10/20 per pair/support and means; retain allpairs including
negative outcomes. DCG and Top1 contrast formulas in immutable FACTOR001config:
query-only, gallery-only, joint, inclusion-exclusion interaction; every seed,
mean and participantN/DCG contrasts. Allslice data remain original; no new slicing,
bootstrap, significance claim or best-pair selection. Interaction is an arithmetic
ranking-utility diagnostic, not a causal or linguistic mechanism estimate.

Resource contract: oneCPU/BLASthread,240s, zeroGPU allowance, no HPO trials,
rank payload ceiling48×1217×100×int16≈11.7MB plus small records. Existing bounded
verifiedtmpfs inputs, no source writes/downloads/uploads/deletion or environment
changes. Config binds original completed prediction/scoring/integrity statuses.
Register source snapshot/checks gate after unit oracles and before first mixed
outcome computation. Failed attempts retained/new output for retry.

Decision rule: do not change original primary conclusion or choose a new policy.
Use complete matrices/contrasts only to qualify attribution in dossier/direction
memo; if effects cannot be localized cleanly, report that limit. Publishability
requires independent closest-work/direction assessment and appropriate scope.
