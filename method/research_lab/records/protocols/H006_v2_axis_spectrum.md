# H006_v2 — Fixed covariance direction/spectrum factorial

LockedD039 before new query results. Adapted diagnostic, inherits H006_v1 native
feature/scorer/access/parity bindings and AXISTEST001 passed manual oracles.
No official benchmark or encoder-unseen claim. Parent fitted_shrinkage.npz from
SHRINKTRANSFER001 and its GEOMTRANSFER001 ancestor bind actual original basefit,
mean/support/query/cohort/cache hashes. All original parent inputs/output hashes
must pass before scoring; current native cache manifests/shards verified. Old
resource cap violation retained, not changed or invalidated as a score outcome.

Data/access:1313COHORT001base train/100classes originally fit covariance, reused
parameters only/basearrays not loaded now; all383COHORT002valqueries, exactly
three100-source train galleries seeds1701/1702/1703/100disjoint local gloss classes.
All661unique gallery/query pooled target vectors loaded via original full_activity
native768D FP32windows16/stride2/RGB224->FP64weightedmean/L2 recipe. No unused
train/baseval arrays or target fit. Literal class/ID/sourcebyte disjointness,
query-to-fit/gallery participant exclusion, label/gallery order/access/counts
asserted. All five dev people reused; geometry-fit class transfer only, checkpoint
pretraining overlap unknownD034, no semantic ontology/population/causal claim.

For residualR/totalT regularized parentcovariances, eigh returns ascendinglambda
and V. Reject nonfinite/asymmetry>1e-12abs/minlambda<=1e-12; minimum adjacentgap/
maxlambda must exceed1e-12 in BOTH source bases, otherwise technical hold before
any query score; no basis selection/degeneracy fallback. Orthogonality and relative
max reconstruction<=1e-10. RR/TR/RT/TT W_ab=(V_a/sqrt(lambda_b))@V_a.T; commonmean
and L2norm afterward, reject transformednorm<=1e-12. This reweights original
feature axes, not an orthogonal rotation of whitened vectors. RR/TT whitening
maxabsdifference<=1e-12 from saved original matrices and all2298original complete
rankings exact. New/all4596ranks must match direct transformed-space Euclidean
pair differences exactly with ascendingclass tie rule. No numerical relaxation
following outcomes; technical mismatches retained, not method negatives.

Metric unchanged metrics.py43db516f...; official ASLCitizen eval_metrics pinned
17f0148b00ef87a5957ec103da8d10bd8eb1fa23/P001_v1_parity. BinaryDCG1/log2(rank+1),
complete100class ranks/383query everymode/gallery, Top1/5/10/20/MRR secondary.
Primary RR-minus-TR mean across queries/threegallery>=.005 AND all three differences
>=0. Failure stops this orientation explanation's scaling; no replacementsearch.
All four cell means/perseed/fullranks/metrics plus three secondary signed effects
reported. Conditional2000pairedparticipant bootstrap/seed1701/fivegroups on
per-querygalleryaverage/micro weighting; report100xDCG units, not population or
selection-adjusted/independent confirmation. No best-cell/seed/metric narrative.

Exact NPYv2 arrays:two768x768FP64bases, two768FP64spectra,768FP64mean,
661x768FP64targetvectors,4596x100int16classindices,4596x6FP64rowmetrics. Metadata
fixes Cartesian mode/support/query order, original classID map and actualgallery
sampleIDs; stored targetvectors allow independent original-space fullrank replay
without new feature/model calls. No recomputed covariance/eigenfloor/modification
of old artifacts. Complete file plan includes every arrayheader/payload, config,
selected source snapshots/env/status/error/combinedlog/report/metadata/manifest
and atomic status/error replacement scratch; beforewrite max must<=20MiB.
New BoundedCPUContext routes all owned files, fails missing/oversized artifacts;
complete success requires rehashed manifest. Logical bytes exclude allocated disk
and RAM. Existing root bulk reserve fails; only registered bounded small records.
CPU120s/single BLAS thread/GPU0,128MiB planned workspace/2GiB reserve, peakRSSunknown.
Exclusive fresh run/actual argv/PID/config/code/status/errors retained; max2specific
technical retries only, no repeat on scientific failure. S35/S37 covariance
precedents reused, not full architectures reproduced or novelty-absence claim.
