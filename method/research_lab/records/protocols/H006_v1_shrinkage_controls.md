# H006_v1 — Base-only shrinkage geometry controls

2026-10-04 prelaunch registration, H006_v1/D029; VALIDATED for adapted diagnostic
by SHRINKTEST001 three mathematical/error oracle groups and unchanged official
binary-DCG scorer parity P001_v1_parity.json. No official benchmark reproduction.

## Fixed data and access

Exactly H006_v0/GEOMTRANSFER001 completed fit/support/query contracts. Config
hash-binds parent status/config/parameters/report/predictions/metadata and original
input hashes, source snapshots authoritative. All original hashes must pass.
Base COHORT001 all1313train/100classes, target COHORT002 all383val queries/
100new local gloss classes, three train galleries100sources each/seeds1701–1703.
Literal gloss/sample/sourceSHA disjointness, train/query participant exclusivity,
exact support IDs/class order/load union and whole-cache integrity are asserted.
Allfive val people were used earlier: no independent domain/population/linguistic
ontology/pretraining-disjoint claim. Base378val and target unused train arrays
not loaded/fitted; metadata/cache-shard hashes can be verified without array use.
Fully gloss-labeled1313base clips plus100target support labels per gallery are
allowed. Encoder frozen, fit inductive/base-only; not unsupervised or globally
one-shot-only. Native SignRep768D/RGB224/window16/stride2/savedFP32/full_activity
FP64 mean/L2 vectors unchanged. No test, new encoder, GPU or H002 search trial.

## Six standard controls and two exact prior references

X base unit clip vectors; global mean mu. True residuals X-minus-own-gloss-mean;
total X-minus-mu; shuffled residuals use default_rng1701 permutation of the same
base class labels/counts, exactly matching H006_v0. Each estimates a covariance
with installed sklearn1.8.0 LedoitWolf(assume_centered=True,store_precision=False).
C=R^T R/N (N1313), s automatic fourth-moment shrinkage, tau=trace(C)/768,
Creg=(1-s)C+s*tau*I. No adjustment N-minus-classes, lambda sweep or target tuning.
Local implementation/dependency files/version hashes bound; online docs1.9.1
are API precedent, not the version executed. Group/shared-mean dependence means
no optimal-shrinkage/statistical consistency claim from the control.

Symmetric eigen-decomposition gives W=Creg^(-1/2). Reject nonfinite/asymmetric
covariance (>1e-12 abs), min eigenvalue<=1e-12, nonfinite transform or zero cosine
norm<=1e-12; no eigenfloor/fallback/rank change/exclusion. All fitted matrices,
eigenvalues/coefficients/means/labels are saved. Whitening identity W^T*Creg*W
max absolute error must be<=1e-8. Pointwise z=(x-mu)W for every support/query.
Each family gets two preregistered metrics:
- *_lw_cos: L2-normalize z, cosine ranking.
- *_lw_maha: do not L2-normalize; squared Euclidean z distance, equivalent to
  common-covariance Mahalanobis in the original pooled vectors. Mean cancels
  from pair differences; keeping it identical isolates final norm/metric.

Original raw and rank8 residual-PCA modes use the frozen saved v0 parameters,
not a new fit/search. All2298prior full rankings and aggregate DCG must match
v0 exactly (floating metric<=1e-12). Their repeats are consistency checks, not
new selection trials. Eight modes×383queries×3galleries=9192complete100class ranks.
Every ranking independently recomputes from direct Euclidean pair differences
in the transformed space, with exact agreement required; tie ascending class.
Unit vectors make cosine and Euclidean ranking equivalent. Numeric mismatch
is technical failure, retained, not scientific loss or post-outcome relaxation.

## Metrics, falsifier and uncertainty

Unchanged metrics.py SHA43db516fa029b20986130cae6b98e95e500cefa372d15ae4c020130b9f56c79c,
official ASL Citizen eval_metrics pinned17f0148b00ef87a5957ec103da8d10bd8eb1fa23,
existing manual/parity fixtures. Binary relevance DCG=1/log2(true_rank+1), all
100classes/no nDCG; all383query IDs eachmode/seed, no silent denominator changes.
Primary mean across383queries and3galleries. Pass only if residual-PCA minus
**both** residual_lw_cos/residual_lw_maha>=.005 and all gallery differences against
both>=0. Raw and all other controls/Top1/5/10/20/MRR/full per-seed metrics retained.
This tests practical hard-rank-removal advantage, not mathematical novelty.
Failure stops scaling that superiority claim; no retuned rank/coefficient/seed.
Success permits further contribution review, not automatic GPU/method promotion.

Secondary class-conditioning comparisons: each residual LW metric minus matched
total/shuffled LW metric. Four signed deltas/conditional bootstrap intervals all
reported, no best-metric selection or old-gate rescue. Fixed2000paired participant
bootstrap seed1701 over five dev IDs using gallery-average query differences and
micro clip weighting. Allsix PCA-vs-LW intervals also retained. 100×DCG units,
same-corpus/reused people/fixed base and supports; no population/causal inference,
selection correction, independent confirmation or optimizer-seed variability.

## Provenance, validation and compute

Three SHRINKTEST001 groups pass: independent fourth-moment coefficient/covariance
oracle and whitening identity; analytic diagonal covariance/distance plus a known
case where cosine and Mahalanobis choose different classes; degenerate/asymmetric/
zero/nonfinite rejection. CLI help passes. Original scorer/native/P003/H005/H006
files remain unchanged. Source S35 is Mahalanobis precedent, S36 conceptual only
because its author-v2 footnote reports code/results errors, S37 class-conditioned
frozen-feature whitening. These controls do not reproduce the full Simple CNAPS,
LFD-ProtoNet or FROST architectures/training/tasks.

CPU180s/single BLAS thread/GPU0/no HPO,128MiB estimated workspace/2GiB reserve;
<20MiB lightweight records permitted while root bulk20%reserve fails. Exclusive
RunContext actual argv/source snapshot/env/status/alarm/walltime/output digests;
failed attempts retained/max2technical retries with specific repairs, no repeat
on scientific failure. Fit/rank checks are same-agent software verification,
not independent scientific review. Peak RSS not currently instrumented.
