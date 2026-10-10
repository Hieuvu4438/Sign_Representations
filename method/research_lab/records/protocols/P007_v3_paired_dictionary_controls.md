# P007_v3 — paired dictionaries for the fixed supervised controls

2026-10-05,D087; prospective metadata repair, not unchanged P007_v2/retry.
P007_v2 actual HEADPOOL001 STOP on P38singleclass preserved. No head/task-quality
outcome observed before this change. Full underlying
[P007_v2](P007_v2_supervised_rejection_controls.md) data, two existing heads,
training seeds, losses/optimizer, fixed evaluations/primary/uncertainty/budgets
are inherited except the training episode law/schema below. Explicitly adapted
frozen SignRep discriminator, no exact full FR-Disc benchmark reproduction.

Keep ALL1027 fit clips/100classes/24 signer IDs and286cal/378dev/MS748 partitions.
Generate1000episodes×16pairs, PCG64seed1701, same arrays for every head/initseed.
Uniform query signer from sorted24, then uniform query clip of that signer.
For each pair with trueclass y, choose50 distinct other class IDs uniformly
without replacement: first random ID is replacementclass, remaining49 are shared
classes. Draw one support clip each for y/replacement/sharedclasses uniformly,
excluding query signer and its source-byte group; STOP if any source signer×
class candidate pool empty, no signer/class filtering. Positive dictionary
contains trueclass+49shared supports; negative dictionary replaces trueclass
with replacement and identical49 supports. Sort each dictionary by globalclass
ID for stable FP64 cosine nearest selection. Query clip/video/signer identical
between positive/negative role; no source/self/signer match to any support.

One compact int16 array1000×16×52: [queryindex,true-supportindex,
replacement-supportindex,49sharedsupportindices]. Indices refer to fixed sorted
source roster. Save full plan before fitting, check all roles/class counts/
shared IDs/source/signer exclusion and deterministic re-generation. Numeric
source vectors1027×768 FP64 unchanged. No need two query classes per signer;
single-class signers remain valid. Report query frequency per signer/class,
do not claim all examples equally weighted under signer-uniform sampling.

For each step construct16 positive/16 negative residual rows with separate paired
dictionaries. Positive predictions/targets use sorted localclass indices. Set
negative targets−1. Reorder negative half: counterpart indices of correctly
classified positive queries first, followed by remaining counterparts, each
in originalpair order. Existing firstNcorrect unknown selector then pairs exact
query clips to retained correct-known positives. Wrong-known positives ignored;
no correct-known → skip update/no resampling;1000 attempts remain fixed. Class
selection is a frozen cosine computation, not feedback from learned scores.

All other P007_v2 primary/jointutility criteria remain locked: mean MS CCRgain≥
.02/all3trainseed gains>0/conditionalCCRlower>0, FARmean delta≤.02/FARupper≤.02,
baseCCRdelta≥−.02, at50/FROZEN50/equal3train×3gallery, fixed signer weights2000.
All controls/two calibration policies/three sizes reported; HPO0/no earlystop.
Extra1027 training labels explicit. Same MSval exploratory, freshconfirm unknown.

Prerequisite new pairmanualfixtures CPU60s/GPU0/2MiB, pooledplan CPU300s/GPU0/
20MiB; pair array1,664,128NPYbytes plusvectors6,310,016bytes before metadata/context
and rewrite overhead, full registered plan required. Cost gate real first100
source episodes/job180s/GPU60s/4MiB, discard cost weights/no heldout quality.
Three seed mainjobs120GPUseconds/300jobseconds/20MiB each only after all gates;
H007420s/program14400s ceilings unchanged,2GiBallocator/maxonejob/grantedGPU0/
min3072MiBfree/max100util/inheritedvisibility/all-rootbound accounting unchanged.
