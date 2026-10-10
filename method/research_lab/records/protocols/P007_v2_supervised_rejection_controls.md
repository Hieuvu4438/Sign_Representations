# P007_v2 — supervised frozen-feature residual controls

2026-10-05, D086. Prospective exploratory strong-baseline comparison, not a new
method or unchanged H007_v1 protocol. Separate output/config/run identities.
Question: can an existing trained residual discriminator improve joint
source-calibrated recognition/rejection over fixed scores? No target-quality
tuning; all existing score controls reported. H007_v1 originals immutable.

## Allowed data and representation

Use exact OSMETA001 partitions: source fit1027 gallerypool clips/24 official
signer IDs, source calibration286/8 IDs, base dev378/5 and MS validation748/19.
Fit/cal/dev ID/signer/source-byte sets must be disjoint; STOP if not. Both ASL,
source classes are exposed to head fit; MS ontology/pretraining/physical signer
overlap unknown. No unseen-source-class or fresh-test claim. One-shot describes
dictionary supports only; extra1027 training labels +286 calibration labels
explicit. Current target cached partition HOLD, no reselection/extraction.

Reuse native SignRep caches and completed1037-vector MS feature run. Source
pooling identical activity-mean/unit normalization in FP64, save1027×768 FP64.
Frozen encoder/features/cosine predictions unchanged. Training and inference
residual is FP32(query unit vector) minus FP32(nearest support unit vector).
Nearest class selected by saved/exact FP64 cosine with ascending-global-ID ties;
no learned representation/classification/normalization or posthoc rank change.

## Training episode plan before head learning

1000 fixed episodes from source fit pool, PCG64 seed1701, shared by every head
and initialization seed. Each has50 classes/one support per class,16 known and
16 unknown queries drawn with replacement. Sort all global IDs and signer IDs.
Uniformly choose a query signer; uniformly choose two distinct classes available
for that signer as mandatory known/unknown anchors. Choose48 other classes
without replacement from remaining98, add known anchor/exclude unknown anchor,
sort dictionary IDs. Uniform support clip per class excluding query signer and
all that signer's source-byte groups. Uniform known/unknown query clip sampling
from that signer's dictionary-present/absent clips. STOP if ANY signer has fewer
than two classes or any signer×class support pool is empty; no group filtering.
This gives matched query signer for positive/negative roles and no self/source
or official-signer match to support. Training sampling is a declared adaptation
of author's loader (not exact original episode law). Save1000×82 int16 ID-index
plan before numeric head fitting; deterministic schema/coverage checks required.

## Existing heads and objective

Two heads, same signed residuals and episodes:
1. Linear 768→1,769 parameters.
2. Author SAFSAR discriminator architecture768→1536→768→64→1, ReLU between
   hidden layers,2410881 parameters, no BatchNorm/dropout. BSD author attribution.

Own module emits pre-sigmoid logits; sigmoid matches pinned author component.
Stable BCEWithLogits has mathematical BCE equivalence in nonsaturated fixtures;
not exact full author optimizer/numerical/paper reproduction. Keep only correctly
classified known positives and first equal number of unknown negatives in
original query order; wrong-knowns ignored. No correct-known → skip update for
both heads, record count, never resample for performance. Exact source known
predictions fixed across models/seeds. Adam lr.001/weight_decay.0001, defaults
betas(.9,.999)/epsilon1e−8, 1000 episode attempts, one update per nonempty batch.
No CE/encoder gradient; author's joint CE+BCE adaptation is not reproduced.
Three fixed initialization seeds1701/1702/1703, all shared training episodes and
all three evaluation galleries per seed. No dev early stopping, best checkpoint,
hyperparameter/grid search or target fit; HPO0. Final weights only, flattenFP32
with explicit tensor schema, no pickle. Log loss/update counts for every episode.

## Evaluation and prospective primary

Exact P007_v1 three support seeds/sizes10,25,50 and1412 query rows; existing
four fixed scores reused, two learned logits added. Source-known calibration
unchanged: highest observed threshold accepts at leastceil(19*Nknown/20), >=ties;
FROZEN50 and MATCHED_SOURCE_SIZE. Scores now include learned-logit methods and
extra supervised fit; do not call comparisons equal unsupervised label budgets.
All source/cal/base/MS metrics/FAR/CCR/known acceptance, curves and known fixed
retrieval metrics reported. Corrected FP64 DCG computation from outset, no
mutation to original FP32 fields. At50 policies identical by construction.

Primary MLP-minus-maxcosine on MS at50/FROZEN50: equal mean across3 training
seeds×3 support seeds. Continue baseline utility screen only if mean CCR gain
≥.02, all3 training-seed support-mean CCR gains>0, and conditional95% CCR-delta
lower>0; unknown FAR nondegradation requires mean delta≤.02 AND conditional95%
FAR-delta upper≤.02. Base mean CCR delta must be≥−.02. This conjunctive screen
prevents calling acceptance-only gains improved rejection. Report all existing
margin/MSP/NAC and linear controls; any gate pass remains existing-method utility,
not novelty/strongest-baseline victory or promotion. Fail → no same-MS rescue
tuning of this fixed head recipe; full jointly trained FR-Disc remains untested.

Bootstrap2000, fixed existing P007_v1 MS/base signer multinomial weight arrays,
same weights across heads/training/support seeds. Parameters/supports/calibration
fixed, equal mean of per-seed ratios, not pooled queries. Use validated cluster
ratio implementation; invalid draw → unavailable CI/no screen pass, no dropping
or replacement. Three training seeds descriptive finite variation; no population
or unconditional head-training uncertainty guarantee. Numerical recount on saved
logits, separate CPU weight replay within combined1e−4, do not claim exact
cross-device threshold/rank equality from tolerance alone.

## Prerequisites/resources/stops

Separate registered component/manual AST/loss/gradient/sampler fixtures CPU≤60s/
2MiB; source pooling/episode plan CPU≤300s/GPU0/20MiB; GPU real-source100episode
cost/parity gate job≤180s/GPU≤60s/4MiB (discard all cost weights, no calibration/
dev/test quality). Cost estimate train1000 plus full evaluation must fit each
120s allocation before three main jobs. Head fit GPU≤120s/job≤300s/20MiB per
initialization; H007 aggregate ceiling420s within unchanged14400s program ceiling.
GPU0 only, inherited visibility, maxone owned GPU job/min3072MiBfree/max100util/
2GiB allocator; fresh effective all-root ledger with bound OpenGL erratum.
No other job changes; CPU workspace≤512MiB in tmpfs plus2GiB reserve. Root bulk
reserve fails, only bounded stores, no new bulk extraction/downloads/install.
All file/header/config/source/log/manifest/status-rewrite maxima preregistered;
reject plan exceeding original cap before output creation. Failed runs retained.

Cost gate reports CPU/GPU initial forward and loss-gradient parity at combined
1e−4, actual peak allocated/reserved bytes, GPU/job wall and100-step source cost;
no checkpoint reuse in quality runs. Full20MiB plan must contain9,646,600 raw
parameter bytes plusheaders/logits/metrics/context/manifest/rewrite overhead.
No quality launch without actual fixture/pool/cost passes and SHA-bound config.
Separate final analysis/recount registration after all three runs terminal.
