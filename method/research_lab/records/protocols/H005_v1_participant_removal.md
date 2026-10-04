# H005_v1 — Held-gloss retrieval under participant-associated removal

Registered2026-10-04 before task outcomes, Codex, H005/D025; diagnostic,
exploratory reused train/dev. **VALIDATED for bounded CPU measurement** via
unchanged P001/VAL001 scorer and PARTREMTEST001 four new oracle groups.
This is a known projection-family control, not a new method or official benchmark.

## Fixed data, labels and context

Exact COHORT001/VID002/BASE002 identity/CACHE005 contract inherited from H005_v0.
Cohort SHA d3327a955efce9128fc10f1e13e5daae884ba0abe8c0920ff48dcc1339b25830,
audit5f93826b211e1d45667c8dc3b8b352b6c5a34644a60218c1e0dc5f77760ca0fe,
cache manifest4b2217203fb25af9110226e06cb56f7171ae4b46b111b9bf77c049c33ab9e737,
checkpointf8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b.
All1691 source arrays native/full activity pooled exactly as v0, no encoder call.
1313train,378val/five val participants. No test paths/features/labels, H002 policy
retuning or P004 short-source rescue. Source/session/pretraining overlap unknown;
reused dev and prior research exposure means no confirmatory novelty claim.

Same SHA(seed1701:gloss)50/50 split as v0; two directions. Each direction uses
all650/663 train sources of its fit50glosses for fitting transformations.
The other50glosses define retrieval vocabulary; all188/190 corresponding val
queries evaluated. Their official participant IDs are disjoint from train.
Each val source evaluated once per mode/gallery; original class IDs preserved.
No query labels/features in fitting. Train-only selected supports: one source
per retrieval class from existing COHORT001 seeds1701/1702/1703; exactly50gallery
instances each. Same supports for all modes. No shared source hashes across
fit/evaluation or support/query, no silent filtering/fallback.

**Supervision disclosed:**650/663 fully gloss-and-participant-labeled base-class
training examples plus50 new-class labeled supports per gallery, not a globally
one-shot-only system. Raw/center/PCA/random/participant modes have the same
allowed base/support access; raw ignores extra fitting labels. PCA and participant
use identical base gloss labels for residualization. No claim that extra labels
are free, no direct score comparison to original100class H002 or official release.

## Prespecified five transformations

One fixed rank8 (8/768dimensions; modest rank screen, not selected from scores),
one recipe, no hyperparameter sweep. For fit rows X, clip-weighted gloss mean
subtracted to give R. Participant matrix C has each fit participant's average
R as one row, then columnwise subtract its unweighted person mean. Leading8
right singular vectors U_id of C define participant-associated subspace.
Generic residual PCA U_pca uses leading8 right singular vectors of the identical
R. Random U_rand is QR of a768×8 Gaussian matrix, fixed RNG seed1701+fold.
All non-raw modes center by the identical fit clip mean mu.

- raw: L2-normalize x, no shift/projection.
- center: normalize(x-mu).
- random/PCA/participant: normalize((x-mu)-(x-mu)UU^T).

Singular value8<=1e-12 or output norm<=1e-12 is a technical failure, never rank
retuning or source exclusion. Orthogonality tolerance1e-10. All trained means/
bases/singular values/fit IDs saved. Basis sign ambiguity leaves projection
unchanged. No sparse sampler, decoder or encoder parameter change.

Fit centroids for the secondary participant probe on the same fit gloss rows,
evaluate only opposite-gloss train rows under each transformation, using v0
strict missing/degenerate/tie denominator. Dev participant identity is never
learned. This secondary observation remains a reused-person proxy.

## Ranking, metric and gate

Exact cosine lookup of transformed queries against50 one-example supports;
complete ranking of original50class IDs, descending score/ascending-ID ties.
Scorer `metrics.score_ranking/evaluate`, unchanged SHA
43db516fa029b20986130cae6b98e95e500cefa372d15ae4c020130b9f56c79c;
binary DCG=1/log2(true_rank+1), no nDCG conversion. Official reference commit
17f0148b00ef87a5957ec103da8d10bd8eb1fa23 `I3D/aslcitizen_testing.py:eval_metrics`;
P001_v1_parity.json/VAL001 tested ranks/ties/coverage at1e-12. Retrieval gallery
and50class partition are adapted diagnostics, not the official protocol.
New PARTREMTEST001 checks manual projection, zero/rank failure, class offsets,
orthogonal control and noncontiguous-ID rank3DCG. No scorer changes.

Primary: DCG mean over all378 queries and three gallery seeds, matched per-query
differences. Continuation requires all: participant minus raw>=.01 DCG;
participant minus **each** center/random/PCA>=.005 DCG; no negative raw difference
for any of the three gallery-seed aggregates; secondary balanced participant
recall drops>=.05 against center mean over both train folds. Failure stops this
fixed removal recipe, not every signer-related method. Thresholds are prospective
screening choices, not population significance or literature-derived effect sizes.
Top1/5/MRR, per fold/seed values and all complete rankings reported; no best mode
or seed selected. H005 trials v0 one probe + v1 one fixed recipe with four controls;
H002 original six-policy budget remains untouched, not an extra sampling trial.

Conditional paired participant bootstrap2000reps/seed1701, average gallery DCG
differences per source then resample five val participant groups with repeated
clips weighted microscopically. Report every fixed control; intervals in pp
(100×DCG difference), not per-video IID/population/causal confidence. Small five
groups, shared supports/folds, prior dev reuse and empirical rank choice are
limits. No p-value, multiplicity-based significance or linguistic preservation.

CPU180s/single BLAS thread/GPU0/no allocation,64MiB planned RAM + existing reserve;
<20MiB durable outputs/snapshot, root bulk reserve remains failed. Exclusive
runtime; no overwrite/restart of completed outputs. Config binds current code/
tests/protocol/literature/cache/scorer SHA before launch. Later change needs new
version. Success would permit further independent protocol/novelty review only;
no automatic GPU scale or paper promotion.
