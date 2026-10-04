# H005_v0 — Cross-gloss participant diagnostic

Registered2026-10-04 before features/outcomes; **VALIDATED diagnostic scorer**
through PARTTEST001, four independent oracle groups plus installed scikit-learn
parity. No official sign benchmark, leaderboard, retrieval or nuisance claim.
Owner Codex; H005/D023. Train-only, frozen RGB SignRep ASL, offline clip proxy.

## Data/access

Reuse COHORT001 (SHA d3327a955efce9128fc10f1e13e5daae884ba0abe8c0920ff48dcc1339b25830),
VID002 (5f93826b211e1d45667c8dc3b8b352b6c5a34644a60218c1e0dc5f77760ca0fe),
BASE002 identity and CACHE005 report/manifest. Manifest SHA
4b2217203fb25af9110226e06cb56f7171ae4b46b111b9bf77c049c33ab9e737;
checkpoint f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b.
Exact complete cached train features only:1313sources/32IDs/100glosses.
Cache integrity hashes also verify dev shards; dev feature arrays/labels are
not used for fitting/scoring. No test input/feature/prediction/quality access.
Pretraining/source/session overlap remains unknown; dataset release/consent
constraints inherited, no output sharing. IDs are official metadata, not
linguistic annotations or verified biological identity.

## Fixed feature/split/probe

Native16-frame stride2, RGB224, saved768D FP32 features and2 raw activities,
complete timestamp/array/source identity checked by CacheReader. Existing
unchanged `pooled_vector(full_activity, all_windows)` computes FP64 weighted
mean then L2 unit vector, native fallback retained. No encoder call, training,
alternative layer/pooling/policy or GPU. Offline source windows are not causal.

Sort100 glosses by SHA256(`1701:gloss`), lexicographic tie, first50 vs last50.
Two fixed directions fit on one half and evaluate on the other; disjoint source
bytes asserted. Each train source is evaluated exactly once. Expected metadata:
650fit/663eval and663fit/650eval,31fit and31eval IDs each; fit-ID source coverage
.995475113122172/.9984615384615385. Coverage is reported, no rows dropped.

Mean of unit clip vectors per fit participant, then normalize mean if norm>1e-12.
Degenerate prototypes omitted from candidates and explicitly counted. Candidate
IDs sorted lexicographically; exact score tie takes first. Missing participant
prototype cannot be predicted and contributes zero recall, not missing denominator.
Cosine dot-product with each fit centroid chooses one ID per eval source.
Prediction schema includes sample/gloss/participant/predicted-ID/fold/correct;
complete sorted unique train IDs required. No video self-match across folds.

## Metrics/controls/gate

Primary: per fold average of per-participant correct/total over **all IDs present
in that fold's evaluation**, then unweighted average of both fold scores.
Not all32-ID fixed-denominator macro, not average-query accuracy, not retrieval.
Micro/per-ID denominators, fit/candidate coverage and train-majority predictor
reported separately. Majority uses fit counts only, lexicographic tie.

100 deterministic controls, RNG NumPy default_rng(1701), sequential draws.
Within every gloss permute the participant labels; preserve exact ID frequencies,
features/split and sample counts. Fit and score each relabeled dataset under
the same rule. All100 fold/mean scores retained. No seed/permutation selection.
Primary effect is real mean minus median of control means; continue screening
if effect>=.15 and both candidate coverages>=.95, otherwise stop this recipe.
180s CPU/0GPU, no HPO. Failure never becomes a negative method result.

Control range and empirical2.5/97.5percentiles are descriptive conditional
comparators, **not** a confidence interval or randomization p-value. Repeated
people/videos within gloss and dependence across folds prevent IID/population
claims. No causal removal of signer cues, hubness measurement, usefulness for
Deaf users, linguistic invariance, crop quality or downstream gain inferred.

## Scorer audit

Own `participant_probe.score_participants`: strict nonempty/equal lengths,
finite registered feature asserts, no denominator dropping. Specification:
[scikit-learn balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
defines mean recall; installed source audited (confusion matrix diagonal/row
sum then mean, adjusted=False). Local installed version/source hashes recorded
in PARTTEST001_reference.json, no upstream commit claim.

PARTTEST001 known cases: perfect1/allwrong0;9A+1B allA micro.9/balanced.5;
identity-only transfer1; lexeme-only exact-tie control.5; absent-ID coverage.5
and balanced.5; degenerate mean counted as wrong; within-gloss multisets preserved;
fold/order and input errors. Float oracle/parity exact on representable values.
Log `method/frozen_probe/runtime/PARTTEST001_H005v0_s1701.log`. One same-agent audit,
not independent review. No published signer-scoring release reproduced.

Immutable code/protocol/input SHA bound in PARTPROBE001 config before run.
Resume semantics exclusive creation; terminal run never overwritten. Change
split/scorer/gate requires new version. No changes to P003/P004 scientific gates.
