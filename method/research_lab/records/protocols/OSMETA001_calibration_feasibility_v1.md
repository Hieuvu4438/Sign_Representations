# OSMETA001 v1 — prospective calibration metadata feasibility, H007 draft

2026-10-05; diagnostic data audit only, not a scientific quality pilot. Original
H001–H006 outcomes/recipes intact. Freeze before first OSMETA execution.

Allowed inputs: terminal CACHE005/BASE002/COHORT001 metadata/window-cache files;
terminal CACHE006/REPQUAL001/COHORT002 metadata/window-cache files; complete
MSFULLFEAT001 config/shard records/hash-bound pooled files and MSDICT001 manifests.
All files content-bound by the new config. Hash payload bytes without decoding
NPZ/NPY, numerical feature reads, videos, test features, rank metrics or thresholds.
ASLCitizen_base1691 and target661 selected native caches are inventories, not full
official corpora. MSASL1037 extraction is the previously frozen289train/748val
union. No target collection/stride/crop/normalization or class-vocabulary change.

Normalized rows retain role/sample/class/split/signer/source/hash/window count
and an exact feature-file reference. Class IDs separate per role; ASL names
retained as publisher labels, not an ontology mapping to MS-ASL. Source grouping
is exact byte SHA for ASL Citizen and publisher parent for MS-ASL. Physical signer,
session and checkpoint-pretraining identity remain unknown. All-row unique joins,
split/group/source/hash exclusivity and native shard contract must pass. No errors
silently excluded and no smaller vocabulary chosen to pass.

For each ASL cache: sort distinct train signer IDs by SHA256 of
`H007-calibration-v1|ROLE|SIGNER`, lexical tie fallback. First
`max(1,floor(number_of_train_signers/4))` groups form calibration; every other train
clip forms gallery pool. No train clip discarded. Dev unchanged, never calibration.
Report count/class coverage and source/signer independence. Missing pool class
makes proposed support infeasible; save the missing list, do not retry groups.

Only if all100 pool classes present, freeze prospective supports for seeds
1701/1702/1703 by minimum SHA256 of
`H007-support-v1|ROLE|SEED|CLASS_ID|SAMPLE_ID`, lexical fallback. One/train/class.
Class nesting sorts100 IDs by SHA256 of `H007-class-v1|ROLE|SEED|CLASS_ID`, string
tie fallback; use first10/25/50 and output each gallery in ascending class ID.
MSASL reuses each MSDICT001 support exactly and the same nesting rule; does not
fit calibration on the union of prior supports. No target labels added for fitting.

Every calibration/dev query receives membership counts for each candidate gallery.
Common known anchors are first10 classes and persistent absent classes last50;
these labels remain fixed across nested sizes. All query IDs retained, including
classes whose membership changes as distractors are added. Counts are metadata,
not FAR/CCR. Missing calibration classes reported, no inferred performance.

Fixtures OSMETATEST001: five manual groups covering exact partition/nesting/counts,
missing class/common-unknown change, duplicates/types/split/source/signer leakage,
ASL cache path/join/accounting, MS native source/shard joins. Stop on failure.
OSMETA001 after passing fixtures: all above joins/partition/counts/hash-only audit.
OSMETAVERIFY001 after completion: independent direct source-metadata count/support
selection/membership recount and bounded-store verification; do not import main
partition/census implementation. Same-agent arithmetic, not external review.

Resource caps: fixtures60s CPU/GPU0/2MiB complete store; census300s CPU/GPU0/8MiB;
recount300s CPU/GPU0/8MiB. Complete declared source/config/environment/status/log/
output/manifest caps and replacement peaks must fit before launch. ~253MiB existing
ASL payload and ~9MiB MS files read/hash in place; no copies/new cache, one thread.
Root20%bulkreserve failure remains; only bounded small metadata stores under prior
authorization. Original GPU ledger unchanged. No encoder or stochastic fitting.

Gate: validated metadata+available disjoint calibration and all100 pool classes
permit exact H007 quality protocol registration next; subset failures go HOLD.
Metadata PASS cannot establish baseline failure, novelty, scientific success,
threshold guarantee or confirmatory readiness. Dev already exposed remains
exploratory. Future evaluator/negative-label/calibration/cost/test gates separate.
