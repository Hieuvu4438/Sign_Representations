# P004_v1 — prospective fresh-vocabulary one-example test replication

This prospective specification is prepared while CACHE006 train/dev extraction
runs, before new dev scores and before selected COHORT002 test access. It is
conditional on a valid positive [fixed-policy dev pilot](P004_v1_dev_pilot.md).
Actual freeze timestamp/config/code hashes and positive pilot evidence must be
issued in P004_v1_test_freeze.json before new selected test pixel/outcome access.
Old P003 outcomes and global CSV metadata are known. No new method selection.

The100 exact Gloss classes selected from train by COHORT002 seed1704 are fixed
and disjoint from original COHORT001 classes. All official test rows with exact
Gloss in COHORT002 must be evaluated; no length/quality/source-driven exclusion,
subsampling or class reselection. Held-out video IDs, bytes and participant IDs
must be disjoint from selected train/dev; official whole split ID/participant
contracts also checked. Full source-byte/PTS/frame/window audit, five-thumbnail
bounded cross-split near screen before GPU. Near candidates or coverage faults
stop measurement pending review, never silently remove sources. Sessions and
checkpoint pretraining overlap remain unknown; same corpus/participant population
as P003, not independent language/domain/generalization. Adapted100-class task,
not official2731-class leaderboard. Test audit ceilings2500 queries,8GiB logical
reads,2000 frames/source and1800 CPU seconds; exceeding any stops all-source
measurement. Numeric ceilings are workload guards, not an exclusion rule.

Freeze unchanged native adapter/checkpoint/public scorer, fixed one-train-example
galleries1701/1702/1703 from verified CACHE006. Methods full_mean, full_activity,
uniform8, random8_s2718, random8_s31415; existing FP32/batch2/TF32off recipes and
768-D pooled feature dimension. Activity pooling and exact max-instance class
cosine unchanged. Fixed lexical-first selected train source
0056041416092114105-INFECTION.mp4 for warmup;4 native windows; warmup excluded from
query timers but charged to allocated GPU wall. This is a new non-test warmup
identity for the new cohort, not a tuning choice or an identical cross-cohort
timing condition. No new HPO; six original H002 policy trials exhausted.

Primary quality: DCG1/log2(rank+1), micro over every selected test query, mean over
three fixed galleries. Candidate uniform8 minus full_activity must have point
>=−.02, every gallery loss <=.02 and conditional paired participant-bootstrap
95% lower bound >=−.02 (2000 replicates,seed1701). Bootstrap all actual selected
test groups; conditional micro participant resampling, not population equivalence.
No independent-gallery variance inference. Top1/5/10/20/MRR, controls and slices
descriptive; no alternative random recipe selected if uniform fails. Full_mean
shares dense forward for quality; no independent latency claim.

Primary cost: actual selected windows at most half dense and median paired
uniform8/dense independent full query-pipeline latency <=.8. Joint success needs
all quality and cost rules. Actual sparse windows encoded before pooling/search,
per-source sparse/dense feature/activity/center parity every query at original
atol1e-5/rtol1e-4. One independent decode/preprocess/encoder/pool/three-gallery
search observation/query/recipe, seed1701 permutation then cyclic ordering by
sorted query ordinal; OS cache uncontrolled/shared eligible GPU. Setup, gallery
extraction/index construction, hashes, parity, full_mean secondary work and
persistence outside per-query timers; all GPU allocation costs counted. No
cached quality timing interpreted as sparse encoder speed.

Train-only quartiles over278 fixed gallery-union sources from REPQUAL001 freeze
length/native-window and effective-activity-fraction slices: Q1<=q25, Q2(q25,q50],
Q3(q50,q75], Q4>q75. All four bins/denominators retained, empty bins reported.
Descriptive per-participant differences. Gallery1701 Top1 categories both success,
both failure,dense only,sparse only; deterministic SHA256 seed1701/category/source
ranking, up to5 each. Numeric rankings/proxies only; no human sign-content or
linguistic claim without separate evidence. All15 complete rankings plus pooled
arrays independently regenerated/verified before scoring.

Exact test-frame/window/count estimate uses original P003 frozen representative
p90 encoder/decode-preprocess/pool-search rates, safety1.5+setup60, unchanged.
After source audit, reject GPU launch if estimate exceeds actual cumulative
H002 remaining allowance or job ceiling. Allocation14400 program/7200 H002,
no increase; every failed attempt charged. Max one owned GPU0 job, persistent user
grant, scheduler visibility preserved,8GiB free/allocator cap,1GiB bounded tmpfs
20% reserve+2GiB CPU workspace. Inference timeout capped by remaining allowance.
No automatic resume; exclusive outputs, at most two diagnosed engineering retries,
no scientific config/recipe/margin/class changes after outcomes. Valid negative
pilot stops scaling; insufficient compute stops before test GPU; negative test
retained as replication failure. No automatic PAPER_READY regardless of result.

Same-agent sequential role review, not independent: Researcher accepts narrow
replication value only; Engineer confirms every new gallery consumer explicitly
uses budget1 and preserves original timing/scorer/pooling; Protocol Auditor
requires positive fixed pilot/hash freeze before selected test access and exact
post-access cost gate; Statistician retains all-gallery/paired group quality rule,
fixed cost thresholds and descriptive secondary scope; Skeptical Reviewer retains
existing stride8 prior art, same-corpus and unknown-overlap limitations. Two new
software oracles reject scientific specification changes and unbound/mutated
config/source/nonpositive-pilot freeze; original scorer/prediction oracles remain.
