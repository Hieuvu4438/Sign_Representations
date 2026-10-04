# P004_v1 — fixed-policy fresh-vocabulary one-example dev replication

Prospective registration following [D012](../decisions/D012.md); H002. This is
within-corpus replication of the already selected stride-8 recipe, not new sampler
selection or independent language/domain generalization. Original P003 outcomes
and whole-corpus test CSV metadata are known. New selected test pixels, embeddings,
rankings and outcomes have not been accessed. This document registers dev only;
confirmatory test requires a separate prospective freeze after a valid pilot.

COHORT002 uses train-only class seed1704, excluding all original100 exact Gloss
classes. No source/quality/cost-based reselection. Fixed100 fresh classes,278
train-gallery union and383dev queries; 28 train/five actual dev participants.
Each gallery contains one labeled train instance per class; gallery seeds1701,
1702,1703 unchanged. All selected dev sources are evaluated with no exclusions.
VID003 independently of predictions audits661 sources/57577 native frames/
24001 complete16-frame stride2 windows; tails321 frames are dropped. SIZE003
bounded fingerprint screen finds no cross-split candidates;103 within-split
candidates are same-participant/different-class. Not exhaustive uniqueness or
session/pretraining-overlap proof. Existing checkpoint may have unknown overlap.

Unchanged original FP32 native decoder, resize224/ImageNet normalization,
16-frame stride2 encoder,768-D features and hand-activity outputs; strict eval,
TF32 off, batch2, no trainable adaptation. Full raw cache all661 sources enables
CPU fixed-policy simulation. Methods exactly full_mean, full_activity, uniform8,
random8_s2718 and random8_s31415. Uniform selects dense-window indices0,4,8,...;
random recipes retain original per-source SHA256-seeded selection with the same
count. Activity weighted mean and zero-mass fallback/L2 normalization unchanged.
Class cosine max over train instances, descending score/ascending class-ID ties.
No dev-fitted transform or new HPO; six original H002 trials remain exhausted.

Primary: micro DCG1/log2(rank+1), averaged over three fixed galleries, uniform8
minus full_activity. Pass only if mean difference >=−.02, every gallery loss <=.02,
and paired participant-bootstrap95% lower bound >=−.02. Bootstrap2000 replicates,
seed1701, micro weighted over participant groups; five actual groups, conditional
on fixed classes/checkpoint/galleries, not population equivalence. Also require
at least2× fewer selected query windows. Full_mean/both random controls and
Top1/5/10/20/MRR are descriptive, all15 complete383×100 rankings retained.
No best-policy selection, no tuning if pilot fails, no random control promotion.
Cached selected-window counts are structural cost only, not latency measurement.

Valid positive pilot permits separate test registration/preparation; it does not
permit test GPU until exact source/overlap/window/count/cost checks pass. Valid
negative pilot stops this replication. Technical failure preserves all outputs
and source/config snapshots, requires a new run and explicit engineering erratum,
at most two technical retries; scientific margin/policy/classes cannot change.
Output directories exclusive; no partial resume. Existing P003 sources/configs
and completed runs immutable. Public scorer parity VAL001 remains applicable.

Budget: original program14400s; D009 H0027200s. Prior H0022455.344824807s,
remaining4744.655175193s. SIZE003 full dev-first planning4710.633480888s, only
34.021694305s estimated slack; not guaranteed runtime or a test launch gate.
CACHE006 extraction estimate1053.499446762s, separate1500s hard job ceiling
(including setup/persistence). Every owned GPU attempt/failure charged to H002
and program ledger; no budget increase. One owned GPU0 job, granted UUID
GPU-3775c606-49da-34fe-322c-756749074f5b, preserve scheduler visibility/others,
min8GiB free and8GiB allocator cap. Shared eligible utilization<=100%, timing
limitations reported. Full raw payload74115088B plus bounded overhead in/dev/shm,
1GiB cap,20% tmpfs/RAM reserve+2GiB CPU workspace. No installs/source deletion.

Same-agent sequential reviews, not independent:

- Researcher: replication tests whether selected recipe survives fresh vocabulary
  and lower gallery budget. Does not establish novelty of existing stride8.
- Engineer: separate wrappers reuse frozen core; four manual software integration
  checks cover one-example cosine oracle, leakage/count rejection, per-gallery
  and group-uncertainty failures, full ranking/dev schema completeness.
- Protocol Auditor: train/dev identity preserved; no test aliasing; hash-bound
  source/audit/config/checkpoint/scorer/native adapter; original P003 preserved.
- Statistician: fixed primary/all-gallery/conditional group-CI rule before scores;
  no selection adjustment claimed and no independent three-gallery population.
- Skeptical Reviewer: same corpus/checkpoint/populations, unknown pretraining and
  sessions, narrow100-class task; dev positive alone cannot satisfy paper gates.
