# Shortlist — draft, không phải hướng paper đã chốt

2026-10-03, liên kết [H001](H001.md), [H002](H002.md), [H003](H003.md).
Điểm theo weights 25/25/20/15/15% của 04_IDEA_PORTFOLIO, nội bộ cùng agent.
Không có measured dev failure; literature chỉ cung cấp motivation.

| H | Problem | Novelty | Protocol | Compute | Identifiability | Weighted /5 | Hard blockers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H001 | 3 | 1 | 3 | 3 | 3 | 2,50 | Feature/parity, exact cohort và closest low-label work |
| H002 | 3 | 1 | 2 | 3 | 3 | 2,30 | Gallery scorer/activity, cost estimate, storage |
| H003 | 3 | 1 | 1 | 1 | 2 | 1,65 | Raw streams/masks, natural failure annotation |

H001: use case có S10, temporal representation motivation S01/S11; chưa
có mới về method nên novelty=1. ASL CSV/coverage và scorer pass giúp feasibility
nhưng chưa có feature parity. H002 reuse SignRep có thể rẻ, song aggregation
đã có S03/S12 và cost gain chưa đo. H003 có collision L01 và unknown inputs;
giữ hold thay vì implement synthetic masking rồi gọi robustness thực.

Queue: H001 prerequisite trước, H002 analysis backup cùng encoder; H003 hold.
REGISTER gate chưa đạt; không chấm 5 cho idea chưa có evidence. Promote tối đa
một hướng chỉ sau pilot valid. Phonology/spotting/cross-corpus retrieval không
active vì nhãn/ontology/novelty blockers, không kết luận các hướng đó vô ích.

Update2026-10-04: H001 cohort/native decoding and GPU numerical parity gates
now pass (COHORT001/VID002/GPU004). CACHE005 extracting full selected cohort;
VERIFY001 and baseline remain pending. Original scorecard retained as draft
history; no performance-driven re-score/promote. H001 analysis priority, H002
backup, H003 hold. P001_v4 reallocates H0012h within program4h.

D0072026-10-04: H001 registered practical utility screen fails on complete
pilot; kill scaling that recipe, retain negative evidence. H002 becomes active
registration priority, H003 hold. No direction promoted for paper.

D0082026-10-04: H002 joint pilot passes, uniform8 selected under P002_v1.
Continue analysis/confirmatory registration; stride8 already native SignRep
ablations, so no sampling-method novelty or paper promotion. Six policy trials
exhausted. H001 tested recipe killed for scale; H003 hold.

D0202026-10-04: H003 crop/CUDA compatibility validated on four clips; three new
CPU audits resolve annotation source-join and confidence semantics. Natural-error
GT still missing. Continue annotation feasibility using prospective intake;
hold scientific pilot. H001 tested recipe stays killed. H002 original confirmation
and P004 dev retained, fresh all-query test stopped D014; scoped contribution
review open without new policy search. No direction promoted for paper.
Matrix25 rows/23 distinct papers/16 selected-text reviews/seven leads. Original
scorecard and earlier pending descriptions are dated history.

D0222026-10-04: new [H004](H004.md) candidate tested two deterministic query
completion recipes on all383 dev sources/three galleries. Both fail practical
half-budget gate; kill their GPU scaling. No promoted method. H002 retains
the strongest scoped utility evidence but contribution/novelty unproven; H003
natural-error labels missing. Queue moves to source-supported failure/closest-
method discovery before any new proposal; no test or threshold/seed sweeps.
Matrix29 rows/27 distinct papers/21 selected-text reads/six leads, bounded scope.

2026-10-05/D073 supersedes historical priority above. H006 orientation analysis
retained/D072, whitening method scaling rejected against raw/centered controls.
New [H007 draft](H007_DRAFT.md) prioritizes metadata feasibility of finite-gallery
rejection operating-point stability. Three RQs screened in
[intake](../literature/OPENSET_INTAKE_20261005_r1.md): A baseline failure diagnostic,
B known FR-Disc module rejected as novel lead, C known-only conformal OOD
guarantee presently unsupported. No promoted or newly registered hypothesis;
H0072.65/5, scientific failure/novelty unknown. Strong standard controls precede
mechanism design; next census train-only calibration/source/signer availability.
S45–S49 new source cards/matrix49rows. H002 backup utility/H003 annotationhold;
old policy searches/test stop unchanged. No new GPU allocation/quality outcome.
