# Evaluation protocol

Primary endpoint is frozen instance retrieval Rec@1 on ASL-Citizen, gallery
official train, queries dev for tuning and test after protocol lock. Cosine
similarity uses normalized encoder embeddings; exact ties sort by gallery ID.
Keep no-gallery-match queries as zero recall/reciprocal rank and report counts.
Report micro Rec@1/5/10, MRR, macro class and signer/source metrics.

Features come from final-stage encoder maps before classifier, spatial pooling
per native temporal cell. Full-interval windows cover the clip; duplicate native
timestamps are averaged once, then valid timestamps are uniformly pooled.
Record sampler/pooling/checkpoint/manifest hashes. The current feature cache is
NPZ for bounded fixtures/pilots; large-corpus sharding remains pending.

This uniform protocol is distinct from official SignRep `fc`/activity readout
and from class-max or prototype ranking used in some previous method folders.
Their existing scores cannot be reused as measured SignCoord-SSL results.

Matched controls share initialization/decoder/data/stats/masks/optimizer/budget.
Compare A7−A3 and A7−A6 using paired per-query results, clustering by known
signer or source. Bootstrap reports number of groups and limitations for fewer
than ten groups. Report seed variation separately; confirmatory ≥3 seeds.
Selection/tuning grids and practical +1 percentage point criterion are in
`experiments/preregistered_plan.yaml`. No real frozen score has been computed.

CTC numerical primitives and a frozen-feature head are implemented/tested;
CTC training/evaluation CLI and translation/probe pipelines remain pending.
No linguistic/nonmanual preservation claim is supported yet.
