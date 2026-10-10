# ASLC001 post-implementation code review

Local source identity is dataset-scoped; no invented YouTube IDs. The corpus path
requires an official train reference and the complete val/test exclusion manifest.
Pretraining source_split must be train, source identity/path must match the train
reference, and held-out identity/path/content pins are rejected when supplied.
Prepared manifests, reference/exclusion files and corpus bone tables are hashed.
A local-corpus audit cannot be labeled the original paper_recipe.

The original YouTube/WLASL protocol remains supported. There are31 passing full-suite
tests after adding local-corpus tests, and8 affected downstream/local tests after
BF16 feature transport support. CPU FP32 paired-training replay remains bitwise equal
for next batch/encoder/discriminator/EMA/metrics; the model-state hash matches the
initial reconstruction regression. Evidence: tests.log,affected_tests.log,
resume_regression.log and the refreshed evidence/S4_training_smoke.json.

Actual CUDA BF16 gradients and Adam updates pass3 resource-calibration steps. Peak
allocator reservation2022MiB; warmed updates0.31-0.36seconds before real-dataset loading.
Dense stride2 extraction from a real71-frame video produces28 windows, float32 cache,
finite unit-norm weighted/unweighted pooled features, peak600MiB reservation. This is
resource/interface evidence only, not a retrieval experiment.

Corpus extraction uses persistent original-teacher workers. Normalization batches
reuse the same pinned teacher and unchanged normalize/FK implementation; no change
in targets/statistics policy. Full corpus execution remains pending until128 raw
caches are complete. The experiment runner trains all128 fixed members for1000 updates,
records complete state at250-step intervals, and performs one fixed final validation.
No retry, extra training or sample/class selection will be driven by retrieval scores.

Module review PASS; actual preparation/training/retrieval outcome will be reviewed in
post_review.md after completion. Initialization/model/data hashes and per-query results
must be inspected before recording the experiment as completed.
