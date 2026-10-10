# ASLC001 pre-implementation review

This is authorized ASL-Citizen self-supervised training followed by dictionary
retrieval. Start from official Kinetics MAE, never the YT-SL-trained SignRep release.
Labels may define fixed sampling/evaluation membership but do not enter pretraining
losses. Pilot versus full-data budget is being clarified before membership is frozen.

Generalize source identity and corpus audit explicitly rather than invent YouTube IDs.
Keep the original WLASL exclusion path and paper recipe enforcement. Local adaptation
must pin official CSVs, require pretraining source_split=train, reject held-out source
IDs/paths/content overlap, retain original signer-disjoint splits and compute bone
medians only from training members. Gallery always consists of train videos; queries
are validation. Test data is reserved for a final evaluation after selection.

Use original MediaPipe extraction and pinned 3-D teacher; existing local133-point
keypoint files use a different schema and cannot be substituted silently. Every
selected video needs a detected contiguous16-frame window. Log unavailable videos
and pose failures rather than silently filtering by label/performance. Fixed pilot
classes/membership are selected before any retrieval score is observed.

GPU is shared and currently has roughly3GB free; do not terminate other jobs.
Benchmark a small batch before choosing execution settings. BF16 AMP on Ada, if used,
is an explicit precision deviation, requires complete checkpoint-contract recording
and finite-gradient tests. All mathematical objectives/weights remain unchanged.

Report Top1/Top5, macro Top1/Top5, MRR and DCG, gallery/query/class counts, initialization,
steps, sampling, pooling and source/model hashes. Compare to step0 on exactly the same
retrieval membership; retain activity-weighted and unweighted results. Pilot metrics
are conditional on the pilot dictionary and never called full ASL-Citizen benchmark.

No unrelated files/jobs/raw datasets are changed. Training/process/timeout monitoring,
execution logs and checkpoint resume are required. User authorization to implement
and run this experiment overrides the skill's command-only/confirmation defaults.
