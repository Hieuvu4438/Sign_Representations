# ASLC002 — fresh ASL-Citizen baseline,300 classes

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-07
- Verification Status: UNVERIFIED
- Protocol Status: frozen before retrieval metrics
- Version Label: ASLC002_v1

User explicitly requested training a reproducible baseline on the proposed300-class scope and actual retrieval statistics after completion. No public SignRep weights or previous pilot training weights are used. Fresh Kinetics MAE initialization follows the paper; SignRep heads start random. “From scratch” here means fresh sign-domain pretraining, not a randomly initialized backbone. This interpretation was stated before execution.

Seed20261008;300 eligible classes;10 official train videos/class;5 official validation queries/class. Prefer distinct training signers.3000 pretrain/gallery videos and1500 queries; no test evaluation. Gloss labels select the balanced subset and define retrieval truth, but do not enter the pretraining objective. Membership/CSV hashes and official signer/source split disjointness are recorded.

Original MediaPipe heavy+pinned3-D teacher processes original RGB video. No existing COCO keypoints substituted. Preserve every decoded frame and missingness masks. Corpus bone medians use only these3000 training videos. Pose-invalid training members, if encountered, must be explicitly recorded and resolved by deterministic same-class reserves before any retrieval metrics are observed; no score-driven replacements.

SignRep reconstruction,variance,covariance,style discriminator and adversarial objectives unchanged.5000 updates,500 warmup,LR1e-4,80% masking,4 views from2 distinct videos. BF16 precision and batch/data/step reductions are documented deviations from the500k/batch20/YT-SL paper recipe. Each run records optimizer/scheduler/EMA/RNG/sampler state every500 steps. No extended budget after seeing scores.

Evaluate freshly initialized step0 and final step5000 on identical gallery/query membership. Full RGB frames,original inference preprocessing,every16-frame window,stride2,retrieval batch4,BF16 compute/float32 caches. Report mean and raw max-hand-activity weighted pooling;L2 after pooling;cosine closest training video. Top5/MRR/DCG rank distinct classes by best gallery video. Primary MRR;also Top1,Top5,DCG,macro metrics,rank distribution and per-query predictions. Independent NumPy arithmetic checks all query ranks.

Step0 activity head is random; actual pose/activity coverage across sampled training batches is reported. This is a single-seed reduced-data/compute baseline,not a full paper-score reproduction. The64-class ASLC001 pilot is historical and cannot be directly compared because the dictionary/query scope differs.

GPU is shared. One experiment GPU process at a time,allocator fraction.042 (~2.06GiB),4 CPU teacher workers. Do not modify or kill other jobs. Dataset remains read-only. Stage hard timeout8hours; checkpointed stages can resume if interrupted. Continue until training,evaluation,verification and report are complete.
