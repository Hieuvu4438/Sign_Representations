# ASLC001 — ASL-Citizen pilot retrieval

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-07
- Verification Status: UNVERIFIED
- Execution Status: completed; single seed; no independent full pilot rerun
- Version Label: ASLC001_v1

## Protocol

64 fixed classes;128 official train videos;64 official validation queries. Original signer/source splits are disjoint. Test is not evaluated. Pilot labels are real official Gloss labels, not synthetic.
Kinetics MAE initialization with randomly initialized SignRep projection/prior/activity heads; no public YT-SL-trained SignRep weights. Actual SignRep losses and discriminator train for1000 steps,4 views/2 different videos per batch,80% mask,100 warmup steps,LR1e-4. BF16 AMP is a recorded precision deviation. Bone medians use only the128 pilot training videos.
Retrieval uses every complete16-frame window, stride2, full RGB frame, original inference preprocessing, BF16 compute/float32 cache; nearest individual training video by cosine. Top5/DCG/MRR rank distinct classes by their best gallery video. L2 follows video pooling. Activity-weighted pooling preserves raw max(LH,RH).
Step0 and step1000 use identical gallery/query membership. Step0 weighted results use an untrained activity head; unweighted results provide the simpler comparison. No test-based selection or metric-driven class sampling occurs.

## Results

| Pooling | Step | Top1 | Top5 | MRR | DCG |
|---|---:|---:|---:|---:|---:|
| unweighted | 0 | 1/64 (1.56%) | 15.62% | 0.0921 | 0.2570 |
| unweighted | 1000 | 1/64 (1.56%) | 25.00% | 0.1262 | 0.2925 |
| activity_weighted | 0 | 1/64 (1.56%) | 7.81% | 0.0769 | 0.2437 |
| activity_weighted | 1000 | 1/64 (1.56%) | 28.12% | 0.1448 | 0.3131 |

Macro Top1/Top5 equal instance Top1/Top5 here because there is exactly one validation query per class. Results are conditional on a64-class/128-video pilot dictionary, not the full2731-class ASL-Citizen benchmark. A single-seed short run does not establish superiority or convergence.

## Artifacts

- `run_config.json`, `inventory.json`, official CSV/source/model pins and run contract.
- `corpus/pretrain.jsonl.audit.json`, corpus bone statistics and original teacher caches.
- `train/metrics.jsonl`, complete `train/last.pt`, lightweight `final.model.pt`.
- `retrieval/{step0,step1000}/metrics.json` and per-query rankings/predictions.
- Preparation/training/evaluation logs; process status and resource calibration records.

## Observed supervision coverage

Fraction of target elements observed across actual sampled training batches; repeated frames/views count each time. Missing targets remain masked.

| Prior | Valid fraction |
|---|---:|
| body_pose | 47.42% |
| right_pose | 44.01% |
| left_pose | 35.19% |
| full_body_angles | 100.00% |
| full_right_angles | 44.01% |
| full_left_angles | 35.19% |
| right_dist | 44.01% |
| left_dist | 35.19% |
| body_dist | 27.24% |
| activity | 0.61% |

## Execution and arithmetic review

The process exited successfully after1000 updates and final retrieval. Independent NumPy float64 recomputation agrees with every query rank and all four primary metrics for both pooling modes and both checkpoints. All1000 losses/gradient norms are finite; all327 encoder/head state tensors changed. The final complete checkpoint contains optimizer/scheduler/RNG/sampler state at step1000. These execution checks do not constitute an independent full training rerun; the Material Passport status above remains UNVERIFIED for that reason.

Mean-pooling Top5 rose from10/64 to16/64, while Top1 stayed1/64. Weighted Top5 rose from5/64 to18/64, while Top1 stayed1/64. MRR and DCG increased in both modes. This is a descriptive result on the fixed pilot, with weak Top1 performance; it does not establish statistical significance or paper-level reproduction. Only0.6125% of activity target elements were valid across sampled batches, so the activity head was weakly supervised. No extra updates or sample changes were made after observing metrics.

Reviewable summaries are archived in `experiments/ASLC001/`. Large checkpoints, target caches, feature caches and per-query prediction JSONs remain in `runtime/ASLC001/`. `query_comparison.csv` records before/after ranks for all64 queries in both modes.
