# ASLC001 post-execution review

Execution: PASS. Actual RGB/teacher corpus preparation completed for all128 training videos,11242 frames. Official train membership and full validation/test exclusions were checked. The pilot uses64 fixed official classes and64 signer-disjoint validation queries; no test evaluation. Labels select the fixed balanced pilot and define retrieval truth, but do not enter the pretraining loss.

Initialization: official Kinetics MAE with random SignRep heads. No YT-SL-trained SignRep release weights enter this experiment. Training completed1000 BF16 updates with unchanged SignRep objectives; all327 model state tensors changed, all losses and gradient norms were finite, and complete scheduler state reached1000. Resource-only calibration runs are separate from ASLC001 results.

Arithmetic: PASS. Independent NumPy float64 pooling/cosine/class ranking agrees on all64 ranks and metrics for both phases/modes. Every feature cache is bound to its actual RGB video, encoder checkpoint and implementation hash. Initial/final windows and membership match exactly:192 videos and6973 windows per phase. The1595 pinned upstream files still match their original hashes.

Outcome: Mean Top1 stayed1/64; Top5 increased10/64 to16/64. Weighted Top1 stayed1/64; Top5 increased5/64 to18/64. MRR/DCG increased. The pilot demonstrates executed training and descriptive retrieval improvement below rank1, with weak Top1. It is not a full2731-class benchmark, a multi-seed result, a convergence claim or reproduction of paper scores.

Limitations: two training videos and one validation query per class; short run and one seed; BF16 computation; original full-frame preprocessing; corpus bone statistics from128 training videos only. Activity valid-target coverage is0.6125%, making weighted pooling a secondary result. Missing targets remain masked. No extra updates or class/video reselection followed the metrics. Full independent training rerun remains absent; numeric verification is PASS while the Material Passport remains UNVERIFIED.

Evidence: REPORT.md, RESULTS.json, VERIFICATION.json, training_summary.json, query_comparison.csv, rank_changes.json, audit.json and preparation/training/evaluation logs. The original reconstruction FINAL_AUDIT remains historical and is not reused as the adaptation audit.
