# ASLC001 execution state

COMPLETED: actual pilot training and validation retrieval finished; training process exited0.

Seed20261007;64 classes;128 official train/gallery videos;64 signer-disjoint validation queries; no test evaluation. Kinetics MAE initialization and random SignRep heads;1000 BF16 updates. Both checkpoints evaluate192 videos and6973 stride2 windows.

Mean pool: Top1 1.5625% ->1.5625%; Top5 15.625% ->25%; MRR .0920812 ->.1262235; DCG .2570292 ->.2924841. Weighted: Top1 1.5625% ->1.5625%; Top5 7.8125% ->28.125%; MRR .0768977 ->.1448148; DCG .2437452 ->.3130694.

Independent NumPy64 arithmetic agrees with all ranks. All1000 losses/gradients finite; all327 model tensors changed; final complete scheduler state1000. Source1595 files unchanged. Activity supervision valid fraction0.6125%; no independent full training rerun or full benchmark claim.

Report: REPORT.md; structured results: RESULTS.json; numeric audit: VERIFICATION.json; execution review: post_review.md; source/artifact pins: audit.json. Large artifacts remain under runtime/ASLC001, including final.model.pt and train/last.pt.
