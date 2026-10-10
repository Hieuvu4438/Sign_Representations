# ASLC002 current results and verifier recovery

Training has not started. The host GPU pipeline completed step0 retrieval then failed its original exact NumPy64-versus-Torch32 rank equality check. No final checkpoint, training history or post-training retrieval result exists.

Step0 scope: 300 classes, 3000 official train/gallery videos, 1500 signer-disjoint official validation queries, 169818 inference windows. Fresh Kinetics MAE backbone and random SignRep heads, no released SignRep checkpoint.

| Pooling | Top1 | Top5 | MRR | DCG |
|---|---:|---:|---:|---:|
| Mean, step0 | 1.6000% (24/1500) | 6.0000% (90/1500) | 0.048479 | 0.191028 |
| Activity, step0 | 1.4667% (22/1500) | 5.7333% (86/1500) | 0.047280 | 0.189825 |

These are BEFORE-training metrics, not trained SignRep baseline scores.

Verifier recovery: pooling/cosine are separately replayed with Torch float32; ranking and metrics are independently implemented in NumPy. All recorded query ranks, top5 labels and nearest-video IDs match exactly. The float64 audit records 10 mean-pooling and 5 activity-pooling rank differences, each within a rank interval certified by the measured cosine rounding difference. Float64 mean Top5 is 5.9333% versus recorded float32 6.0000% (one query). This is numeric sensitivity, not a fabricated exact-float64 agreement. Original failure logs and initial_rank_diagnostic.json remain archived.

INITIAL_VERIFICATION.json now PASS. Historical ASLC001 read-only regression and rejection of an intentionally corrupted per-query rank pass. No model, membership, initialization, training-budget or core implementation changes. Only verification and its review description changed.

Resume the existing pipeline on the GPU-accessible host terminal using the same pipeline command. Completed preparation, step0 inference and initial verification are skipped. No training results should be reported until train history and final retrieval artifacts actually exist.
