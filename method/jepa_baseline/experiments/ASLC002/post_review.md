# ASLC002 post-execution review

Execution and numeric verification: PASS. All 5000 updates completed, with finite losses/gradients and changed backbone/projection/prior tensors. Complete training state reaches the recorded budget. No released SignRep or previous pilot training weights were used; fresh Kinetics MAE initialization is pinned.

Data: 300 real classes; 3000 official train/gallery videos; 1500 official validation queries. Original signer/source split separation and full held-out exclusions are checked. Features in both phases have identical membership and window starts, and bind to their actual video/checkpoint/implementation hashes.

Arithmetic: independently implemented NumPy ranking and metrics match all recorded query ranks, top-five labels and nearest videos from separately replayed Torch float32 pooling/cosine scores. NumPy float64 pooling/cosine is a sensitivity audit; any differing ranks and metric values are explicitly recorded in the verification reports and certified against the measured score-rounding difference. Float32 pooling/cosine replay shares the original arithmetic backend and is not an independent implementation of those operations. Per-class scores and before/after query rankings are archived. These checks do not imply an independent full training rerun.

Limitations: one seed; reduced corpus, batch4 and 5000-step budget; BF16; corpus bone medians derived from selected train only; original full-frame inference. Activity valid-target coverage is 0.1900%. Step0 activity head is random. Any weighted score must be read with that coverage. Results are conditional on the fixed dictionary and cannot be compared directly with ASLC001 or full paper scores. No extra training or sample changes followed the metrics.
