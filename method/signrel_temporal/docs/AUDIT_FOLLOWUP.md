# Remaining audit actions

The scientific training source stays frozen until the matrix terminates. These are engineering audit items, not a change to the retrieval hypothesis or a request for more experiments.

1. Reserve test extraction budget at each training boundary using actual recent timings. The original supervisor was paused while its active training child continues; `artifacts/matrix_continue.py` resumes from existing state and preserves all frozen configs/locks. Missing rows remain explicit.
2. After the matrix, enforce exact training preprocessing fingerprints in training validation, CLI export and the cached public inference API. Existing extraction calculates the correct fingerprint and refuses incompatible reuse; some consumers currently check teacher hashes without checking all preprocessing fields. Verify unchanged vectors for compatible existing caches and explicit rejection of changed crop/teacher fingerprints. Preserve frozen experiment archives as the authoritative source for those runs.
3. Ensure dedicated inference uses the encoder without evaluating discarded decoders; keep decoder diagnostics explicit for full checkpoints. Verify full vs dedicated vs saved embeddings and unchanged context computations.
4. Measure CPU cached export and saved-vector retrieval costs separately from raw perception/teacher extraction. Record actual pair counts, training costs, cache coverage and uncertainty.
5. Finalize acceptance receipts, README, STATE, model card, reports, reproducibility manifest and research recommendation from actual matrix results. Do not call incomplete mechanism controls complete or unavailable minimal-pair scores zero.
