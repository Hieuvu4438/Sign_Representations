# Engineering verification

Current pytest receipt: artifacts/pytest_current.log (37 passing tests at this revision). Independent checks cover fractional quality weighted means and gradients, sum-L2 hand reduction/stream balancing, coordinate validity, NaN rejection, relation algebra against a NumPy oracle, empty differentiable loss, canonical sparse signed edges/boundaries/same-source duplicates, shuffled cross-window donors, explicit nonidentity g correspondence, clean masked-feature leakage, independent layer initialization, deterministic masks/windows, strict configs/manifests/cache hashes, anatomical crossed-hand association, pixel aspect, recording/signer/content split audits, real VFR PyAV decoding, unique-gloss max-exemplar ranks, actual encoder padding-invariant pooling, gloss bootstrap, minimal-pair lexical-cluster bootstrap, phase receipt invalidation and exact dropout checkpoint resume.

Synthetic CLI integration: artifacts/smoke_current_a1.log, smoke_current_a4.log, export_toy_*.log, eval_toy_*.log, aggregate_smoke.log, report_smoke.log, minimal_pairs_toy.log. A1/A4 each20 CPU updates; toy scoring is only software verification. Earlier logs are preserved; the initial VFR fixture codec failure was corrected and the current suite passes.

Real source/runtime: artifacts/sources.lock.json, doctor.log, resolve_sources.log. DINO outputs [2,384], repeat forward error0, frozen no gradients; pose/hand blank-frame VIDEO calls completed. Source files/weights have actual hashes, not unresolved placeholders.

Real data/cache: artifacts/split_audit.json, cache_audit.json, teacher_suitability.json, cache_qa/review.md. Training CPU dry-run: artifacts/real_cpu_dry_run.log, output shape [8,64,3,256]. Full-cache extraction and GPU training results remain distinct from engineering tests and will be recorded in the research report.

The current test receipt is authoritative. No upstream baseline or missing natural annotation is marked reproduced merely because its API exists.

## Staged consumer/inference audit

`artifacts/pytest_staged.log`:42 tests pass in an isolated copy, including five additional checks for incompatible preprocessing/teacher provenance, cache-manifest hashes, mixed preprocessing, decoder-free inference and portable bindings. `artifacts/staged_inference_verification.json`:all47 real training caches accepted with their exact existing preprocessing fingerprint; full and dedicated encoder vectors equal the immutable pilot export with max absolute error0. A portable artifact works without a machine-specific source-lock path. An actual real CPU dry-run returns [8,64,3,256] without a checkpoint.

These changes are staged only while the scientific matrix runs. `artifacts/promote_verified_validation.py` refuses to modify the main source until terminal locked-test evaluation. Current scientific source SHA remains163204175d1f1c1a1436dcffcccd9819992b0f4cb0967a6b8d7440772de0b2af; historical run archives preserve it. Final package verification will rerun these checks after promotion and compare every scored export.
