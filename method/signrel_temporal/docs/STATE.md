# SIGNREL implementation state

phase: 7
status: running_frozen_controlled_matrix
last_successful_command: paired real A1/A4 1000 updates; dev frozen retrieval; strict aggregate/paired bootstrap; fixed held-out shortcut diagnostics; completed-checkpoint resume no-op; dedicated encoder inference equals saved export
artifact_paths: results/pilot_results.csv, results/pilot_results.json, results/research_report.md, artifacts/diagnostics_a1.json, diagnostics_a4.json, inference_api_verification.json, matrix_plan.json, confirmatory_freeze.json, matrix_status.json
verification_evidence:37 pytest tests;47 SSL train+3 SSL dev+120 gallery/dev caches audited; pilot init and4000 microbatch window/mask plans exactly match; A0/A1/A4 dev macro R@1=.1375/.1000/.0875; A4−A1=−.0125 (gloss bootstrap95% CI[−.0375,0]); A4 improves relation prediction error but not dev retrieval
blocked_by: curated natural minimal-pair annotations BLOCKED_EXTERNAL; external pretrained adapters NOT_RUN; SSL signer and lexical identities unknown
next_action: finish budget-permitted frozen400-update rows, then locked-test evaluation; final audit includes preprocessing-fingerprint enforcement and decoder-free inference, measured compute/coverage and reproducibility manifest
remaining_budget: authoritative artifacts/budget_ledger.json; new timing-only allocation uses existing global2 GPU-hour cap, not a reset; predicted training3770 seconds including20% reserve plus442-second test extraction reserve,4396 seconds available at allocation

Phase7 horizon400 was selected by a preregistered timing-only rule, independently of retrieval scores; this is a separate family from the1000-update pilot. Test configs/protocol/training selection are frozen before the first test score. Pilot is negative evidence for current retrieval utility, not proof against all coordination-learning ideas. Three-seed results are still incomplete while matrix runs.

Actual shared-GPU slowdown exceeded the pilot timing reserve. Three A1 rows are complete. The original supervisor was paused, its active training child allowed to finish, and the supervisor replaced by artifacts/matrix_continue.py without retraining completed rows or regenerating configs/locks. Each subsequent row must fit recent measured timings plus the original locked-test extraction reserve; otherwise it is incomplete_budget. All caps remain unchanged. artifacts/matrix_budget_guard_intervention.json and matrix_runtime_budget_decisions.json trace this resource-only decision.

Additional actual CPU evidence: artifacts/inference_profile.json reproduces all120 saved A4 dev embeddings with maximum absolute error0; cached validation+encoder/pooling mean .01224 seconds/video, ranking80 queries ×40 gallery mean .00721 seconds over5 repetitions. Raw perception/teacher and file writing are excluded. docs/AUDIT_FOLLOWUP.md lists remaining engineering verification; docs/RESEARCH_DECISION.md separates observations, interpretations and continuation criteria.

Latest engineering progress: isolated artifacts/engineering_validation source passes42 tests (artifacts/pytest_staged.log). It checks exact real preprocessing fingerprints and manifest hashes, rejects mixed/incompatible cache provenance, carries portable encoder fingerprints and skips decoders during inference. Staged verification accepts all47 real train caches and reproduces full/legacy/portable pilot inference exactly. Main scientific source is unchanged. Promotion is gated on terminal locked-test evaluation; artifacts/verify_final_inference.py then checks every scored learned export before packaging.

Latest runtime evidence: A1 seeds17/29/43 and A4 seeds17/29 are complete400 updates; A4 seed43 was allowed by the actual ledger and is running. A2/A3/A6 remain planned until their timing-only gates. No locked-test score has been read yet. Do not predeclare missing rows from an estimate; use actual matrix_status.json and runtime budget decisions.
