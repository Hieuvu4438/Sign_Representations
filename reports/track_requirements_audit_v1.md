# Track requirement audit — 01/10/2026

This audit preserves the experiment scope in sections 14–16 of the execution
plan. It separates implementation evidence from evaluation on verified gold.
The overall research objective remains active; method novelty is unestablished.

| ID | Requirement and inspected evidence | State and remaining evidence |
|---|---|---|
| A00 | Text-only prior; shorter-token prior implemented in `scripts/run_semantic_pairs.py`, with deterministic candidate permutations | External gold blocked. No measured candidate-prior accuracy |
| A01 | Frozen translator ranking; `features/semantic_pair_backend_smoke_attempt1/report.json` verifies strict loading, source-caption NLL and identical-text ties; `reports/semantic_pairs_readiness_v1/status.json` records real readiness rejection | Numerical integration only. Distinct canonical semantic pairs, reviewed media/caches and external evaluation remain missing |
| A02 | Sentence-level grammar probe; `runs/ncslgr_diagnostic_v1/summary.json`, `runs/fair_ncslgr_v1/summary.json`, `runs/ncslgr_author_pool_v1/summary.json` | Completed limited internal positive recorded-function diagnostics. One test signer, archive grouping and pretraining limitations remain; no inferred absence |
| A03 | Controlled face/body/hand branch ablation with aligned streams, missingness masks and matched heads | In progress: `configs/protocol_ncslgr_branch_ablation_v3.json` locks eight interventions on the exact175common cohort with24equal-budget heads. Camera comparisons remain confounded. V2 failed original-cache reproduction guard; v3 separately registers a freshly recomputed within-run all-stream baseline and unchanged1e-5repeat guard. Original-cache equivalence is unestablished. No ablation classification metric until full extraction |
| A04 | Temporal grammatical scope with original clock and declared interval metrics; `runs/ncslgr_intervals_v1/summary.json` and audits | Partial internal functional-core diagnostic only. Adjudicated full grammatical scope, multiple-event detection, verified absence and calibrated presence are missing |
| A05 | Validation-selected calibration; `runs/ncslgr_calibration_v1/summary.json` and `reports/ncslgr_calibration_audit.json` | Exploratory categorical temperature sensitivity completed. All nine choices hit upper grid bound; independent validation and boundary/presence calibration remain unverified |
| A06 | Conditional adaptation supported by mechanism and evaluated on independent held-out groups | Pending mechanism and independent confirmation. Native temporal controls are negative and below the train-only duration prior; no encoder adaptation or supported method gain |
| Q00 | Locked random/length prior evaluated on occurrence/absence gold | Blocked by B-G0/B-G1/B-G2. No real grounding evaluation |
| Q01 | Cosine sliding-window matching with validation-selected threshold | Blocked gold; no completed implementation/evaluation claim |
| Q02 | Subsequence-DTW; `src/signrepr/grounding.py`, `scripts/run_grounding.py`, `tests/test_grounding.py`, `reports/integrity_checks.json` | Numerical implementation and synthetic end-to-end integration verified. Real readiness attempt is `BLOCKED_GOLD_OR_MAPPING`; no real occurrence AP. Validation policy and IoU thresholds remain unset |
| Q03 | Drop-DTW with exact pinned paper/code and cost policy | Blocked gold; reference implementation not yet invoked/pinned for an experiment. Standard Q02 is not Drop-DTW |
| Q04 | Equal-capacity query-conditioned temporal head trained on gold train occurrences | Blocked gold. No head trained and no pseudo-boundaries used as evaluation gold |
| Q05 | Conditional adapter with background/drop handling and matched controls | Pending diagnostic mechanism and verified gold. No method implementation or novelty claim |

Q02 now implements macro and pooled AP with score-ranked one-to-one matching,
registered secondary IoU thresholds, recall, false positives/minute, matched
boundary error and whole-recording bootstrap retaining true negatives. Controls
for isolated-to-isolated retrieval, query swap, context-matched negatives and
short/long windows still require a verified population and locked protocols.

No test metrics have selected an architecture, layer or search threshold. The
author-imputed pooling sensitivity was locked after strict result files existed
and is exploratory. NCSLGR functional-core intervals are not automatically full
grammatical scope. Gold status strings in synthetic fixtures only exercise the
validation contract and are not expert reviews.

The A03 v1 import failure and v2 absolute-cache-reproduction failure remain
preserved in `evidence/ncslgr_branch_ablation_import_failure_v1/` and
`evidence/ncslgr_branch_ablation_reproduction_failure_v2/`. A read-only numerical
diagnosis on the failed training clip found exact repeatability within its own
run, but a maximum difference4.72e-5from the old cache. V3 does not widen that old
guard or claim that it passed: it changes the comparison baseline to freshly
computed all-stream features shared with its own interventions and records old
cache differences separately. This technical choice occurred before any A03
classification metrics and does not make the reused test cohort independent.
