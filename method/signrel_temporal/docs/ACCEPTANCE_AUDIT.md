# Implementation and evidence audit

This audit maps the proposal7 requirements to executed evidence. Implementation acceptance is distinct from support for a research hypothesis. Current matrix status is authoritative in `artifacts/matrix_status.json`; this document will be finalized after the bounded matrix.

| Requirement | Implementation and executed evidence | Status |
|---|---|---|
| Isolated package and environment | Own folder, editable distribution and two environment locks; `artifacts/environment.json`, `doctor.log` | Verified |
| Required source provenance | Actual original DINO ViT-S/14 source commit, checkpoint and MediaPipe hashes; strict frozen [2,384] forward; `artifacts/sources.lock.json` | VERIFIED_RUNTIME |
| Strict configuration | Inheritance resolved; unknown keys, dimensions, invalid lag/mask/loss settings, DDP and unapproved full scale fail; `tests/test_data_config.py` | Verified |
| Label-free SSL | Separate How2Sign train/dev manifests and numerical caches; recursive forbidden semantic fields; evaluation labels joined only after export | Verified |
| Split and duplication audit | Actual video-byte hashes, recording groups, known query/gallery signers; `artifacts/split_audit.json` | Verified; SSL identities unknown |
| Real decode and cache contracts | Actual VFR PyAV fixture, physical nearest-PTS25Hz mapping, tolerance, source-frame dedup, validity and no hand interpolation; numerical/provenance cache audits | Verified |
| Anatomical hands and crop QA | Pose-wrist association, ambiguous-assignment exclusion, aspect-aware pixel distance; 20 real train previews and five eval previews; `artifacts/cache_qa/review.md` | Implemented and reviewed; no expert accuracy estimate |
| Fixed teacher and target suitability | Frozen original DINO384D normalized targets; nonconstant lag targets on20 real clips; `artifacts/teacher_suitability.json` | Verified; visual proxy only |
| Window/mask/g contracts | Recording-uniform deterministic draws; natural missing vs SSL masks; explicit correspondence; unit normalization and strict cache bindings | Verified |
| Joint encoder and inference | Four256D Transformer blocks, anatomical streams, physical time; overlapping canonical-frame merge, complete tail and physical whole/half pooling | Verified |
| State/relation losses | Independent NumPy oracle, fractional weighted denominator, stream-balanced sum-L2, coordinate masking, detached targets, empty differentiable loss, signed sparse edges/no wrap/no duplicate-source nonzero edges | Verified |
| Leakage and gradients | Clean hidden feature mutation leaves encoder output unchanged; relation gradients reach student; NaNs rejected; `tests/test_losses.py`, `test_encoder.py`, `test_pairs_mapping.py` | Verified |
| Optimizer and resume | Actual BF16 GPU AdamW/accumulation/clip/cosine runs; exact CPU dropout resume; real completed checkpoint no-op; source/config/cache/RNG bindings | Verified |
| Real target fitting | Two fixed real windows,100 CPU updates, decreasing state/body/relation errors; `artifacts/overfit_fixed_windows.json` | Debug verification only |
| Controlled A1/A4 pilot | Both1000 updates, same initialization, same4000 microbatch window/mask plans; source archives and checkpoints | Complete, one exploratory seed |
| Real retrieval and statistics | Saved vectors and separate indices; gloss-level max-exemplar ranking;80 dev queries/40 glosses, overlap/coverage audit, paired gloss bootstrap | Complete |
| Inference artifact | `artifacts/inference_api_verification.json`: full and dedicated encoder embeddings equal saved export, max absolute error0 | Verified |
| Minimal-pair evaluation | Lexical-pair clustered bootstrap and scorer tested; toy CLI exercised; `results/minimal_pairs_real.json` | Real BLOCKED_EXTERNAL; no numeric score |
| Shortcut/collapse checks | Train-only constant/lag means and nearest-visible predictor; held-out SSL dev errors, norms/cosines/effective rank; `artifacts/diagnostics_a1.json`, `diagnostics_a4.json` | Executed; spectrum limited to3 videos |
| Primary three-seed and mechanism matrix | Fixed400 updates; A1/A4 seeds17/29/43, A2/A3/A6 seed17; timing-only allocation and pretest freezes | Running; not yet confirmatory evidence |
| Global compute accounting | Downloads, actual extraction, profiles and training charged cumulatively; `artifacts/budget_ledger.json` | Active, unchanged caps |
| Source/data terms and novelty scope | `third_party/NOTICE.md`, hashed local ASL Citizen terms, `docs/RELATED_WORK.md` bounded primary-source review | Disclosed; no exhaustive/first claim |

All fourteen commands have real or explicitly synthetic execution receipts: doctor, resolve-sources, import-data, audit-splits, extract-cache, audit-cache, make-synthetic, profile, train, export-embeddings, eval-retrieval, eval-minimal-pairs, aggregate, build-report. Synthetic values are excluded from scientific tables. Training dry-run creates no placeholder checkpoint; unavailable real annotations do not pass evaluation.

Optional A5/A7, probes, SignRep/SHuBERT pretrained adapters and full-scale training are NOT_RUN. SignSeek/SignDino assets remain UNVERIFIED_ASSET. Required teacher/perception are verified, so those optional omissions do not block the main A1/A4 implementation. The folder does not contain a reproduced upstream pretrained baseline.

The pilot does not support H1. It neither proves the idea can never work nor justifies a method-improvement paper. A positive confirmatory point estimate would still need uncertainty, seed consistency, practical utility against A0 and broader evaluation before such a claim.
