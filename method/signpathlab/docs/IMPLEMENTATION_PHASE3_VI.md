# Implementation và code review — vòng 3, 12/10/2026

Vòng này đã nối APIs thành pipeline CLI thật và chạy đủ **15 trials E00–E04 × 3 seeds**.
Toàn bộ proposal vẫn đang thực hiện; chưa đóng các yêu cầu embed/export/deployment,
continuous evaluation và full acceptance audit.

## Code đã triển khai

| Nhóm | Hành vi và verification |
| --- | --- |
| `preprocess` | Manifest-driven PTS decode, MediaPipe IMAGE, raw source/interval/model hashes, two-process shards, verified reuse; đã chạy 32 train/16 val clips |
| `cache build` | Strict DINO load, ROI/pose/visual cache, source-indexed resampling, locked atomic index, sample/interval/metadata checksums, failure reports; đã chạy cả train/val |
| `targets fit` | Train-only geometry stats/matrices tách thành đúng NPZ paths; all center phases; seven-stream K256 MiniBatchKMeans, valid/deduplicated source pool, source cap, centroids/occupancy/version/provenance |
| `verify pipeline` | Exact source/hash/signer train-val audit; temporal perceptual candidates; recorded review with fingerprint identity; closed masks/fully-hidden geometry; pilot pass, no near-duplicate candidates |
| `train` | Real cache provider; threaded verified readers with main-owned RNG; global training engine; frozen linear validation selection; best/last/retained step checkpoints; failure memos/resume identities |
| `downstream fit` | Fixed LR trials, train-only labels, validation PC Top1 selection with validation-loss tie-break; linear/attentive/temporal fine-tune; full-clip tiled outputs; head/encoder identity and protocol lock |
| `evaluate` | Locked head, val/test explicit split, source/signer/hash audit, classification and source-excluded cosine retrieval, predictions; pilot used val only |
| `profile` | Actual cached loading + teacher/student/predictor/backward/update timings and peak CUDA memory; excludes decode/detector/DINO, as explicitly labeled |
| `experiments run/summarize` | Actual trial statuses, per-seed CSV/JSON, separate incompatible comparison signatures, matched E03/E04 successful steps and mask digests; no fabricated missing metrics |

**103 tests passed, 0 skipped** in the recorded environment, including earlier CUDA AMP,
Gloo DDP/resume tests and new corpus/codebook/probe/tiling/fingerprint/review/LRU tests.
Evidence: `evidence/phase3_tests.json`, `phase3_code_snapshot.json`.

## Code review findings and fixes

Review is a self-review by the coding agent, not an independent agent review.

1. Cache indexing checks sample ID, source SHA, start/end interval, split, metadata SHA,
   feature identity and root containment. A cached clip cannot be selected merely because
   its directory name matches. Fixture roots are explicitly marked and rejected by real CLI.
2. Cache selection FPS is recorded and checked. Changing FPS/extraction requires a new
   compatible cache rather than silently reusing a prior sample schedule.
3. MediaPipe close originally ran in an atexit callback, after Python's callback executor
   had stopped. Data extraction succeeded, but shutdown printed an error. Shard tasks now
   close detectors explicitly before returning. Real two-shard verified-reuse run completed
   without that shutdown exception.
4. Train-only target fitting keeps real source IDs/checksums and validates output schema.
   K-means does not reduce K silently when valid pool<K; requested K256 must fit actual data.
   Missing observations are -100 and never manufactured as cluster labels or static paths.
5. Full-clip inference covers the tail; overlapping window outputs align by original sample
   slots and timestamps, then average before clip pooling. It never evaluates only a prefix.
6. Linear/attentive probes freeze the encoder; fine-tune actually changes online weights.
   Tests verify both policies. Frozen foundation DINO is still frozen in temporal fine-tune.
7. Mask/frame budgets are recorded. Primary E03/E04 pairs have identical successful steps
   and mask schedule digests for each seed; summaries do not pool incompatible model/data/head budgets.
8. Profiling found repeated support decoding dominated the update. A 64-sample bounded
   LRU now reuses **validated** immutable arrays/supports, with inode/size/mtime/ctime checks
   every lookup. Changed files trigger content/hash validation again. Tests verify tampering
   rejection and isolate caller metadata edits. No weights/data are repaired or synthesized.
9. The initial pre-LRU run was deliberately interrupted at E01 step93 after saving last.pt.
   Its results/checkpoint are retained in `artifacts/pilot8_pre_lru_diagnostic` and excluded
   from the final comparison. All 15 reported trials were run anew on one frozen implementation.
10. Subsequent input-validation review kept original proposal14.1 configs compatible by
    defaulting only new optional fields; unknown keys still fail. Retrieval no longer mutates
    caller float64 arrays. Missing train labels produce a useful error. Final test audit now
    includes the validation selection membership as well as train/test overlap.
11. Near-duplicate reviews must match the actual fingerprint pool and identify each pair
    as distinct with a reason/reviewer; stale, duplicate or absent decisions cannot unlock
    training. Actual duplicates require manifest repair and a fresh audit.

Frozen training source archive/hash: `evidence/phase3_pilot_source.json`. All 12 SSL
checkpoints share that recorded implementation. Current API validation fixes do not alter
the model/loss/data used by those trials; current E04 seed17 evaluation was rerun and all
16 predictions/logits match within1e−6. This distinction is recorded in the source snapshot.

## Pilot protocol and results

Protocol was written before examining model scores: selection seed137, 8 official ASL
Citizen classes, 4 train and 2 validation clips/class, official signer-disjoint membership,
source bytes hashed, no test CSV/videos accessed. The pool is **label-curated** by class and
availability; this is disclosed and is not a full-corpus or target-label-free selection claim.
SSL loss/sampling consumes no gloss/text/class labels; downstream fitting does.

Each SSL trial: D96, 2-layer encoder, 1-layer predictor, 100 successful updates, batch4,
accumulation1, seeds17/29/43. Same DINO/pose cache, heads, LR trials [.001,.003,.01], max40
head epochs/patience10. E03/E04 share requested32D output, windows, masks, queries and λ=.1.
Native descriptor rank/information content remains different, as disclosed in the proposal.

| Method | Validation PC Top1 mean ± seed SD (%) | Retrieval R@1 mean (%) |
| --- | ---: | ---: |
| E00 frozen visual | 25.00 ± 0.00 | 31.25 |
| E01 cluster control | 20.83 ± 9.55 | 12.50 |
| E02 raw JEPA | 27.08 ± 3.61 | 16.67 |
| E03 ordinary motion targets | 27.08 ± 3.61 | 16.67 |
| E04 logsignature targets | 27.08 ± 3.61 | 16.67 |

Seed SD is not a confidence interval. Validation has only16 clips and8 classes; these
scores do not establish general ASL Citizen performance, reproduction, SOTA or superiority.
E03/E04 difference is exactly0 PC Top1 in every seed. Their online weights do differ
(L2 differences about.151/.153/.151), so the tie is not a dummy/untrained geometry branch.

Evidence: [comparison CSV](evidence/phase3_comparison.csv), [full JSON](evidence/phase3_comparison.json),
`phase3_runs.json`, `phase3_pilot_plan.json`, `phase3_pipeline_checks.json`.

Decision: **no evidence yet for a signature advantage**. Keep E00 as the current retrieval
reference and E02 as the simpler SSL reference; retain E03/E04 as runnable controlled
ablations. Do not scale degree/lift complexity based on this pilot. Next useful experiment
needs stronger training/data/quality and static phonological checks, not a novelty claim.

## Profiling limits

For D96 cached E04, CUDA updates after warm-up were about.026s; prior batch preparation
was1.4–2.3s. The last5 batch preparations in the LRU profile averaged.512s, first5 averaged
1.454s. This is one small-corpus cache profile, not full-corpus throughput or end-to-end latency.
Peak CUDA allocation in the first profile was about47MiB, excluding detector/foundation extraction.

Original total preprocessing/GPU-hour accounting is incomplete: resumable stage reports
currently retain the latest view, and not every original timing is retained. Missing costs
must stay null/unknown, not be inferred from cached update time. A versioned cost ledger
and larger-corpus support decoding profile remain open.

## Remaining objective scope

- Raw-video `embed` and deployment `export` CLI, metadata/schema package and export parity.
- Anatomical cross-modal/stream mask strategies, kept distinct from verified strict temporal
  hiding; the current training provider explicitly rejects those unimplemented regimes.
- Full artifact schema, quality buckets, uncertainty/phonology protocols and collapse monitors
  integrated into the generic runner. Basic heads/bootstrap APIs do not prove those full protocols.
- Continuous real runs and optional adaptation/backbone/geometry-input variants, or explicit
  unavailable-backend contracts where allowed; no silent fallback.
- Full environment/reproduction audit, versioned preprocessing cost ledger and final
  requirement-by-requirement acceptance. The official full corpus has not been imported/audited
  as a research benchmark by this pilot.

The goal remains active. This report is the implementation/code-review handoff for phase3.
