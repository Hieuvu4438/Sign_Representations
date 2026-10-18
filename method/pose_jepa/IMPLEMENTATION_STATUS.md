# Implementation status — 2026-10-14

**Implementation baseline and adapters: complete.** This does not claim full-scale
benchmark quality. Read [the completion report](docs/COMPLETION_REPORT_VI.md) and
[model card](docs/model_card.md) for experimental scope and limits.

| Milestone | Implemented/tested evidence | Trained/evaluated scope |
|---|---|---|
| M0 | `pyproject.toml`, validated configs, dependency lock, `docs/codebase_review.md`, provenance, source lock (13 pinned reference files) | Scratch encoder; external RGB/Uni-Sign/MT5-base weights not loaded |
| M1 | Cache schema/converter/tracker, streaming native PTS, explicit score adapter/raw sidecars, split/dedup/coverage, fixed reference and sampling tests | 32 actual ASL-Citizen clips audited; overlay inspection on 20 videos |
| M2 | Part/frame-independent GCN, shared hands, ordered projection, packet/time/part encoding, full encoder | Full encoder width 256, 8 layers; 10,255,872 parameters |
| M3 | Constrained M/C, packed/scattered predictor, supported CE/geometry/motion, no-grad teacher | Hidden support, negative control, padding, missing/root/timing, physical-unit tests |
| M4 | AdamW/AMP/global denominator/center, deduplicated EMA, atomic save/RNG/cursor, worker prefetch resume | CPU/GPU fixture and real 20-update smoke; actual 2-process Gloo and single-GPU AMP tests |
| M5 | Same-backbone controls, diagnostics, validation checkpoint selection, shuffled targets | Five completed 5,000-update pilots on 16 real train clips; fixed-update comparison, tiny 8-clip validation |
| M6 | Frozen/full ISLR, exact dictionary retrieval, namespaced P*K/global SupCon, warmup/joint/replay, controls | Real overfit (16/16 train correct); geometry/hybrid frozen probes; 3 metric controls at 20 updates each |
| M7 | Full calendar/time stitch/cache, CTC repeated-label checks/decoder, frozen/fine-tuned CSLR with cache disabled for encoder training; mT5 bridge/label masking/full utterance | Isolated-gloss CTC adapter sanity; actual tiny mT5 workflow and differentiable long-sequence tests |
| M8 | Matched encoder/head bundles, hashes/objective metadata, CPU parity/latency, runnable example, reports/audit | Geometry selected by validation; hybrid and metric bundles retained; CPU output `[27,1024]` / raw `[1024]` verified |

**53 tests passed**, with warnings treated as errors, on the pinned environment.
Executed tests are recorded in `docs/evidence/tests_torch251.xml` and the final
JSON summary. Coverage includes meaningful mathematical/negative-control tests,
not output shapes alone. No TODO/NotImplemented production stub remains; the
`pass` in worker shutdown drains prefetched batches intentionally.

Runtime artifacts are excluded from Git: environments, poses, checkpoints,
feature caches, source snapshots and inference bundles. Every pilot terminated
successfully; `runtime/pilot_controls/results.json` records actual process exits.
Source snapshots are checksum-verified against run provenance, including versions
used before final validation/metadata fixes. Default scientific update parity was
checked bitwise on the full backbone against the original pilot source.

**Not trained/evaluated at full benchmark scale:** WLASL2000, full ASL-Citizen,
NMFs-CSL, full 1,000+10,000 Stage 2 budget, real continuous CSLR or real SLT. Required
audited continuous transcripts/translations, linguistically annotated minimal pairs
and calibrated/exact detector provenance are unavailable for these runs. Multi-GPU
NCCL is untested on the single-GPU host. These are experimental/data limitations;
they do not stand in for missing baseline implementation.

The legacy pose scores are unbounded SimCC maxima, explicitly converted to bounded
uncalibrated reliability with raw audit retention. Offscreen extrapolation and
occlusion remain detector limitations. Neither fixture nor tiny-subset metrics
establish SOTA, general transfer or superiority of the proposed hybrid objective.
