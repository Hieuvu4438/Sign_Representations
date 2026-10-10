# SIGNREL temporal

Independent implementation of [proposal7](../../docs/proposal7/SIGNREL_IMPLEMENTATION_SPEC.md). The core is a three-stream Transformer with frozen original DINOv2 targets. A1 reconstructs masked hand/body states; A4 adds sparse signed cross-hand cosine supervision. Raw-video inference requires the pinned MediaPipe and DINO pipeline.

Engineering verification: 37 tests pass, all fourteen CLI commands have execution receipts, and full-checkpoint and dedicated-encoder inference reproduce saved 2304D embeddings exactly. See [RUNBOOK](RUNBOOK.md), [model card](MODEL_CARD.md), [method](docs/METHOD.md), and [test report](docs/TEST_REPORT.md).

The completed 1000-update, seed17 pilot is **negative for the primary retrieval hypothesis**. On a frozen custom ASL Citizen dev protocol (40 glosses, one gallery exemplar per gloss, 80 queries), macro R@1 is 13.75% for raw teacher/body A0, 10.00% for A1, and 8.75% for A4. A4−A1 is −1.25 percentage points, with paired gloss-bootstrap 95% CI [−3.75, 0.00]. This does not establish a positive improvement or statistically significant degradation. Lower relation prediction error has not transferred to better retrieval.

The separately frozen 400-update matrix is running: A1/A4 have seeds17/29/43; A2/A3/A6 each have seed17. Its timing-only allocation shares the original global two GPU-hour cap. Locked test is evaluated after all budget-permitted training finishes. [STATE](docs/STATE.md) and `artifacts/matrix_status.json` carry the actual completion status; planned rows are not evidence.

Results: [pilot table](results/pilot_results.csv), [pilot report](results/research_report.md). The small corpus, generic visual teacher, perception errors and unavailable verified natural minimal-pair annotations limit conclusions. This is frozen gloss retrieval, not sign translation or direct phonology evaluation. External pretrained methods have not been reproduced here.

Use the own `.venv-core` and `.venv-perception` from this directory; another method also uses the `signrel` namespace. Dataset videos, derived caches and QA images stay local under their terms. No existing method or shared dataset is modified.
