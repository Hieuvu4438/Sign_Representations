# Requirement-level setup audit

**Historical audit, not current job status:** the user retired this method on
2026-10-17. Runtime/checkpoints/evidence and the systemd service were removed;
only code, configs, tests and documentation remain.

## Material Passport

2026-10-16. Objective:create a separate combined-method folder, choose/review fusion, install persistent background full training with intermediate validation, continuation and a freeVRAM queue. **This audit closes implementation/launch setup, not the160epoch experiment or final performance evaluation.** Those continue under the installed service.

Authoritative implementation review:[REVIEW_LOG.md](REVIEW_LOG.md). Machine evidence:`runtime/verification.json`, `runtime/live_verification.json`; these can be regenerated with `scripts/verify_setup.py` and `scripts/verify_live.py`. No subagents, external-model review or external data transmission was used.

| User requirement / design invariant | Current-state evidence | Finding |
|---|---|---|
| New method folder, isolated from existing jobs | `method/plctx_pose_fusion/{fp,configs,scripts,tests}`;14upstream source hashes rechecked | Complete |
| Think through handshape+body/relations fusion | `DESIGN.md`, `SOURCES.md`, `fp/{graph,model,trainer}.py`; frozenPL, fresh body/arm GCN+relation streams, bounded per-group residual gates, explicit SSL/retention targets | Complete; no deletedPose weights are claimed |
| Preserve PL and gate0 reference | Strict originalPL weight equality, real original-extractor parity test, gate0 token equality, frozen-state/gradient assertions; D/N/face768 output | Complete |
| Correct raw masking / no hidden-copy input | Raw dependency/duplicate closure before features/GCN; perturbation tests in near/far; explicit hidden-endpoint geometry, no dropout-only targets | Complete within the declared shoulder/dominance side-information policy |
| Full data and training setup | `configs/full160.json`;preflight confirms40154train/10304val/32941test, source/checkpoint/data hashes;batch32,200640additionalsteps,seed42 | Complete setup; optimizer budget not yet finished |
| Intermediate/stage validation | `validation_due` tests0/2000/every5000/final; realCLIsmoke validated0/1/2/3; real installed training process is currently performing full initial validation | Complete mechanism and live entry; full validation metrics accumulate during the job |
| Continue after interruption | Full atomic state, consumed-step sampling; exactCPU/CUDA continuation; actualSIGTERM validation subprocess resume; latest-corrupt fallback and mismatch refusal | Complete and exercised |
| Background persistence | `plctx-pose-fusion-full.service` active/running/enabled;Linger=yes;actual supervisor/child commands verified using `/proc` and systemd | Complete; independent of the interactive session |
| Wait for VRAM, join if sufficient | Supervisor threshold10240MiB,15s polling;real isolated subprocess queue→launch/single-instance test; productionjob admitted alongside existing GPU workload | Complete; no other job stopped |
| Frozen scientific job version | `runtime/full160/{code,ops,config.json,lock.json}`;current hashes match selected checkpoint/source verification | Complete; ops-only venv-entrypoint repair documented separately |
| Validation interrupted in live startup | `runtime/review/venv_entrypoint_repair.json`;real lastcheckpointstep0,pendingvalidation0,partial receipts reused after graceful stop/resume | Complete recovery; no optimizer progress discarded |
| Full final evaluation chained | Supervisor dispatches `fp.final_eval` only aftertraincomplete;selectedvalidationcheckpoint;raw/white512 val/test and frozenLP;independent orchestration fixture passes | Complete setup; official test results do not exist until training completes |
| Review/testing/setup evidence |17passed0failed/errors/skips;compileall;full-dimensional realGPUbatch32smoke;2.22GiBpeak;realCLIvalidation;source/data integrity preflight | PASS; execution evidence, not lexical improvement proof |

The live phase at audit time is **initial validation,step0**, within the running `fp.train` pipeline. It first computes a frozenPL baseline over the full train gallery and validation queries, then advances optimizer updates automatically. The checkpoint already contains the trained-normalization buffers and complete optimizer/RNG/model state plus the pending-validation transaction. It is not waiting for an interactive confirmation, and the service is enabled for reboot continuation.

Outstanding experiment work is intentionally asynchronous:complete200640updates, scheduled full validation, model selection and official test/probe evaluation. Accuracy and wall-time estimates remain unproven until those phases run. The setup objective does not require completing a multi-day experiment inside this interactive turn.
