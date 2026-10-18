# PLctx–Pose gated body/relations fusion

**Stopped and cleaned on 2026-10-17 at the user's request.** Only source, configs,
tests and documentation remain. The entire `runtime/` (checkpoints, logs, feature
caches and smoke runs) and the systemd service have been removed, freeing about
4.32GB (4.02GiB). Shared PLctx weights and dataset cache were retained unchanged.
The old fusion run cannot resume; commands and runtime evidence paths below are
historical setup documentation, not an active service.

## Material Passport

Implementation and authorized full ASL-Citizen job setup,2026-10-16. The complete pretrained PLctxfull160 EMA encoder is preserved as a frozen reference; a new body/relations graph branch is learned and fused through bounded residual gates. Pose-JEPA's old weights were deleted, so this **does not reuse a pretrained Pose-JEPA30k teacher**. No fusion accuracy improvement is claimed before validation/full-test results exist.

Design and mandatory invariants:[DESIGN.md](DESIGN.md). Provenance:[UPSTREAM.json](UPSTREAM.json),[SOURCES.md](SOURCES.md). Review/evidence:[REVIEW_LOG.md](REVIEW_LOG.md).

## What is combined

- Frozen PLctx: all original7 channels and Transformer6×256/4heads, retaining handshape, placement and face context. Initialized exactly from `method/phonojepa/runtime/runs/PJ_PLctx_FULL_160ep/final.pt`, SHA256 `8f8a10687f2df869fd0d7f4703edfeb467c59977d7baaacb2d567c15789e41f9`.
- Trainable geometry: separate frame-local GCNs for core torso and both shoulder–elbow–wrist chains, plus explicit wrist/body/head/wrist relation tokens. Spatial widths64/64/128/256 with per-joint LayerNorm and ordered-node aggregation;8×256 temporal Transformer/8heads,128 graph tokens plus a null key. This adapts the local Pose-JEPA spatial design to the shared103-joint cache; it is not the old four-part Pose-JEPA model.
- Fusion: each PL token queries graph context. Normalized residual is scaled by three learned D/N/face gates bounded at.3. Gate0 reproduces original PL token outputs under the same input/reference; the training graph cannot update frozen PL parameters. Actual initial gate strength after warmup is about.0054, not.3.
- Loss: masked contextual EMA CE +.1 masked torso geometry +.1 masked relative geometry +1.0 visible-token PL retention. The predictor4×128 starts from compatible PL predictor/ctx-head weights. Body/relations targets come from clean teacher joints in the student's affine/reference, not deleted Pose embeddings. No gloss labels enter SSL losses.
- First2000 optimizer updates keep fusion closed while graph/predictor geometry learns. Gates ramp in over the next8000. These are separate warmup phases within a single run; all checkpoints retain the same optimizer and EMA stream.

The graph reads body units and estimator confidence, not cropped RGB or external3D weights. Shared shoulders and full-clip dominance remain explicitly allowed side information. Student raw masks use PL-style time/channel masking plus torso tubes, duplicate wrist/nose closure and exact interpolation support before GCN/features. Graph input keys are absent if any declared supporting raw entry is forbidden; context-drop dependencies are respected. Missing joints never carry an artificial zero-filled coordinate as an observation. Shoulder-unavailable normalization uses a fixed origin/scale instead of the old all-joint centroid fallback; parity with the old PL pipeline is conditional on the same preprocessing/reference, and that safety correction is declared.

The final readout stays768D: dominant hand pair, nondominant hand pair, face; mean within groups/windows then raw-window mean → videoL2. The graph contributes through the residual, not an extra unbalanced concatenated1024D block. The same extraction also saves `pl_gate0` as a within-run frozen-reference control. There is no test-set weight/readout optimization.

## Full run and validation

- Full official train:40.154 clips; batch32;160 epochs =200.640 updates; seed42; bf16; AdamW peakLR3e-4/wd.05; gradient clipping1; EMA.996→.9999.
- This is **additional fusion training after PLctx160 pretraining**, with a fresh graph branch. It is not a scratch-compute-matched comparison with PLctx16k or old Pose runs.
- Validation: full40.154 train gallery and10.304 official val queries, raw class-max cosine with pessimistic ties. Step0,2000,every5000 and final;64-sample windows/stride32. Both fusion and PL gate0 are reported. Best fusion checkpoint uses validation Rec@1; ties retain the earlier checkpoint, including the initial baseline if no gain occurs.
- Final evaluation: selected checkpoint only, after training completes. Official test32.941 queries/2731 glosses; full gallery40.154; dense stride2 for final extraction. Report raw DCG/Rec1/5/10/MRR and gallery-only pooled-feature whitening512. Frozen linear classifier uses train labels, selects on validation and scores test; this downstream supervised probe does not update either encoder. Both fusion and PL gate0 receive the same protocol.
- The main comparison target is raw full dictionary, not transductive LOSO. White512 here fits pooled gallery features, so it is separately identified from the old window-whitening ASLC300 evaluator. No ASLC300/probe result is claimed unless separately measured.

Configuration:[configs/full160.json](configs/full160.json). Official test is never encoded in the training loop; preflight only checks split membership and source presence/stat inventory. Training input arrays and checkpoint are byte-hashed; source keypoint inventory uses ID/size/mtime, not a claimed full-video byte audit.

## Persistent execution and status

The isolated job lives under `runtime/full160/` and uses the existing `.venv-signrep` environment. Large train-cache arrays are read-only memory maps inside the existing uncompressedNPZ, sharing OS pages between spawn workers; no second3.4GB array cache or RGB copy is made.

The systemd user service `plctx-pose-fusion-full.service` is enabled with Linger. Source/config are copied to immutable `code/`, `ops/`, `config.json` and hash-pinned in `lock.json`. Editing this method's working source does not alter an already installed run.

From the workspace root:

```bash
.venv-signrep/bin/python method/plctx_pose_fusion/scripts/job.py status
.venv-signrep/bin/python method/plctx_pose_fusion/scripts/job.py stop
.venv-signrep/bin/python method/plctx_pose_fusion/scripts/job.py resume
tail -f method/plctx_pose_fusion/runtime/full160/train/train.jsonl
```

The detailed training log is `runtime/full160/train/train.jsonl`; the process output is `runtime/full160/train.log`. `progress.json` shows consumed updates/validation/extraction; `supervisor.json` records the live child PID or resource queue. `systemctl --user status plctx-pose-fusion-full.service` is authoritative for service liveness; a JSON file by itself is not.

The supervisor joins concurrent work whenever at least**10.240MiB VRAM** is free, and does not wait for all other GPU jobs to exit. It never stops them. Less memory produces a15-second queue poll; CUDA OOM requeues after60seconds from the last committed checkpoint. Memory fraction cap.23 (~11GiB), serviceRAM high24GiB/max32GiB,CPU quota8cores/Nice5. Disk reserve8GiB prevents starting another phase when space is exhausted. Nonresource code/data errors stop with logs rather than retrying silently. A single-instance lock prevents duplicate supervisors.

Periodic checkpoints every100 updates use atomic fsync/replacement with `last.pt` and `previous.pt`; best validation and sparse stage0/2000/final checkpoints are retained. Full state includes graph/fusion/frozenPL, teacher, predictor, heads, optimizer, CE center, normalization, RNG and exact consumed-update data stream. A corrupt latest checkpoint can fall back to previous; changed scientific source/data/config cannot. SIGTERM/INT/HUP saves at update boundaries. Hard loss of power/SIGKILL can redo at most the updates since the last commit; no claim of zero lost compute.

Validation is a transaction: a pending marker is checkpointed first; an interrupted validation resumes before any next update. Per-clip extraction receipts match teacher content/step/source identity. Committed validation caches are removed to avoid keeping42 copies of full-gallery features; their metric/coverage records remain. Final extraction retains resumable per-clip receipts bound to the selected checkpoint; the linear probe resumes at epoch boundaries.

Manual setup for a new isolated job (do not duplicate the installed full run):

```bash
.venv-signrep/bin/python method/plctx_pose_fusion/scripts/job.py setup
.venv-signrep/bin/python method/plctx_pose_fusion/scripts/job.py install
```

## Verification

```bash
PYTHONPATH=method/plctx_pose_fusion .venv-signrep/bin/python -m pytest method/plctx_pose_fusion/tests -q
PYTHONPATH=method/plctx_pose_fusion .venv-signrep/bin/python method/plctx_pose_fusion/scripts/verify_setup.py
```

Scientific risk: explicit body geometry does not guarantee lexical gain; the model can improve body probes while hurting retrieval. Gates, a frozen baseline and validation selection make this measurable, not guaranteed. Old ASLC300 rank complementarity is exploratory evidence, not an expected improvement on full official test. One seed cannot establish run-to-run reliability.
