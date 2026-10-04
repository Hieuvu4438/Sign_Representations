# FEAT001 — bounded SignRep preprocessing and feature parity

- Registered 2026-10-03 before execution; H001/H002, P001_v1 prerequisite.
- Type: CPU data/software smoke; no classifier, labels, test inputs or HPO.
- Question: does a new batched adapter match the read-only upstream inference
  recipe in the existing `.venv-signrep` environment?
- Inputs: five train/dev videos fixed by DATA001_r1; same source hashes,
  checkpoint and model parameters as LOAD001. No new sample selection.
- Reference: upstream `Transformation.aug_video` and the single-window forward
  loop in `example_usage.py`; device changed to CPU, model in eval mode.
- Adapter: RGB, deterministic per-frame Resize224/ImageNet normalization;
  16 native frames, stride2, incomplete tails omitted; first/last valid windows
  per video, batch2, float32. No FPS resampling, padding or pooling.
- Gates: source hashes match; OpenCV decoded count equals fresh ffprobe PTS;
  finite strictly increasing PTS; exact window count/index coverage;
  preprocessing atol1e-6/rtol1e-5; `features`, `latent`, raw `cls_active`
  batch-versus-serial atol1e-5/rtol1e-4, expected shapes and finite outputs.
- Independent oracle: constant RGB normalization and hand-computed irregular
  PTS/window boundary fixtures. No claim of independent encoder implementation.
- Budget: one CPU thread, ≤900s, zero GPU-hours, no download/install;
  new outputs <10MiB. Two retries maximum, only with a diagnosed fix.
- Config: `method/frozen_probe/configs/FEAT001.json`; seed1701;
  run `FEAT001_20261003_P001v1_s1701`, exclusive output under runtime.
- Artifacts: config/code/upstream/environment/source hashes, command, terminal
  status, parity report, tiny selected-window features and temporal metadata.
- Fail: stop and retain artifacts; a crash is not a negative scientific result.
- Limits: ten windows from five clips, one CPU software environment; does not
  validate full extraction, GPU precision, activity-weight interpretation,
  benchmark comparability, release/license or pretraining overlap.

## Observed result

Completed exit0,85.0751s CPU, no retries, GPU-hours0. Five full video decodes
match ffprobe and DATA001_r1 PTS. Each adapter preprocessing tensor equals
the upstream tensor (max absolute difference0). Independent color oracle
max difference2.3842e-7. Ten selected windows pass batch-versus-serial:
features max9.5367e-7, latent0, raw activity outputs2.3842e-7.
All shapes/finite values pass. Nine unit tests include irregular-PTS/end
boundary and short/invalid timeline rejection; FEAT001_tests.txt records them.

Environment: Python3.13.5, torch2.11.0+cu128,timm1.0.20,albumentations2.0.8,
numpy2.5.3,cv2 5.0.0. No package install/shared environment changes.
Evidence: FEAT001_status.json, FEAT001_environment.json, FEAT001_parity.json.
Original hashes/snapshots and60311-byte selected-feature NPZ retained in the
exclusive runtime directory. This resolves environment blocker B003 for CPU.
Batch forward40.6553s over10 windows is a software timing observation only;
not GPU throughput or end-to-end latency. Full corpus/GPU parity not established.
