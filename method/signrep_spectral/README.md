# signrep_spectral — SignRep-initialized canonical continuation + spectral auxiliary

Implements `docs/proposal3/signrep_spectral_implementation_spec.md` (mode
`canonical_continuation_v1`). The RGB encoder (vendored Hiera + `fc`) is warm-started
from the local SignRep checkpoint and trained to reconstruct canonical 2-D hand
trajectories through a **new** temporal decoder. Arms differ only in the auxiliary
objective: none (B1), masked multiscale log-magnitude spectral L1 (M1), velocity (B2),
acceleration (B3), windowed time-domain L1 (B4). Inference is RGB-only (`backbone+fc`).
Results are **not** a SignRep reproduction and the retrieval metric is an adapted
protocol, not an official benchmark.

Records: `records/` (discovery, decisions/deviations, pre-registration, results).
Runtime outputs: `runtime/` (git-ignored, one exclusive directory per run).

## Commands (run from the repo root)

```bash
PY=.venv-signrep/bin/python; export PYTHONDONTWRITEBYTECODE=1
C=method/signrep_spectral/configs; R=method/signrep_spectral/runtime
P="--paths-config $C/local_paths.json"   # git-ignored machine paths (dataset/manifest/scaler/calibration)

$PY -m method.signrep_spectral.cli preflight   --config $C/base.json $P --run-dir $R/preflight_XXX
$PY -m method.signrep_spectral.cli audit-data  --config $C/base.json $P --run-dir $R/data_audit_XXX
$PY -m method.signrep_spectral.cli fit-scaler  --config $C/pilot_base.json $P --run-dir $R/scaler_XXX
$PY -m method.signrep_spectral.cli build-targets --config $C/pilot_base.json $P --run-dir $R/targets_XXX  # coverage audit
$PY -m method.signrep_spectral.cli calibrate   --config $C/pilot_base.json $P --run-dir $R/calibration_XXX
$PY -m method.signrep_spectral.cli train       --config $C/spectral_mvp.json $P --run-dir $R/M1_train_XXX
$PY -m method.signrep_spectral.cli export      --config $C/spectral_mvp.json $P \
    --checkpoint $R/M1_train_XXX/checkpoints/final.pt --output $R/M1_export_XXX/encoder.pt --run-dir $R/M1_export_XXX
$PY -m method.signrep_spectral.cli extract     --config $C/spectral_mvp.json $P \
    --encoder $R/M1_export_XXX/encoder.pt --splits train,val --run-dir $R/M1_features_XXX
#   (arm B0: --reference-checkpoint instead of --encoder)
$PY -m method.signrep_spectral.cli evaluate-retrieval --config $C/pilot_base.json $P \
    --features-dir $R/M1_features_XXX --query-split val --run-dir $R/M1_retrieval_XXX
$PY -m method.signrep_spectral.cli evaluate-probe --config $C/pilot_base.json $P \
    --features-dir $R/M1_features_XXX --eval-split val --run-dir $R/M1_probe_XXX
```

`scripts/run_pilot.sh <tag> B0 B1 M1 ...` chains train → export → extract → evaluate per arm.
`local_paths.json` must set `dataset_root`, `manifest` (from `audit-data`), `scaler`
(from `fit-scaler`) and `calibration` (from `calibrate`); missing fields exit non-zero.

Tests (CPU; vendor tests are skipped when the checkpoint is absent):

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv-signrep/bin/python \
  -m unittest discover -s method/signrep_spectral/tests -t . -p 'test_*.py' -v
```

## Dependencies actually used
Existing `.venv-signrep` only: torch 2.11, numpy, OpenCV (cv2.VideoCapture decode, as upstream),
PyAV (PTS audit), albumentations (parity test only). Nothing was installed.

## Known limitations
- 2-D hand targets only (no z, no body/face/angles/distances); pose backend identity unknown;
  handedness from the backend, no track-swap detection beyond single-person frames.
- 16-frame context: N=8/16 windows cannot represent slow, long-range rhythm.
- Resume, DDP and gradient accumulation are rejected (not implemented).
- Extraction stride 16 and bf16 for budget; per-window features are not stored.
- SignRep pretraining overlap with ASL Citizen is unknown.
