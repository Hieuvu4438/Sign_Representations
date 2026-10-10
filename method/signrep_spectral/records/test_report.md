# Test report (2026-10-05)

Command (interpreter `.venv-signrep/bin/python`, CPU, `CUDA_VISIBLE_DEVICES=`):
`PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s method/signrep_spectral/tests -t . -p 'test_*.py' -v`

Result before the regularizer addition: 35 tests, all passed (~55 s). After it, `test_losses_targets`
(including the new `TestRegularizers`) passed again.

| Group | Kind | Covers |
|---|---|---|
| test_spectral | synthetic | window/unfold shapes, periodic Hann, independent NumPy direct-DFT oracle (<=1e-10), identity/offset/amplitude/sign-inversion, reversal limitation, Parseval incl. one-sided weighting, all-missing, partial-missing window selection, invalid payload invariance, declared-valid NaN and prediction NaN raise, short clip raise, `gradcheck` (float64), zero grad at invalid times, fixed denominator, window-time control |
| test_losses_targets | synthetic | joint-major/wrist origin/translation invariance/aspect, scale rules, r2 extent rule, stable scaler sampling, geometric/velocity/acceleration losses, tiny encoder+decoder spectral-only gradient wiring, lambda>0 changes gradients, baseline never calls spectral, config validation, var/cov regularizers |
| test_evaluation | synthetic | class-max ranking oracle (RR=0.5 example), duplicate support, ties, micro/macro MRR, self-match / absent class / NaN errors, paired bootstrap |
| test_vendor_integration | **real** checkpoint/vendor | strict load + head dims, original-vs-adapter parity (0.0), SDPA shim parity (1e-5) and restoration, explicit mask semantics, seeded decoder init, preprocessing vs vendored albumentations transform, import-collision rejection, cv2 vs PyAV pixel equality on a real ASL Citizen video |

Real-data pipeline checks outside unittest: export parity (max abs 0.0) for every exported
encoder, identical initial-state hash and sample schedule for B1/M1, audit of all 83,399 videos.
