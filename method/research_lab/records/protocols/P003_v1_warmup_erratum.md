# P003_v1 — warmup source split correction before scoring

2026-10-04 local. Engineering incident found during the CPU source audit, before
any test encoder inference or score. Original P003/source configuration names
7597632941375478-HATCH 2.mp4 as a train warmup source. COHORT001 identifies it as
**val**, participant/class/sample/source identity fixed before test. The original
prediction guard would abort at warmup after model load. No prediction attempt
has launched; no failed/scientific run to invalidate and no test scores seen.

Correction: permit the registered fixed warmup source from train or val, always
reject test. Keep the identical video, number of four warmup windows, seed,
encoder, policy order, galleries, native input, methods, metric, uncertainty,
slices/case rules, thresholds, budgets and all three scientific configs. The
historical key `warmup_train_source` is retained to preserve its exact config/hash;
its train adjective is incorrect descriptive provenance. Do not use the original
train phrase in new reports. The existing pilot also used that same dev source.

Original pre-test freeze and protocol remain immutable. An additional issued
technical gate explicitly links their original SHA and records **after metadata,
before scoring** timing; it must not pretend its own timestamp precedes the first
test metadata access. No test feedback caused policy/design changes. Predictor
diff is only the eligibility guard/comment/error message. confirmatory.py diff is
only acceptance of this documented technical gate and enforcement of unchanged
scientific config hashes; other scientific functions unchanged. A fixture verifies
the erratum cannot reopen scientific configuration. CONFTEST002 regression result
and source snapshot bind the corrected code before GPU use.

PRETEST001 was already running with the original source snapshot/gate and can
complete: its audit logic is unchanged. Both its original gate reference and the
new prediction technical gate are retained. Measurements are scoped to the
original scientific design with this disclosed warmup engineering correction,
not a silently edited pristine-code freeze. No automatic source replacement,
new trial/seed or extraction retry is authorized by this erratum.
