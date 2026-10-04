# LOAD001 — SignRep checkpoint/interface audit on CPU

- Registered: 2026-10-03 before execution; H001/H002 prerequisite; P001_v1.
- Type: CPU software audit; no downstream performance measurement.
- Inputs: local checkpoint and pinned read-only SignRep `example_usage.py`.
- Question: Can the shipped inference architecture load every checkpoint
  tensor without the upstream broad exception silently dropping the prior head?
- Method: safe `weights_only=True` load, AST literal model parameters,
  instantiate official model on CPU, compare all keys/shapes, then strict load.
- Gate: zero missing/unexpected/mismatched keys and all expected heads exist.
- Limits: one CPU thread; <15 minutes; no GPU/no download/no shared env changes.
- Output: exclusive `method/frozen_probe/runtime/LOAD001_20261003/`.
- Failure is an environment/interface blocker, never a scientific negative result.
- Not evidence of preprocessing, inference parity or paper benchmark reproduction.
- First attempt `LOAD001_20261003`: failed before model construction because
  `literal_eval` rejects the example's integer shape products. Fixed the new
  audit script with an AST transform restricted to integer multiplication.
- Retry 1: `LOAD001_20261003_r1`, same budget, new output directory; keep the
  failed attempt and its script hash. No environment modification.
- Retry 1 failed at import: base Python 3.13 / timm 0.4.5 lacks `Mlp` at
  `timm.models.layers`. No weights were loaded, no inference ran.
- Retry 2: `LOAD001_20261003_r2`, existing read-only `seds` environment,
  Python 3.10 / torch 2.3.1+cu121 / timm 1.0.29. Import of Mlp/DropPath
  confirmed before this attempt. This is the final allowed retry for LOAD001.
- Retry2 completed, exit0, 1.8028 s CPU wall time. All keys/shapes match;
  strict load pass, coverage1.0. [Coverage](LOAD001_coverage.json) and
  [status](LOAD001_20261003_r2_status.json) support F004. No feature inference.
