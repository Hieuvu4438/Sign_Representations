# DATA001 — Train/dev video timeline smoke

- Registered: 2026-10-03 before execution, H001/H002, P001_v1.
- Type: CPU metadata/decoder preflight, not scientific performance.
- Inputs: ASL Citizen train/val CSV only; seed1701 selects two train/three val
  by seeded sampling from sorted filename rows. No test video touched.
- Probe: ffprobe frame PTS, stream dimensions/rate; hash five source files.
- Gate: positive frame count/dimensions, finite monotone PTS, readable source;
  report VFR/tail windows/short clips. A 5-sample pass cannot prove full decode
  coverage or estimate corpus runtime without representative stratification.
- Budget: one CPU process, each probe≤30 sec, total≤5 min; 0 GPU-hours.
- Output: exclusive `method/frozen_probe/runtime/DATA001_20261003/`.
- `DATA001_20261003`: completed, five timeline checks pass. Review found the
  diagnostic `vfr_signal_over_1ms` could be true at exactly 1 ms due to floating
  point residue; this field is invalidated. PTS/frame/source data remain valid.
- Fixed threshold with 1e-9 s numerical tolerance; renamed field to
  `pts_jitter_over_1ms` because jitter alone does not establish VFR.
- Retry 1 `DATA001_20261003_r1`: new report, same five seeded train/dev IDs.
  Added a regression oracle for exactly-1-ms roundoff and a real 7-ms gap.
  No science claim/run affected. Preprocessing/model parity still not tested.
