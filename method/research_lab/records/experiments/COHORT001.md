# COHORT001 — train-only selection and support manifests

- Registered2026-10-03 before execution; H001 proposed pilot, P001_v2 draft.
- Type: CPU metadata/source-integrity audit, no features/predictions/training.
- Select100 classes using random.sample(sorted train vocabulary), seed1701;
  sort selected classes for contiguous local IDs. Do not inspect test metadata.
- All validation rows for these classes retained. Three support seeds1701/2/3;
  each class gets a shuffled train-ID list; budgets1/2/4/8 are prefixes of the
  same first8 samples. Every later method uses these manifests unchanged.
- Gate: train/dev sample/participant separation, complete100-class support,
  train-only mapping, nested supports and all source paths exist. Hash only the
  union of selected train supports and dev videos; list exact duplicates.
- Stop measurement for any cross-split exact duplicate; within-split duplicates
  reported for support-design review. Near-duplicates/session identity remain
  unaudited and cannot be represented as passing here.
- Budget: single CPU process, ≤900s, ≤4GiB source reads, zero GPU/downloads;
  outputs <2MiB. No test video/labels or package/shared-env mutation.
- Exact config `method/frozen_probe/configs/COHORT001.json`.
- Exclusive output `method/frozen_probe/runtime/COHORT001_20261003_s1701/`.
- Success freezes candidate cohort IDs, not a VALIDATED experiment protocol.
  Feature coverage, numerical controls, novelty and runtime gates remain.

Observed: completed exit0,.9129s CPU. Selected100 classes,1313 train-union
support videos and378 dev videos. SHA256 computed over1,038,294,125 source
bytes; no exact duplicate group. No test access. Cohort JSON871,541bytes,
hash d3327a955efce9128fc10f1e13e5daae884ba0abe8c0920ff48dcc1339b25830.
Eleven unit checks pass, including nested supports and leakage rejection.
Evidence COHORT001_status.json, inventory/COHORT001_cohort.json and test log.
No full video decode/near-duplicate audit or measurement gate pass claimed.
