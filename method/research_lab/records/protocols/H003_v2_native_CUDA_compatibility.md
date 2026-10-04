# H003_v2 — Additional native CUDA producer compatibility

Prelaunch decision [D019](../decisions/D019.md), config AUTHORGPU001.json
ac538b19cfb1fa3d89b562d587b85be2034a5492f9486eaabd94444b8299e8a6.
This version adds an exact author GPU function comparison and does not change,
replace or rescue the failed H003_v1 CPU/GPU gate. Question: do separate literal
native and own strict-loaded producer paths return bitwise equal12×T×768 arrays
on all4fixed train/dev sources/396nativeframes? Primary software predicate:
all four arrays finite/correct shape/bitwise equal. Null/falsifier: any difference.
No task score, quality gain, population inference, HPO or semantic-failure labels.

Baseline: readonly author load_model/process_sample with exact published bytes.
Changed variable: callable assembly/loader path; controlled checkpoint/input
bytes/precision/device/eval/masking/context. Own strict226/226, native permissive
load verified all226values; zero labels; whole offline context; saved layer-last
separate from final_x. Inputs replay previous SHA-bound DINO streams and native
body features, masks/PTS carried forward. Seed1701, no trial/seed selection.
Direct first-context prior-GPU equality is diagnostic, not selection criterion.

GPU<=180s, one allocated device/8GiBallocator/min8GiBfree;RAM64MiB/20%reserve.
RunContext saves original command/config/env/ownedsource snapshot/terminalstatus;
report includes reference/output paths/hashes and all source IDs. Fail immediately
on any predicate breach; no repeated identical extraction or tolerance change.
Scientific pilot/scale/claims remain hold after either outcome. Same-agent roles
and resource/evidence limitations are in D019.
