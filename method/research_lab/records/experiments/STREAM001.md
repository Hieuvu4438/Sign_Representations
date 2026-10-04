# STREAM001 — four raw SHuBERT input candidates

H003_v0 input audit only, two train/two dev IDs fixed lexically from COHORT001.
Bound raw source hashes/native frame counts/upstream code; max4MiBtotal/1MiBfile,
60sCPU. Raw coordinates/scores described literally; no threshold/reliability
label/crop extraction/model performance. New direction source snapshots and
separate pinned upstream snapshots retained. Basic run infrastructure reused
from this new research program's own frozen_probe; no legacy/project-method code.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/stream_reliability/scripts/audit_inputs.py --config method/stream_reliability/configs/STREAM001.json --seed 1701 --run-id STREAM001_20261004_H003_v0_dbd8c39f_s1701 --output-dir method/stream_reliability/runtime/STREAM001_20261004_H003_v0_dbd8c39f_s1701
```

Terminal CPU.028631688s,0GPU; four files finite133-point/coordinate and score
arrays, lengths match native156/87/82/71frames; no timestamps or verified score
semantics. [D015](../decisions/D015.md) holds model measurement and queues actual
input/provenance/environment audit; existing DINO/model assets verified present.
