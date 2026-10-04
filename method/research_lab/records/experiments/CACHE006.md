# CACHE006 — P004 fixed-policy dev-first native cache

H002/P004_v1, extraction only/no quality outcomes.661 fixed train/dev sources,
24001 complete16-frame stride2 windows,278train/383dev; one-label three galleries
used later. New classes fixed by COHORT002/VID003/D012. Native encoder/adapter,
checkpoint/packages and upstream sources unchanged from GPU004 parity gate;
original P003 source/config bindings validated. [Prospective dev gate](../protocols/P004_v1_dev_gate.json)
issued23:33:32.194846UTC after REPTEST001 four manual integration oracles pass.
Old GPU004 input audit is not asserted to cover new source identities; new
VID003/cohort hashes bound separately. Shared granted GPU0 UUID identity,
8GiB allocator cap; one owned job;1500s hard ceiling/program14400/H0027200s.
All allocated GPU time including setup/persistence charged, failures retained.
Tmpfs1GiB cap/reserve; no test access, no new policy/HPO selection.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-signrep/bin/python method/frozen_probe/scripts/extract_replication.py --config method/frozen_probe/configs/CACHE006.json --seed 1701 --run-id CACHE006_20261004_H002_P004v1_bf385467_s1701 --output-dir method/frozen_probe/runtime/CACHE006_20261004_H002_P004v1_bf385467_s1701 --launch-gate method/research_lab/records/protocols/P004_v1_dev_gate.json
```

Terminal status/report/source snapshots remain authoritative; raw native payload
bounded in RAM and hashed durable manifest copy. CPU wrapper/test preparation
may overlap extraction; do not interpret allocated wall as uncontended hardware
throughput. No per-query timers/quality claim from this extraction.
