# CACHE001 — conditional full candidate window extraction

Update2026-10-04: superseded before execution by [CACHE002](CACHE002.md).
V1 manual accounting limitation below is historical; current code counts
all owned terminal attempts automatically. No CACHE001 output/cache exists.

Registered2026-10-04 local, H001/P001_v3 preprocessing prerequisite. Config
CACHE001.json, seed1701, proposed run CACHE001_20261004_P001v3_s1701,
exclusive output method/frozen_probe/runtime/CACHE001_20261004_P001v3_s1701.
**DEFERRED**: requires completed valid GPU001 parity/cost gate and free GPU.
No feature cache/run exists. No science score/HPO/test access.

All1691 fixed sources/61339 windows, no subsets or failures dropped. GPU0,
FP32 features768 and raw regression activity2, native16/stride2, maximum2000
decoded frames, fixed upstream/checkpoint/preprocessing/library identity.
Full source size/SHA/native timeline checked again. Manifest key includes all
those identities and context; source-specific metadata binds class/split/
participant/source hash, complete native PTS/FPS/window mapping via VID002.

Scratch only in /dev/shm under dedicated signrep_frozen_probe_<run-hash>, cap
1GiB, both tmpfs and hostRAM retain≥20% free after planned bytes and2GiB CPU
workspace. Feature payload≈180.2MiB plus timestamps/NPZ/manifest overhead.
No bulk writes to root partition while persistent20% reserve fails. Progress/
completion manifest and status kept durable under runtime, payload in RAM;
reboot loss requires explicit rebuild, never pretend payload persists.

Requires GPU001 hashes/environment/preprocessing and estimate match; allocator
cap8192MiB and idle preflight. H001 total≤3600s includes all attempts; program
≤4GPUh. Set timeout to minimum3600s/remaining H001 allowance. If a failed GPU
attempt already spent budget, register revised config before any retry; CLI
v1 supports one successful smoke and fresh extraction, not a cumulative-budget
scheduler. No partial automatic resume; audit owned partial shards/status and
register a diagnosed new attempt. Completed cache consumer only verifies,
never mutates/reuses by filename alone. Full manifest+hash required to train.

Success: all shards finite float32, native centers float64, complete coverage,
atomic manifest completed. Source failure/timeout is engineering failure.
Exact command deferred until GPU and representative-cost gates are satisfied.
