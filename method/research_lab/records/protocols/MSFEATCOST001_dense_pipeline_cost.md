# MSFEATCOST001 — seven fixed dense extraction cost clips

Pre-quality diagnostic, fixed planSHA/IDs/native counts from
MSFEATURE_COST_METADATA_PLAN: train_13060/03183/14520, val_02934/03598/02277,
train_02910;396native16stride2 windows, ONE firstwindowwarmup. No selectedqueries,
supports or model changes based on outcomes. Fulltask1037clips/36424windows fixed.
Validated MSFEATTEST001 collector/manual pooling, prior MSESTENV001 actual43input
parity and MSFEATUREGPU002 checkpoint/wrapper/CPU numerical compatibility bound.

For each accessedsource verify bound SHA by context and byte/mtime before/after
decode; preserve estimated/native raw prefix and ordinals, crop coordinates,
allRGB/first16/count/PTS hashes identical frozen extractionmanifest. No source/
RGB/video copies retained. Normalize with unchanged upstream-compatible wrapper;
strict SignRep/checksum/eval/mask0/no-grad/FP32/batch1. ALLcomplete native16 stride2
windows, collect features768/rawactivity2. Pool original max(clip(activity,0,1))
FP64 weighted mean/fallback/L2. Do not interpret activity as labelgroundtruth.
No scores/ranks/geometry/labelsfit; only finitefeature and inputidentity checks.

Timings disclose source/reference/decode/crop, preprocess, GPUencoderwithtransfer/
copy, FP64pool, ONEwarmup/model load. Perclip pipeline includes firstwarmup where
applicable; statuswall includes context inputhash/import/setup/output overhead.
GPU undercontending load, no idlelatency/end-to-end corpus guarantee. Record actual
versions/device/memory/forward counts. Keep raw396features/activity and7pooled
vectors for independently recomputable pooling/hash/join verification. Complete
store4MiB incl snapshots/config/status/log/summary/records/arrays/manifest.

GPU90s/job180s, H006cumulative120/program14400, charged before CUDA config/load.
Fresh budget with three roots/erratum; physical GPU0UUID/maxone ownedjob/inherited
visibility; allocator2GiB/min3GiBfree/max100util. Decoder/OpenCV/Torchthread1;
327selected/4096decode frames/source512MiB/ffprobe1MiB30s; RAM1GiB+2GiBreserve.
No silent policy retries, deadline/caps failures retained. No fulltaskquality
authorization from this run: conservative new fullallocation/protocol required.
