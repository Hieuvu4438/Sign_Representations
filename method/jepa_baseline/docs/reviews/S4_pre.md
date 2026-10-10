# S4 pre-implementation review

Authority: supplement A2/A3/A4, main training and retrieval protocols. Training data
must be frame-aligned standardized teacher cache and matching video content. Pretraining
manifest excludes every WLASL YouTube ID, including held-out splits. IDs, file hashes,
bone-table hash and preprocessing policy travel with each run.

Dataset item is one video, yielding two independent contiguous 16-frame windows;
batch default is 10 unique videos/20 total views (U07). Select only windows whose
body detector succeeded throughout; reject short inputs rather than silently resample
time (U06). Spatial/color transforms share parameters across every frame of a view,
with independent parameters across views. No horizontal flip without transforming
priors. Reuse official Planckian jitter code; preserve original inference resize and
ImageNet normalization. Augmentation probabilities/crop range are explicit U06 values.

Representation AdamW and discriminator Adam use separate parameter sets and schedules.
Layer decay maps patch/position embeddings to layer0, transformer block i to i+1,
remaining heads to final layer; bias/norm/position exemptions follow Hiera MAE convention
(paper does not state the exemption rule). Warmup first update is LR/50000; final
update zero by explicit endpoint convention. EMA follows S3 ordering.

Exact resume requires optimizer, step, scheduler, scaler, EMA, Python/NumPy/Torch RNG,
pair sampler permutation/cursor/generator, and dataset RNG; prohibit worker prefetch
for exact replay in default num_workers=0. Validate run configuration and data hashes.
Save checkpoints atomically after complete encoder steps. Best selection must receive
validation-only retrieval accuracy at 25000-step intervals; never accept test scores.

Acceptance: real paired data backward/update/checkpoint/resume; schedule endpoints;
unique sample coverage and pair identities; reproducible next batch; held-out exclusion;
coherent augmentation; optimizer coverage/disjointness; validation-only selection.
