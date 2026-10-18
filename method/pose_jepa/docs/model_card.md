# Pose representation model card

Implementation: `pose_jepa_hybrid_v1` architecture family, proposal14 adaptation.
The code is independent; it does not claim to reproduce Uni-Sign, S-JEPA, SignRep,
MASA or MAMP exactly. See `method_provenance.md` and `sources.lock.json`.

## Architecture and input

COCO-WholeBody 133-joint viewport-normalized 2D pose cache with confidence,
observations, source indices, monotonic known timestamps and track continuity.
Selected parts are body/LH/RH/face with 9/21/21/18 joints. Shared hand spatial GCN,
channel LayerNorm, ordered joint projection, independent 2-frame packet tokens,
8-layer width-256 Transformer, stride 2. The encoder has **10,255,872 parameters**.
Sequence and raw clip dimensions are 1024. Raw clip pooling is deterministic valid
time mean per part followed by concatenation. A 256-dimensional metric projector
is included only in a separately trained, matched online Stage 2 bundle.

Normalization uses the full clip's fixed shoulder reference before windows/masks.
This is offline inference, not causal streaming. Confidence is reliability, not
depth. The 18 face points and 2D geometry do not resolve every non-manual marker,
orientation/depth ambiguity or visually indistinguishable sign.

## Training and evaluation scope

Five real pilots used the same 16 official ASL-Citizen train clips, four classes,
source PTS, full architecture, seed 42, effective update batch 2, 5,000 successful
updates each and frozen validation every 1,000 updates. Train labels were used to
stratify this small subset; labels/text never enter Stage 1 features or losses.
Protocol is `in_domain_ssl`, with **target-validation selection**, not zero-shot
or cross-corpus transfer. Validation/test contain 8 independent clips each.

Existing detector weights/profile are unverified. Positive unbounded SimCC scores
use an explicit `q=s/(1+s)` legacy adaptation with raw-score sidecars. Scores are
uncalibrated. Visual audit found motion blur, occlusion and extrapolated offscreen
hands. The subset cannot establish full ASL-Citizen, WLASL2000 or NMFs-CSL quality.

| SSL objective | Best validation R@1 | Selected update |
|---|---:|---:|
| Masked geometry | 0.750 | 2000 |
| Masked motion | 0.250 | 1000 |
| JEPA only | 0.625 | 1000 |
| JEPA hybrid | 0.625 | 1000 |
| Shuffled JEPA diagnostic | 0.250 | 1000 |

Primary dictionary retrieval ranks glosses by maximum video cosine, with fixed
query denominators and deterministic ties. The selected raw example bundle is
the **geometry baseline**, selected by validation; the hybrid bundle is retained.
Test was not used to change that selection. Geometry/hybrid test R@1 are 0.125/
0.250 on 8 queries. Their frozen linear probe test Top1 values are 0.250/0.375.
These tiny-sample results expose selection uncertainty and do not demonstrate
hybrid superiority. R@5/10 and Top5 saturate on a four-class experiment.

The shuffled-target final inter-clip cosine is approximately 0.99995 with mean
feature std 0.00573, while the matched JEPA control retains substantially more
variation. This is a correspondence/collapse sanity control, not a benchmark.

Three Stage 2 controls each used 2 head-only +18 joint updates, P=4, K=2, same
train/validation data: hybrid with/without SSL replay and geometry initialization
with replay. Each validation R@1 is 0.375. These are short supervised adapter
sanities, not the full 1,000+10,000-update metric recipe or evidence of a gain.

A supervised scratch sanity overfit all 16 train clips (100% train Top1). Frozen
CSLR ran on single isolated gloss labels only; it is not a continuous CSLR result.
Real continuous transcripts and paired translation corpora were not supplied for
this recipe. mT5 bridge tests use an actual small randomly initialized mT5, not
loaded `google/mt5-base` weights; its external config/revision is pinned separately.

## Bundles and usage

`runtime/selected_pose_encoder`: validation-selected geometry online encoder.
`runtime/hybrid_pose_encoder`: hybrid EMA encoder from best validation.
`runtime/metric_pose_encoder`: hybrid online encoder with trained online metric
projector after the 20-update supervised sanity. All are experimental artifacts.
Training checkpoints remain separate from stripped inference bundles.

CPU parity was verified against source checkpoints. On the audited 53-frame
cached pose, sequence output is `[27,1024]`, raw clip output `[1024]`; measured CPU
sequence latency is about 29 ms at two threads. This includes canonicalization,
encoder and stitching; it excludes pose detection/video decoding. It is not
end-to-end camera throughput or a latency guarantee.

Checkpoint/bundle/config/source/data checksums and executed dependency versions
are recorded in the evidence directory and runtime metadata. Exact source
snapshots are retained for source-matched resumes. Multi-process CPU Gloo and
single-GPU AMP are tested; multi-GPU NCCL was not executed on this one-GPU host.

No RGB, 3D lifting, saliency, SIGReg/VICReg or untrained fusion module is enabled.
No SOTA, full-benchmark, strict unseen-language or linguistically annotated
minimal-pair result is claimed. Failed queries remain in metric denominators.
