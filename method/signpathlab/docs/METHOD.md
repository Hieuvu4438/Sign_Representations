# Method and implemented contracts

Default architecture is the D0 target-only variant in proposal13. Four frozen DINOv2
visual streams and three static body/hand pose streams have separate MLP projectors.
Each projected token receives public stream identity and segment-relative physical
time encoding. Hidden/padded values are zeroed before their projector, then entirely
excluded from the encoder attention sequence. The full valid teacher sequence has
the same time origin; masking never resets the first-visible timestamp to zero.

The teacher is an exact, detached copy of projectors/type embeddings/encoder, without
the predictor. It remains in eval mode even when the research model is trained.
EMA and the LR schedule advance only after globally finite, successful optimizer
updates. Bias/norm parameters have no weight decay in the default recipe.

Semantic prediction uses packed metadata-only decoder queries and normalized L1
or cosine targets. Deterministic geometry uses the same public path/scale/requested
time-window metadata, with no anchor/quality/ROI/teacher values in queries. Auxiliary
SmoothL1 is averaged by globally active group/scale; semantic loss is averaged by
globally active stream. DDP local losses multiply by world size before its averaged
gradient reduction. Accumulation averages microbatch global means, as specified.

Geometry retains the ordered degree-2 Lie basis, anchors and actual timing/motion.
Ordinary-motion controls share paths/windows, q32 output and query architecture.
All target standardization/projection is train-only. Fitting covers every center
phase a randomly offset segment can expose, rather than only full-clip indices 0,4,….
Cached support propagation includes extraction/normalization prerequisites; strict
geometry targets need fully hidden support. Raw RGB masks close over source-frame
aliases. Duplicate observations contribute at most once to semantic targets and
clip pooling. Encoder context also includes one key per source observation; encoded
values scatter back to all selected alias slots. Real timestamps/dense validity remain available for export.

An empty sequence receives a safe zero attention sentinel to prevent all-masked
softmax NaNs. Sentinel/padding outputs never contribute to loss or pooling. Empty
local ranks still use a connected autograd graph and the same collective order.
Global empty semantic targets skip the optimizer/EMA/scheduler consistently.

Unmasked online output scatters back into the seven raw slots per timestamp, then
averages valid streams and valid distinct-source frames. Predictor and teacher are
excluded from inference APIs. Linear/attentive heads consume frozen or trainable
embeddings depending on the explicit downstream setting. CTC uses stride1 lengths,
rejects impossible repeated-label alignments and exposes greedy decoding. The toy
seq2seq decoder has causal teacher forcing and greedy BOS/EOS decoding.

Optional geometry-input/deep/visible-dense/regularizer variants reject execution
until implemented. Current code does not claim faithful reproduction of an external
JEPA/SHuBERT/SignRep recipe or establish signature utility from small diagnostics.


Anatomical cross-modal masks use a separate pose-observation primitive domain. RGB
and its detector-derived ROI priors remain public; edits test hidden pose measurements
with those RGB priors fixed. Stream-temporal and modality-dropout masks operate on
explicit feature-observation primitives. The same set-intersection closure resolver
applies in every regime; these masks never acquire the strict spatial hiding label.

The deployment graph uses explicit eval MHA algebra with the trained QKV/out weights,
then the trained normalization/MLP blocks. This avoids Torch 2.11 SDPA export guards
without restricting T to a fixed sequence length. Parity checks include packed eager
inference, serialized dynamic graph, padding, missing streams and all-invalid inputs.
The selected linear/attentive head is exported; predictor and EMA teacher are absent.
`ExportedSession.full_clip` applies the same slot-aligned tiling and global head policy.
