# Method provenance

This package implements the proposed adaptation `pose_jepa_hybrid_v1`, independently
from proposal14, not an exact reproduction or checkpoint-compatible variant of an
upstream paper.

* [Uni-Sign](https://arxiv.org/abs/2501.15187): 2D COCO-WholeBody input and 69-joint
  body/hand/face partition. Its generative sign-text pretraining, RGB fusion and
  mT5 performance do not establish performance of this pose-only SSL model.
* [S-JEPA](https://sjepa.github.io/): full-context EMA teacher, latent predictor,
  centered/sharpened latent CE. Its action-skeleton augmentation and motion-aware
  masks are not enabled in this baseline.
* [SignRep](https://arxiv.org/abs/2503.08529): motivation for sign-specific physical
  priors and frozen retrieval evaluation. Its RGB backbone, 3D priors and released
  weights are separate from this 2D encoder.
* [MASA](https://arxiv.org/abs/2405.20666) and
  [MAMP](https://arxiv.org/abs/2308.07092): motion-target motivation. Our timestamp-
  and confidence-supported 2D losses are adaptations, not their original recipe.
* SignSeek: guide-specified motivation for gloss-supervised metric adaptation.
  No source/checkpoint is assumed available; this is a later supervised stage.
* V-JEPA/V-JEPA 2.1: reference for predictor, trainer and EMA bookkeeping only.
  Dense multi-layer targets and RGB teachers are not enabled.

Guide-specific decisions: spatial-only GCN with channel LayerNorm, shared hand
frontend, ordered flatten/project aggregation, clip-median shoulder reference,
2-frame packet tokenization, static part/span masking, quality-weighted center,
geometry/motion weights and deterministic per-part mean concatenation. Reference
origin/scale are fixed before masking and windowing; this is offline part-segment
prediction, not causal/raw-coordinate masking. Confidence is reliability, never z.

Stage 1 starts from scratch. Stage 2 uses actual train gloss labels and must be
reported as supervised metric adaptation. Frozen raw retrieval does not learn a
gallery projection. Source/weight status is recorded in `sources.lock.json`.
