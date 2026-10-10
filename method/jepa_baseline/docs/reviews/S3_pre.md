# S3 pre-implementation review

Primary authority: main §§5.1–5.2 equations 2–10, supplement A3 Table9 and A4.
Reuse upstream heads; training objectives are absent from the released repo.

Reconstruction: scale targets exactly once, retain unscaled predictions, use SmoothL1
beta=1 and mean over valid elements per prior, then weighted sum (U05). Fully masked
priors give a graph-connected zero. Reject valid nonfinite targets/predictions.
Variance sums feature-dimension hinges, sample standard deviation (N-1); covariance
uses denominator N and sums both directions of every off-diagonal. No VICReg mean.

Style is a channel×channel Gram matrix of final encoder tokens, averaged over columns;
tokens×channels normalization is U04. Discriminator emits logits for stable BCE, but
equations 7–9 use probabilities, not logits. Two style linears plus GELU after each,
LayerNorm; four spectrally normalized match linears with GELU except the last.

Matched sample is the other view of the same video. Unmatched sample must come from
a different video, not just a different batch index. Encoder and discriminator have
disjoint optimizers. Discriminator sees detached content/style. For encoder adversarial
loss, freeze discriminator parameters and spectral-normalization updates; keep content
gradient and detach style (explicit U04 convention). EMA momentum means new=0.9old+0.1mean,
initialized to zero; update after the discriminator step from its current probabilities.
Gate strictly when matched EMA > unmatched EMA; minimize squared hinges to lower
matched scores and raise unmatched scores. Verify this with independent derivatives.

Acceptance: hand-calculated masked/scaled SmoothL1 and feature losses; Gram oracle;
independent analytic adversarial derivatives; EMA/gate/resume state; invalid pairing;
finite constant-feature gradients and frozen-discriminator gradient ownership.
