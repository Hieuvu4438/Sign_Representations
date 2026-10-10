# Reproduction delta

This implementation is the proposal9 mechanism study, not an asserted
reproduction of published SignRep results. Official code/weights are source
assets, not proof that the full prior/style training recipe is available.

The dense target is 119 2D COCO-WholeBody nodes with global/local geometry,
velocity and optional analytic lag/phase. The official pooled prior head uses
different pose/angle/distance shapes. No official angle lists, style targets
or unpublished training pipeline are invented. A1-inspired must be labeled
explicitly: it is now implemented as a pooled-vector MLP predicting the shared
2D geometry sequence under the same geometry/regularization loss. It has no
cross-attention to dense encoder tokens and no velocity loss. A1_ref now uses
the actual official preprocessing and strict-loaded `fc`; native T16/stride2
per-window readout matches the full official wrapper (post-fc max error
4.77e-7). Averaging post-fc windows into a video embedding is our disclosed
downstream extension to the published window-feature example. Generic MAE and continued
SignRep adaptation remain separate studies.

The tiny Conv3D encoder is a test fixture only. Its synthetic losses and
retrieval metrics cannot populate scientific benchmark tables. Real How2Sign
A7/A5/A9 short gradient checks have run; no real matched frozen-transfer result
has been produced yet. Generic final-stage readout excludes MAE fusion and
encoder_norm; all arms within that study share this choice. SignRep and generic
weights are never loaded across backends.
