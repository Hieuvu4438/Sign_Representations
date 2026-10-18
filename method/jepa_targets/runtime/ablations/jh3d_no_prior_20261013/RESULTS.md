## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-13
- Verification Status: ANALYZED (single training seed)
- Version Label: jh3d-no-prior-v1

## Experiment Result

JH3D ablation: remove all nine geometric priors plus activity supervision; retain latent JEPA prediction. Both use Kinetics-MAE initialization, the same 3,000-video ASLC002 corpus, seed 20261008, six workers, 16 videos / 32 views per batch, 4,000 steps, 80% masking, predictor, optimizer and EMA 0.99925. Existing JH3D checkpoint is the reference.

Primary metric: ASL Citizen validation leave-one-signer-out R@1, white512 + mean pooling. Whitening is fit on validation windows as in the original exploratory EX16 protocol; no test-set access. No seed replication or checkpoint selection.

| Readout | JH3D R@1 (%) | No prior R@1 (%) | Delta (pp) | Signer bootstrap 95% CI (pp) |
|---|---:|---:|---:|---|
| pipeline_embedding | 39.15 | 5.63 | -33.52 | [-36.67, -25.77] |
| backbone_before_fc | 38.05 | 5.59 | -32.46 | [-35.42, -24.94] |

The pipeline readout includes fc: its weights are learned by prior loss in JH3D and remain at initialization in the ablation. The backbone readout uses mean tokens before fc for both arms, to compare encoder representations independently of the projection head. Confidence intervals resample validation signers, not training seeds; a single seed cannot establish robustness across training randomness. Full pooling, probe and geometry results are in evaluation.json; per-signer deltas are in comparison.json.
