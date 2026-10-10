# Experiment recipes and scientific gates

`configs/ablations/` contains resolved-compatible **unrun recipes**, not paper
results. `reports/ablation_recipes.json` lists their families and gates. Every
recipe uses 200 optimizer updates and effective batch 4 on this one-GPU host.
Use the same manifest, seeds, optimizer and frozen evaluation protocol within a
family. Spatial resolution/frame-count experiments also change FLOPs and memory;
report those costs rather than describing them as equal compute.

| Specification family | Available configuration | Status and interpretation |
|---|---|---|
| Generic / node / relation / combined targets | Existing `configs/pretrain/{jepa_baseline,articulator_mvp,relation_controlled,combined_controlled}.yaml` | Fixed three-seed A/B/C/D pilot is the primary causal target-family comparison |
| Standalone node / relation | `articulator_only.yaml`, `relation_only.yaml` in `configs/pretrain` | Separate objectives; do not substitute their results for adding a target to A |
| Relation graph and face removal | `graph_lh_rh`, `graph_hand_face`, `graph_full`, `face_off` | Random masks and no auxiliary node loss isolate the graph; face removal here means removing face targets/edges, not censoring face pixels |
| Projector | `projector_concat`, `projector_concat_diff_product` | Other recipe settings held fixed |
| Masking | `mask_random_multiblock`, `mask_articulator`, `mask_relation_break`, `mask_mixture` | Structured branches intentionally have different realized corruption sizes; log their visible ratios |
| Stronger relation masks | `both_endpoints_node`, `both_endpoints_combined` | Both endpoints removed during the span; 380/380 structural draws feasible, no trained comparison yet |
| Latent loss | `loss_l1`, `loss_smooth_l1`, `loss_cosine` | Same target normalization |
| Resolution | `resolution_224`, `resolution_256`, `resolution_384` | Official RoPE adapter; actual shape/load checks still required for each run |
| Clip duration | `frames_16`, `frames_32` | Same single-scale target; architecture/clock checks required |
| Temporal scales | `temporal_short`, `temporal_short_medium`, `temporal_short_medium_long` | Phase-2 recipes only; do not train before the single-scale gate passes |
| Regularization | `regularization_false`, `regularization_true` | Optional label-free variance/covariance objective; never silently enable it to rescue failed relation results |
| Pose dependency | `rgb_without_landmarks` | Generic RGB-only objective. To test no landmark dependency, use an SSL whitelist manifest without `landmark_path`; it is not a pose-free relation-target discovery method |
| Initialization | `random_vitb`; existing pretrained configs | Frozen random/pretrained references and same-initialization continued-SSL comparisons have different meanings |
| Geometry in target | Current visual-only relation target | Explicit relative geometry is optional and remains deferred |

Multi-scale, learned salience/discovery, hypergraphs, adversarial invariance and
large-corpus training remain gated by specification §§19, 44–48. A negative pilot
must first inspect target predictability, contextual node redundancy and masking;
adding regularization or increasing the backbone is not evidence for the original
relation hypothesis.

The controlled pilot uses the original archived masking implementation. Its
random branch bounded a block/tube mask by dropping excess spatiotemporal tokens,
so it was a hybrid random corruption rather than strictly complete tubes. The
final engineering implementation retains complete temporal tubes. All A/B/C/D runs in the
fixed suite use the same original implementation; future comparisons must use a
new run root and record the new source fingerprint.
