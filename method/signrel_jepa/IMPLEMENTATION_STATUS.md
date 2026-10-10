# Implementation audit — 2026-10-06

The isolated engineering MVP and specification §45's fixed A/B/C/D evaluation are
complete. The scientific result is **no-go for multi-scale research training and
scale-up**: relation variants do not show consistent frozen retrieval/phonology
benefits over both controls. This is a completed negative pilot, not an unfinished
experiment or a claim of SOTA. See [REPORT.md](REPORT.md).

## Verified gates

| Requirement | Evidence | Scope |
|---|---|---|
| Isolated method | Entire new `method/signrel_jepa` | Other method source/configs and existing environments were not modified |
| Environment/provenance/licenses | `requirements.lock.txt`, `third_party/PROVENANCE.md`, `reports/assets.json` | Pinned official architecture/assets; existing SignRep import is read-only |
| Official weights and shape compatibility | `reports/backbone_verification.json`, `reports/backbone_ablation_compatibility.json` | Strict 158-key ViT-B load; 16 frames at 224/256/384 and 32 frames at 224 have finite full/visible tokens |
| SSL annotation boundary | `tests/test_core.py`, real dataset/collate/model assertions | Structural whitelist and recursive semantic-key rejection |
| Source leakage audit | `reports/leakage_mvp_v2.json` | 95 SSL clips versus all 21,083 WLASL metadata records and 320 pilot evaluation clips |
| Exact content audit | `reports/leakage_content_mvp_v2.json` | Actual video SHA256 against all 320 gallery/val/test videos; no byte duplicates; perceptual audit not claimed |
| Frame/cache alignment | `SSLVideoDataset`, NPZ `frame_indices` and source/model hashes | Exact sampled-frame join; no guessed keypoint-clock resampling |
| Localization | v2 cache/repair report | Mean LH/RH/face/body validity 0.913 / 0.931 / 1.000 / 0.999 over the sampled 100 clips |
| Coordinate/mask QA | `reports/geometry_qa100_v2/`, `reports/augmentation_qa100_v2/` | 100 original/transformed displays, four mask kinds, forced flip/crop, landmarks, boxes, token grid, edge and span; 99 feasible and one explicit infeasible mask in the uncurated source |
| Structural filtering | `reports/data_preflight_mvp_v2.json` | 95/100 retained after four seeded augmented draws; all five exclusions recorded without semantic criteria |
| Teacher gradients and EMA | Behavioral tests and per-step trainer assertions | Both teacher encoder/projector remain frozen; student projector receives gradients; exact EMA on parameters/buffers |
| Actual pretrained mask leakage | `reports/pretrained_mask_invariance.json` | Replace all RGB pixels of hidden tubelets in ViT-B/384/BF16; visible contextual tokens remain bit-exact |
| Tests | `reports/pytest_final.log` | 41 passed, directly importing production modules |
| Final-source resume | `reports/final_source_resume.json` | Eight updates on real RGB, split at four; all 221 tensors match exactly, including periodic validation |
| Final-source DDP/BF16 | `reports/final_source_ddp_verified.json`, `reports/final_source_ddp_bf16.log` | Two CPU/Gloo ranks, four finite updates and two frozen validation events; correct best checkpoint |
| Empty local DDP queries | `reports/ddp_empty_query_loss.json` | Analytical loss 3.0/gradient 1.5 with one empty rank; globally empty targets fail on all ranks |
| GPU BF16 | research smoke and pretrained mask-invariance evidence | Actual official encoder and local predictor smoke previously trained; one-GPU host cannot verify multi-GPU NCCL |
| Periodic evaluation/selection | Final resume/DDP reports and `signrel/eval/frozen.py` | Frozen RGB evaluation; all RNG streams preserved; validation-only selection; best score/alarm state restored; DDP broadcasts failure |
| RGB-only inference | `encode_video`, `extract_features.py`, feature reports | No localization model or metadata input needed at inference |
| Scientific A/B/C/D | `reports/controlled_pilot/summary_{val,test}.json/.md` | All twelve 200-step runs audited: same data/init/batch/optimizer/masks, exact final steps and hashes, identical query/gallery rows and probe coverage |
| Frozen references | generic V-JEPA, random ViT-B and released SignRep val/test reports | Same local sampling; external SignRep differs in backbone, resolution and pretraining |
| Representation diagnostics | `reports/controlled_pilot/diagnostics/`, `reports/feature_diagnostics/` | Retrieval, linear/phonological/signer probes, matched-property minimal-pair confusion, embedding geometry and fixed random neighbors |
| Robustness | `reports/localization_robustness/`, `reports/occlusion/` | Target/mask jitter sensitivity and RGB occlusion retrieval; noisy-localization retraining is not claimed |
| Both-endpoint mask | `reports/strong_mask_preflight.json` | 380/380 augmented structural draws feasible; no training improvement claim |
| Extraction provenance repair | `reports/provenance_repairs/summary.json` | 36 metadata reports repaired from verified checkpoints; original backups retained; features/metrics/checkpoint bytes unchanged |
| Independent studies | `tests/test_study_isolation.py`, driver `feature_root_for` | Distinct derived feature paths and overwrite refusal for unrelated studies |
| Reproducibility | Each `runs/*/config.resolved.yaml`, `source.tar.gz`, environment/freeze, input/checkpoint hashes | Original pilot and final source versions remain distinguishable |

The initial v1 face localization rate was about 1%; the shoulder/head-crop repair
was completed before research training. Owned v1 files remain as diagnostic
artifacts. A failed two-GPU launch on the one-GPU host and the CPU oneDNN BF16
limitation are preserved; CPU reference-kernel BF16 and single-GPU checks passed.

The controlled suite's shell returned 143 after atomic `complete` status was
written. The driver and children were absent; independent audits validated all
twelve complete checkpoints, features and metric reports. Nothing was restarted.
This operational receipt is in `reports/post_suite_integration.json`.

## Source versions and completed result

- Research suite source: `0fe5684ff8e302971da9584c1f4f7bd5c11f073cb3e95bd1d44ed738bb14aaf2`.
- Final implementation: `8a29bcb631ac9f73b08ef3bc128790498d57a35df02e253d4a547191a0a71734`.
- Original random masks were hybrid bounded block/tube/random corruption. Final
  random masks preserve complete temporal tubes. All original A/B/C/D runs share
  the old implementation; future comparisons require a new study/source contract.
- Final code restores original upstream provenance for trained inference, supports
  zero local DDP queries, preserves validation RNG/best checkpoints and isolates
  study feature roots. Original source archives are retained for exact old resume.
- Validation R@1 means A/B/C/D: 0.0375 / 0.0417 / 0.0250 / 0.0458.
- Test R@1 means A/B/C/D: 0.0208 / 0.0292 / 0.0167 / 0.0208.
- Both audited decisions: `relation_benefit_not_supported_by_this_pilot`.

## Conditional scope beyond this MVP

1. Multi-scale operators have a backward/validity unit check and recipes, but no
   multi-scale research run or corpus scale-up is claimed. The single-scale gate
   did not pass, so specification §§19, 44–48 stop those experiments.
2. 27 recipes in `ABLATIONS.md` are unrun. The full serious-paper ablation matrix
   is experiment work for a broader study, not evidence already produced.
3. Sem-Lex and licensed NMFs-CSL videos are absent. Sem-Lex's form requires the
   researcher's identity and commitments; no form was submitted. The current legal
   phonological diagnostic joins ASL Citizen to ASL-LEX tabular metadata.
4. SHuBERT requires compatible multi-stream inputs, clock and gallery validation;
   no local SHuBERT baseline is claimed. VideoMAE and supervised fine-tuning are
   optional broader-study comparisons, not part of this representation-MVP result.
5. Explicit geometry targets, learned salience/discovery, hypergraphs and
   adversarial invariance remain deferred extensions. Core inference is RGB-only.
6. This small pilot does not establish full-benchmark performance, cross-language
   transfer, broad phonological disentanglement or signer invariance. Its negative
   result is reported with target-predictability and representation diagnostics.
