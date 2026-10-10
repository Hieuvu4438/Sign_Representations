# SignRel-JEPA implementation and pilot result — 2026-10-06

The isolated engineering MVP is implemented and verified. **The fixed pilot does
not establish that relation prediction improves the representation.** Relation
targets improve some phonological probe scores, but retrieval gains are not
consistent against both generic and articulator controls. Multi-scale research
training and large-corpus scale-up were therefore not started.

## Controlled comparison

All twelve runs used the same official V-JEPA 2.1 distilled ViT-B initialization,
95 audited How2Sign clips, RGB `16×384×384`, optimizer/schedule, effective batch 4,
200 optimizer updates and random corruption. Seeds were 42, 43 and 44. The recipe,
40 glosses and evaluation videos were fixed before looking at features or scores.
Gallery/query sizes were 160/80 for each official signer-independent val/test
split. Semantic annotations were used only in the separate evaluation pipeline.

Values below are means over three **training seeds**. Full sample SD and paired
seed deltas are in [validation summary](reports/controlled_pilot/summary_val.md)
and [held-out test summary](reports/controlled_pilot/summary_test.md).

| Variant | Val R@1 | Test R@1 | Val phonology macro-F1 | Test phonology macro-F1 |
|---|---:|---:|---:|---:|
| A: generic JEPA | 3.75% | 2.08% | 0.2134 | 0.2093 |
| B: A + articulator | 4.17% | 2.92% | 0.2329 | 0.1820 |
| C: A + relation | 2.50% | 1.67% | 0.2667 | 0.2329 |
| D: A + articulator + relation | 4.58% | 2.08% | 0.2324 | 0.1983 |

C's validation phonology gain over A is positive in all three seeds, with mean
delta +0.0532, but its mean R@1 delta is −0.0125. D's mean validation R@1 delta
over B is only +0.0042, while phonology is essentially unchanged (−0.0005).
Neither relation variant has a reproducible two-metric benefit over both controls.
Test results do not reverse that conclusion. Lower signer-probe accuracy cannot
be called invariance when linguistic quality is not maintained.

## Frozen reference models

| Reference | Val R@1 | Test R@1 | Val phonology macro-F1 | Test phonology macro-F1 |
|---|---:|---:|---:|---:|
| Random ViT-B, seed 42 | 2.50% | 0.00% | 0.1477 | 0.1907 |
| Generic pretrained V-JEPA 2.1 | 5.00% | 6.25% | 0.3907 | 0.3143 |
| Released SignRep | 87.50% | 58.75% | 0.8381 | 0.7068 |

The random/pretrained reference uses the same RGB temporal sampling and local
pooling API. SignRep is an external released model with a different backbone,
resolution and pretraining history; its checkpoint's complete pretraining source
inventory was not independently reconstructed. These values are local pilot
references, not official full-benchmark reproductions or equal-data causal
comparisons. The same center-window evaluation recipe is documented for all.

## What the diagnostics show

- Frozen validation mean pairwise cosine rises from 0.9780 at generic pretrained
  initialization to about 0.9983 / 0.9990 / 0.9986 for A/B/C seed 42. Effective
  centered embedding rank falls from 20.2 to 13.1 / 10.7 / 8.5. This is reduced
  feature diversity, not a verdict based on training loss alone.
- For C seed 42, per-edge centered energy accounts for only 0.39–0.66% of
  normalized EMA relation-target energy. A leave-one-clip-out constant predictor
  has L1 0.046–0.060. Replacing the second endpoint with another clip still gives
  raw cosine 0.998–0.999. Centered cosine is separately reported. A shared target
  mean can make the objective easy; low relation loss is insufficient evidence.
- Three-property minimal-pair retrieval covers 66/80 validation queries. Matching
  all five selected properties covers only 28/80 queries across 14 lexical pairs.
  On the strict subset, seed-42 A/B/C/D R@1 is 0.393 / 0.357 / 0.500 / 0.464.
  This small one-seed diagnostic is encouraging for selected pairs but does not
  override the complete paired retrieval result or control unselected factors.
- Box-jitter diagnostics on 95 clips show feasible masks at all tested levels.
  At translation sigma 0.4 times box size, mean IoU is 0.348 and relation-target
  cosine is still 0.9966. Target sensitivity is measured here; no robustness from
  noisy-localization retraining is claimed.
- RGB occlusion uses a fixed clean gallery, identical query rows and a seeded tube
  occluder. C seed-42 validation R@1 varies between 1.25% and 3.75% over 0–40%
  occlusion; this does not demonstrate strong transfer robustness. Pose is never
  needed by this inference path.

Artifacts: [target predictability](reports/target_diagnostics/C_relation_seed42_teacher.json),
[feature geometry/neighbors](reports/feature_diagnostics/C_relation_seed42_val.json),
[localization curve](reports/localization_robustness/C_relation_seed42.png),
[RGB occlusion](reports/occlusion/C_relation_seed42_val.json),
[validation tradeoff plot](reports/controlled_pilot/pareto_val.png).

Specification §41 suggests testing stricter relation-breaking masks, early/local
teacher features and deterministic versus learned relation functions. Both-endpoint
masking is implemented and passed all 380 seeded augmented structural draws. The
paired stronger-mask recipes are provided, but no result from them is claimed.
Extra losses or a larger backbone were not used to disguise this negative pilot.

## Engineering and reproducibility

41 tests pass. Actual final-source real-video training/resume matches all 221 model
tensors exactly after eight updates split at four, with periodic frozen validation.
Two-rank CPU/Gloo BF16 training, validation and best-checkpoint selection are finite.
An analytical distributed regression checks correct loss/gradient when one rank
has zero valid queries. Replacing every hidden RGB pixel in the actual pretrained
ViT-B adapter leaves visible contextual tokens bit-exact. The host has one GPU,
so multi-GPU NCCL is not claimed as exercised.

All twelve research checkpoints, their final step/config/source/data contracts,
feature/metric byte hashes, identical sample rows and probe coverage were audited.
36 extraction reports were repaired to retain original initialization provenance
when loading trained weights; original reports are backed up. Feature arrays,
metrics and training checkpoint bytes were unchanged. The driver shell returned
143 after the suite's atomic `complete` status; the process was absent and every
completed artifact was verified independently. No experiment was restarted.

The pilot source fingerprint is `0fe5684ff8e302971da9584c1f4f7bd5c11f073cb3e95bd1d44ed738bb14aaf2`.
The final implementation fingerprint is `8a29bcb631ac9f73b08ef3bc128790498d57a35df02e253d4a547191a0a71734`.
Every research run retains its original source archive and resolved config. Final
source enforces complete random temporal tubes; the original pilot used a bounded
hybrid block/tube/random corruption identically across A/B/C/D. Treat future runs
with the new source as a new experiment. New run/report roots derive distinct
feature roots and refuse overwriting unrelated studies.

This is a 95-clip, 40-gloss engineering/scientific MVP. It is not a full benchmark,
significance test, SOTA result, universal-language representation or evidence that
the hypothesis cannot work at another validated regime. Sem-Lex/NMFs-CSL access,
SHuBERT's compatible stream protocol and serious-paper ablations remain separate
requirements for a broader study. See [DATASETS.md](DATASETS.md),
[ABLATIONS.md](ABLATIONS.md) and [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).
