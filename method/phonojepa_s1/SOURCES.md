# Source and design provenance

S1 is a new adaptation, not a reproduction of one published method. The complete research audit, cached PDFs and pinned official repositories are in [proposal12 research bundle](../../docs/proposal12/research_update_20261011/LITERATURE_AND_CODE.md).

| Source | What S1 inherits | What it does not inherit |
|---|---|---|
| Local `method/phonojepa/pj` | Tested keypoint/cache/window/channel pipeline, MLP/Transformer/RoPE/predictor/EMA implementation, centered CE, extraction format | Seven-channel constants, unsafe fallback anchors, unclosed physical duplicates, old stochastic resume behavior |
| [S-JEPA, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) | Masked latent prediction, EMA teacher, centered CE | Original 3D input, 8-layer encoder, 90% masking or reported performance |
| [SignRep, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.html) | Geometry/interaction supervision as useful sign priors | RGB Hiera input, pretrained weights, 3D teacher geometry, an asserted torso-only causal benefit |
| [MASA](https://arxiv.org/abs/2405.20666v1) | Explicit motion residual targets and attention to pose quality | Its semantic alignment/text supervision and full motion-aware masking recipe |
| [STC official source](https://github.com/sakura2233565548/Self-Supervised-Representation-Learning-with-Spatial-Temporal-Consistency-for-SLR/blob/9dc803188b69a93d0ab5123842b8ff62189bead9/moco/GCN_Transformer.py) | Reviewed separate hand/body encoding and temporal fusion | GCN tokenizer, contrastive architecture or checkpoint weights |
| [SignSeek](https://arxiv.org/abs/2609.03695v1) | Reviewed multipart body/face/hand coverage | Same-gloss supervised positives or claiming its MAP teacher is EMA |
| [Path-JEPA](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/5207.pdf) | Dependency-aware masking principle is compatible with our raw-support audit | No path/log-signatures in S1; that belongs to S2 |

`UPSTREAM.json` records SHA256 for every **local** file copied/adapted. External official source snapshots were read, not copied into S1: STC commit `9dc803188b69a93d0ab5123842b8ff62189bead9` (Embed shares hand GCN and has a separate body GCN); SignRep commit `06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2` (prior heads). No new third-party source code or weights are vendored here. Existing inherited losses contain optional baseline utilities that S1 does not activate; see their original docstrings for attribution. This document grants no new license to external papers/repositories.

S1-specific choices: five-point torso MLP features, always-visible shoulder policy, torso tube lengths, nose/hip body targets, five 2D relation vectors/distances, auxiliary λ=.1, training-only scale fitting and matched 768-dimensional gallery-fitted PCA. These are experimentally testable engineering choices, not claims that papers validated this exact recipe.
