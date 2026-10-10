# Prior-art comparison (primary sources checked 2026-10-06)

This is a bounded comparison, not an exhaustive novelty search. No claim of first multi-stream SSL, first sign JEPA, or first temporal relation loss is justified.

| Work | Targets and supervision | Stream fusion / temporal relation | Evaluation and compute context |
|---|---|---|---|
| [SHuBERT, ACL 2025](https://aclanthology.org/2025.acl-long.1397/) | Masked prediction of offline clustered hand, face and pose streams; unlabeled pretraining | Contextual multi-stream hidden-unit prediction | Translation, isolated recognition, fingerspelling; approximately 1,000 h ASL, much larger than this pilot |
| [SignRep, ICCV 2025](https://arxiv.org/html/2503.08529v1) | RGB masked autoencoding with pose priors, feature and adversarial style objectives | RGB backbone; pose required during pretraining, not inference | Recognition, frozen dictionary retrieval, translation; pretrained external weights/data are not a causal relation ablation |
| [SignSeek, September 2026 preprint](https://arxiv.org/html/2609.03695v1) | Gloss-supervised GAC/MAC contrast plus stop-gradient articulator prediction MAP | Pose articulator graph encoders fused through Conformer; saliency masks | Dictionary retrieval; 266K labeled multilingual clips; supervision differs from our label-free SSL |
| [SignDino (Hu et al.), September 2026 preprint](https://arxiv.org/html/2609.06296v1) | Frozen DINOv3 embeddings, temporal EMA DINO/iBOT/KoLeo; second-stage temporal Gram matching | Independently trained hand/face temporal encoders, downstream fusion; within-stream frame Gram structure | Translation, recognition, phonology, fingerspelling; 400K + 50K SSL steps; target type and compute differ materially |
| [Spatial-temporal consistency, TIP 2024](https://arxiv.org/html/2406.10501v1) | Skeleton joint/motion contrastive learning; momentum targets and KL consistency | Hand/trunk consistency and bidirectional joint/motion transfer | Isolated recognition; already establishes prior work on cross-part coordination |
| SIGNREL proposal7 (this implementation) | Original frozen DINOv2 ViT-S/14 continuous hand states; body coordinate prediction; sparse clean cross-hand cosine relations, no lexical labels | Joint time-major LH/RH/body attention during SSL; signed LH(t)–RH(t+lag) target from hand decoders, masked endpoint required | Custom frozen K=1 40-gloss ASL Citizen pilot; 47 SSL train clips; <=1,000 updates/run; inference still needs perception and DINO |

The candidate distinction is whether sparse signed cross-hand relations improve a joint contextual encoder beyond state reconstruction under matched controls. This is a hypothesis requiring A4–A1, A4–A2/A3 and shuffled A6 tests. Architectural differences alone do not establish novelty or usefulness. Cross-hand DINO appearance cosine is a visual proxy; it is not annotated phonological coordination.

[S01 DINOv2](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md) supplies the fixed image teacher. [SALT](https://arxiv.org/html/2509.24317v1) motivates frozen-teacher prediction, not an exact recipe reproduced here. [I-JEPA](https://arxiv.org/abs/2301.08243) motivates masked latent prediction. This implementation does not reproduce those complete training pipelines.

The verified public code for [phonological perception](https://github.com/kayoyin/sign-phonology) does not establish availability of its curated natural minimal-pair annotations; real minimal-pair results are BLOCKED_EXTERNAL. Synthetic evaluator tests cannot support a phonology claim.

## Bounded current-literature refresh

Two searches on2026-10-06 for self-supervised sign-language cross-hand temporal relations and frozen-teacher coordination recovered the main works above plus:

- [SignMAE, ICPR 2026](https://arxiv.org/abs/2605.02094): primary abstract describes segmentation-adaptive body/hand masking and reconstruction for recognition. Abstract read; complete method/recipe not reviewed or reproduced here. Another reason not to claim novelty from articulator masking alone.
- [BiView-Touch, September 2026 preprint](https://arxiv.org/abs/2609.23352): primary abstract describes masked target-hand latent completion from contralateral tactile context, EMA targets and temporal/layout counterfactuals. It is a tactile robotics setting, not sign video, but cross-hand predictive dependence already has close conceptual prior art outside signing. Abstract read; no checkpoint/implementation used.

The scan is bounded and cannot establish absence of competing work. Secondary aggregators were discovery only; above claims link primary papers. A publication novelty claim still needs a full method-level review and updated search close to submission.
