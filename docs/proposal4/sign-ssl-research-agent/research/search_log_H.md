# Search log H — prediction targets / priors for JEPA-style sign-language SSL

Date: 2026-10-07. Engine: WebSearch (US). Abstract verification: `curl https://export.arxiv.org/api/query?id_list=...` (Atom XML; title, authors, date, abstract parsed locally). WebFetch was NOT used (known to hallucinate numbers in this project). No numeric result was copied from search-engine summaries into the review.

Read levels: ABSTRACT_READ = abstract text fetched from export.arxiv.org and read; DISCOVERED = snippet / search summary only.
Output: `target_priors_review.md` (same folder). Source IDs (S..) refer to that file.

| # | Query | Kept sources | Notes |
| --- | --- | --- | --- |
| H1 | masked motion encoding self-supervised video representation MME arXiv | S19 MME 2210.06096; S17 MotionMAE 2210.04154 | motion-trajectory target (MME); frame-difference target (MotionMAE) |
| H2 | MGMAE motion guided masking video masked autoencoder optical flow | S20 MGMAE 2308.10794 | flow used for *masking*, not as target |
| H3 | Masked video distillation MVD teacher features video transformer arXiv 2212 | S22 MVD 2212.04500; S29 SALT 2509.24317 (first seen here) | teacher-feature targets; frozen-teacher JEPA |
| H4 | Unmasked Teacher UMT CLIP video foundation model masked arXiv | S23 UMT 2303.16058; S24 SMILE 2504.00527 (first seen) | CLIP-feature alignment of unmasked tokens |
| H5 | MaskFeat masked feature prediction HOG video arXiv | S16 MaskFeat 2112.09133; S21 TrackMAE 2603.27268 (first seen) | HOG target |
| H6 | V-JEPA 2.1 dense features self-supervised video arXiv 2026 | S28 V-JEPA 2.1 2603.14482; S30 MotionJEPA 2609.23881; S31 FWM-HW-LD 2605.17165 | dense/context loss + deep supervision |
| H7 | SignRep self-supervised sign language representation sign priors Hiera arXiv | S01 SignRep 2503.08529; S07 SSVT-ISLR 2309.02450 (first seen); Spatial-Temporal Consistency 2406.10501 (listed in S10 notes) | anchor |
| H8 | SHuBERT self-supervised sign language multi-stream cluster prediction hands face body | S02 SHuBERT 2411.16765; S03 SignDino 2609.06296; handshape zero-shot 2609.18772 (S14 notes) | DINOv2 hand/face features → k-means cluster targets (snippet-level detail) |
| H9 | SignDino self-supervised sign language DINOv3 articulator crops | S03 SignDino; S04 SignMusketeers 2406.06907 | frozen DINOv3 crop features + temporal DINO/iBOT |
| H10 | MASA motion-aware masked autoencoder sign language representation | S10 MASA (arXiv id 2405.20666 resolved via arXiv API title search) | pose-input motion-residual reconstruction |
| H11 | SignBERT+ hand-model-aware self-supervised pretraining sign language | S08 SignBERT 2110.05382 / SignBERT+ 2305.04868 | MANO hand-model-aware decoder (pose input) |
| H12 | BEST BERT pre-training sign language recognition tokenized pose triplet units | S09 BEST 2302.05075 | discrete pseudo-labels from pose triplets |
| H13 | JEPA sign language self-supervised joint embedding predictive | S27 I-JEPA 2301.08243 | **no sign-language JEPA found** |
| H14 | skeleton JEPA S-JEPA self-supervised skeleton action recognition latent prediction | S33 S-JEPA (ECCV 2024, no arXiv id found; arXiv API title search empty); GFP 2509.03609; SLiM 2603.10648 | skeleton latent / feature-prediction targets |
| H15 | SSVP-SLT self-supervised video pretraining sign language translation MAE privacy | S06 SSVP-SLT 2402.09611 | pixel MAE on blurred sign video |
| H16 | Uni-Sign unified sign language understanding generative pretraining CSL-News | S11 Uni-Sign 2501.15187 | text-supervised generative pretraining (not SSL target) |
| H17 | SignMAE masked autoencoder sign language video pretraining | S05 SignMAE 2605.02094; S13 MS-MAE (IJCNLP 2025, DISCOVERED) | segmentation-driven masking / hand-only reconstruction |
| H18 | MotionMAE motion-aware masked autoencoder frame difference target video | S17 MotionMAE; S18 MAM2 2210.05234 | RGB-difference target |
| H19 | MotionJEPA preventing temporal feature collapse visual changes latent space | S30 (confirmed) | JEPA slow-feature bias; difference-image embedding regulariser |
| H20 | Rethinking JEPA compute-efficient video SSL frozen teachers SALT | S29 (confirmed) | frozen pixel-MAE teacher → latent student |
| H21 | SMILE spatial and motion semantics masked video learning CLIP synthetic motion | S24 (confirmed); S36 HuC-VideoMAE 2610.08433 (first seen) | CLIP-feature target + synthetic motion |
| H22 | TrackMAE video representation learning track mask and predict point tracks target | S21 (confirmed) | point-track displacement target (CoTracker3) |
| H23 | Improving Sign Language Understanding with a Multi-Stream Masked Autoencoder Trained on ASL Videos | S13 MS-MAE (ACL Anthology 2025.ijcnlp-long.66) | skeleton-based; no arXiv id found |
| H24 | Factorized Latent Dynamics for Video JEPA auxiliary objectives empirical study | S31 (confirmed) | 18 auxiliary JEPA objectives; trade-offs |
| H25 | non-manual features mouthing facial expression self-supervised sign language representation pretraining | S37 SignMouth/SignClip 2509.10266 | **no SSL work with non-manual geometry as target found**; mouthing work is supervised SLT |
| H26 | sign language MANO hand mesh pretraining 3D hand pose prior representation learning HaMeR | S38 "Seeing in 2D, Thinking in 3D" (ICCVW 2025, DISCOVERED); DexAvatar 2512.21054 (seen, not kept: reconstruction, not representation) | HaMeR → fingerspelling distillation |
| H27 | HuC-VideoMAE human-centric video masked autoencoding synthetic data | S36 (confirmed); S15 VideoMAE 2203.12602 | keypoint-driven masking |
| H28 | MultiMAE multi-modal multi-task masked autoencoders depth semantic segmentation targets | S34 MultiMAE 2204.01678 | pseudo-label depth/segmentation targets |
| H29 | human motion pose JEPA self-supervised latent prediction 2025 2026 arXiv keypoints | S32 Human-JEPA 2608.21160 | built from V-JEPA 2.1; frozen-init anchored targets |
| H30 | SignMusketeers multi-stream hands face body self-supervised DINOv2 sign language translation | S04 (confirmed) | DINOv2-S SSL on hand/face crops |
| H31 | data2vec contextualized target representations EMA teacher masked prediction | S26 data2vec 2202.03555 | EMA contextual latents |
| H32 | BEiT v2 vector-quantized visual tokenizer CLIP teacher; MILAN masked image pretraining language assisted | S25 BEiT v2 2208.06366, MILAN 2208.06049 (EVA 2211.07636 added by id) | CLIP/caption-supervised feature targets |
| H33 | masked video modeling DINOv2 features as reconstruction target distillation video ViT | none new (MVD already kept) | no video-MAE paper with DINOv2 targets specifically identified in this query |
| H34 | phonological features handshape location movement prediction auxiliary pretraining sign language recognition ASL-LEX | S14 Kezar et al. 2509.04745; 2609.18772; Kezar et al. EACL 2023 "Improving Sign Recognition with Phonology" (DISCOVERED) | phonology as supervised auxiliary / semi-supervision |
| H35 | optical flow self-supervised pretraining sign language recognition RGB motion stream | S12 M3-SLR (IEEE doc 11132300, DISCOVERED) | flow used for MaskFeat masking in sign |
| H36 | Human-JEPA human-centric vision model anchored forecasting pose re-identification | S32 (confirmed) | — |
| H37 | M3-SLR MaxFlow MaskFeat multi-view sign language arXiv | S12 (no arXiv id found) | — |
| H38 | Self-Supervised Video Transformers for Isolated Sign Language Recognition MaskFeat comparison 2309.02450 | S07 (confirmed) | direct sign evidence for HOG target |
| H39 | masked video modeling predict human keypoint heatmaps pose target action recognition pretraining | none kept (STH-MAE, skeleton heatmap MAE, PMC, seen only) | no RGB video MAE/JEPA with keypoint-heatmap *target* found in this query |
| H40 | Sapiens foundation for human vision models body-part segmentation arXiv | S35 Sapiens 2408.12569, Sapiens2 2604.21681 | target generator, not SSL-target paper |
| H41 | sign language video latent prediction V-JEPA pretraining isolated sign recognition 2026 | none | **no sign-language V-JEPA pretraining found** |
| H42 | sign language masked video modeling distillation from hand crop features teacher self-supervised 2025 2026 | none new; CVPR 2026 "Learning Effective Sign Features without Text for Gloss-free SLT" (SignDINO, Gan et al.) seen via papernotes (DISCOVERED, not read) | name collision with S03 SignDino — distinct papers |

Direct arXiv-ID lookups (no web query; ID from earlier results or recall, then verified by the arXiv API title): 2203.12602 VideoMAE, 2404.08471 V-JEPA, 2506.09985 V-JEPA 2, 2211.07636 EVA, 2406.10501 Spatial-Temporal Consistency SSL, 2509.03609 GFP, 2603.10648 SLiM, 2509.10266 SignMouth.

## Scope statement for novelty wording

"Not found" in the review means: not found in searches H1–H42 on 2026-10-07 (WebSearch top results + arXiv abstracts listed above). It is not a claim of absence from the literature. Full-text reading was not done for any source.
