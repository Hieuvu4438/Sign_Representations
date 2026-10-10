# Search log B — General video / sequence SSL mechanisms transplantable to sign representations

Date: **2026-10-05**. Agent: literature worker B (family B). Scope: 2019–2026 core papers on (a) long temporal structure over short-clip encoders, (b) masked video modeling with motion targets, (c) negative-free regularisation & collapse diagnostics, (d) noisy-teacher / heteroscedastic regression, (e) background/appearance debiasing, (f) post-hoc temporal alignment for retrieval. Budget: ~15 notes, depth over breadth.

Tools: WebSearch (US index), WebFetch (arXiv HTML), direct `curl` of arXiv PDFs + PyMuPDF text extraction (scratchpad only). Metadata (title/authors/version/comments) verified from arXiv abs pages for every kept paper.

## Queries

| # | Date | Where | Query | Result used |
| --- | --- | --- | --- | --- |
| Q1 | 2026-10-05 | WebSearch | `V-JEPA Revisiting Feature Prediction for Learning Visual Representations from Video arXiv` | 2404.08471 → P201 |
| Q2 | 2026-10-05 | WebSearch | `V-JEPA 2 self-supervised video models enable understanding prediction planning arXiv 2506.09985` | 2506.09985 → P202; DISCOVERED: V-JEPA 2.1 (2603.14482), "Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers" (2509.24317), Human-JEPA (2608.21160), SiamJEPA (2607.04044) |
| Q3 | 2026-10-05 | WebSearch | `Masked Motion Encoding for self-supervised video representation learning MME arXiv` | 2210.06096 → P207; DISCOVERED: MotionMAE (2210.04154), "A Large-Scale Analysis on Contextual Self-Supervised Video Representation Learning" (2504.06153), "Image Classifiers are Efficient Self-Supervised Video Representation Learners" (2609.40347) |
| Q4 | 2026-10-05 | WebSearch | `Temporal Context Aggregation for Video Retrieval with Contrastive Learning arXiv` | 2008.01334 → P206 |
| Q5 | 2026-10-05 | WebSearch | `Motion-aware Contrastive Video Representation Learning via Foreground-background Merging FAME arXiv` | 2109.15130 → P213 |
| Q6 | 2026-10-05 | WebSearch | `sign language retrieval dynamic time warping frame embeddings isolated sign dictionary lookup` | sign priors (see below): SignSeek 2609.03695, SignMatch 2609.01886, IEEE 8806601 |
| Q7 | 2026-10-05 | WebSearch | `sign language pose estimation confidence weighted loss self-supervised pretraining keypoint uncertainty` | SignBERT+ 2305.04868 (loss locator checked) |
| Q8 | 2026-10-05 | WebSearch | `sign language masked video modeling motion target hand-aware masking self-supervised 2024 2025` | MASA (TCSVT 2024), SignMAE (2605.02094), 2309.02450 |
| Q9 | 2026-10-05 | WebSearch | `Drop-DTW Aligning Common Signal Between Sequences While Dropping Outliers NeurIPS 2021 proceedings` | venue verification P215 |
| Q10 | 2026-10-05 | WebSearch | `"Revisiting Feature Prediction for Learning Visual Representations from Video" TMLR OpenReview` | venue verification P201 (TMLR 2024 via dblp/mlanthology listing) |
| D1 | 2026-10-05 | arXiv direct (known IDs, metadata verified on abs page) | DPC 1909.04656, TCLR 2101.07974, LSTCL 2106.09212, VideoMAE 2203.12602, MGMAE 2308.10794, VICReg 2105.04906, RankMe 2210.02885, Dim. collapse 2110.09348, β-NLL 2203.09168, BE 2009.05769, TCC 1904.07846, Drop-DTW 2108.11996, Hiera 2306.00989, MaskFeat 2112.09133 | see Kept / Excluded |

Note: one direct-ID guess for FAME (2204.03659) returned an unrelated physics paper; discarded and corrected via Q5. Recorded as a near-miss, not used.

## Kept (paper notes written)

| Note | Paper | Family | Read level | Why kept |
| --- | --- | --- | --- | --- |
| P201 | V-JEPA (2404.08471, TMLR 2024) | a | FULL_READ | Latent prediction; Table 3 shows avg-pool vs attentive probe gap — directly relevant to SignRep weighted-mean retrieval |
| P202 | V-JEPA 2 (2506.09985) | a | FULL_READ (pretraining/understanding parts only) | 64-frame cooldown schedule = template for cheap continued long-clip pretraining |
| P203 | DPC (1909.04656) | a | FULL_READ | Video instantiation of CPC; shortcut analysis |
| P204 | TCLR (2101.07974, CVIU) | a | FULL_READ | Temporal distinctness vs multi-clip aggregation + NN retrieval; global-local loss template |
| P205 | LSTCL (2106.09212, CVPR 2022) | a | FULL_READ | "Long-short" consistency for video transformers |
| P206 | TCA (2008.01334, WACV 2021) | a/f | FULL_READ | Aggregation over frozen frame features; chamfer vs cosine evidence |
| P207 | MME (2210.06096, CVPR 2023) | b | FULL_READ | Motion-trajectory targets + shuffle diagnostic; maps to pose-derived dense targets |
| P208 | Hiera (2306.00989, ICML 2023) | b | FULL_READ | SignRep backbone; pos-embed/attention facts determine transplant cost |
| P209 | VICReg (2105.04906, ICLR 2022) | c | FULL_READ | SignRep regulariser; var/cov ablation |
| P210 | RankMe (2210.02885, ICML 2023) | c | FULL_READ | Label-free collapse/checkpoint diagnostic |
| P211 | β-NLL (2203.09168, ICLR 2022) | d | FULL_READ | Failure mode of heteroscedastic NLL; fix for uncertainty-weighted pose distillation |
| P212 | BE (2009.05769, CVPR 2021) | e | FULL_READ | Background debiasing; frozen-retrieval gain much smaller than fine-tune |
| P213 | FAME (2109.15130, CVPR 2022) | e | FULL_READ | Foreground/background merge; inter-video background matters |
| P214 | TCC (1904.07846, CVPR 2019) | f | FULL_READ | Cycle-consistency alignment; needs same-class pairs (= gloss supervision for sign) |
| P215 | Drop-DTW (2108.11996, NeurIPS 2021) | f | FULL_READ (§4.2 not read) | Training-free outlier-robust sequence matching for retrieval |

## Excluded / deferred (with reasons)

| Paper | ID | Read level | Reason |
| --- | --- | --- | --- |
| VideoMAE | 2203.12602 | DISCOVERED (PDF downloaded, not read) | Baseline covered via MME/Hiera/V-JEPA comparisons; note budget |
| VideoMAE v2 | — | not searched | Blind spot; scale-focused (dual masking) — low transplant value expected (INFERENCE) |
| MGMAE | 2308.10794 | DISCOVERED (PDF downloaded, not read) | Motion-guided masking via optical flow; overlaps with "hand-guided masking" idea — read next if masking becomes active hypothesis |
| MotionMAE | 2210.04154 | DISCOVERED | Same reason as MGMAE |
| MaskFeat | 2112.09133 | DISCOVERED (PDF downloaded, not read) | HOG target evidence available via Hiera Table 3c |
| Understanding Dimensional Collapse in Contrastive SSL (Jing et al.) | 2110.09348 | DISCOVERED (PDF downloaded, not read) | Contrastive-specific; RankMe chosen as actionable diagnostic. Read if collapse found |
| CPC (van den Oord et al. 2018) | — | not fetched | Pre-2019; DPC is the video instantiation |
| Barlow Twins | — | not fetched | VICReg covers var/cov; Barlow differences noted in VICReg §5.3/Table 5 |
| DINOv3 / KoLeo feature spreading | — | not searched | **Blind spot**; should be covered in a follow-up query |
| Arrow-of-time / temporal order pretext | — | not searched | **Blind spot**; time reversal is semantically risky for sign (rule 10) — lower priority |
| Kendall & Gal 2017 (heteroscedastic) | — | cited in β-NLL, not read | Pre-2019; β-NLL analyses its failure mode |
| V-JEPA 2.1 | 2603.14482 | DISCOVERED | Dense features; possibly relevant to per-segment features — follow-up |
| Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers | 2509.24317 | DISCOVERED | **High-priority follow-up**: frozen-teacher JEPA ≈ continued pretraining from existing checkpoint |
| Human-JEPA, SiamJEPA, BiJEPA, JEPA-WAM | 2608.21160, 2607.04044, 2603.00049, 2609.20277 | DISCOVERED | Not core to the six mechanisms; triage later |
| Graph Based Temporal Aggregation for Video Retrieval | 2011.02426 | DISCOVERED | Alternative to TCA; not needed |
| A Large-Scale Analysis on Contextual SSL Video Repr. | 2504.06153 | DISCOVERED | Survey-like; triage later |

## Sign-language priors found incidentally (for "closest prior" mapping; not family-B notes)

| Work | ID / source | Read level | Relevance |
| --- | --- | --- | --- |
| SignSeek: Learning Transferable Representations for Sign Dictionary Retrieval (Asasi, Sincan, Bowden; 2026-09-03) | arXiv 2609.03695 | ABSTRACT_READ | Contrastive same-gloss alignment (gloss supervision) + saliency-guided articulator masking + latent masked prediction; cross-corpus dictionary retrieval SOTA claim. Very close competitor for frozen retrieval; must be read by literature worker for sign family |
| SignMatch: Matching Dictionary Signs to Continuous Sign Language Video (Wong, Jang, Momeni, Varol, Zisserman; 2026-09-01) | arXiv 2609.01886 | ABSTRACT_READ | Prototype-structured embedding from annotated continuous video; ASL-Citizen dictionary retrieval. Same first author as SignRep |
| Self-Supervised Video Transformers for Isolated Sign Language Recognition | arXiv 2309.02450 | ABSTRACT_READ | MaskFeat best on WLASL2000 (79.02% top-1); hierarchical ViTs important for phonological probing |
| SignBERT+ | arXiv 2305.04868 | Loss section checked (§3.2: L_rec = Σ 1(s≥ε)·s·‖·‖ confidence-thresholded & weighted) | Closest sign prior for confidence-weighted pose regression |
| MASA: Motion-Aware Masked Autoencoder With Semantic Alignment for SLR | TCSVT 2024, DOI page dl.acm.org/doi/abs/10.1109/TCSVT.2024.3409728 | DISCOVERED (snippet only) | Closest sign prior for motion-aware MAE |
| SignMAE: Segmentation-Driven Self-Supervised Learning for SLR | arXiv 2605.02094 | DISCOVERED | Hand/arm decoder masks |
| Dynamic Time Warping Based Sign Retrieval | IEEE Xplore 8806601 | DISCOVERED (page failed to load; metadata unverified) | Closest sign prior for DTW retrieval |

## Coverage statement

Coverage is provisional: one search pass + targeted direct fetches; **no forward/backward snowballing round completed**. Not found within scope Q1–Q10/D1 (2026-10-05): a sign-language paper doing long-clip *SSL cooldown* from a 16-frame checkpoint, or β-NLL-style heteroscedastic weighting of pose targets for RGB sign encoders — this is "not found in this search", not "not done". Next queries: `KoLeo DINOv3 feature spreading`, `frozen teacher JEPA continued pretraining`, `sign language long clip pretraining 64 frames self-supervised`, `heteroscedastic pose distillation video`, `sign language background augmentation signer segmentation`, forward citations of TCLR/TCA/Drop-DTW restricted to sign.
