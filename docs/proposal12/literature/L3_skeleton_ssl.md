# L3 — Self-supervised skeleton representation learning (as of 2026-10-11)

Scope: skeleton SSL for action recognition and nearby human-motion pretraining, read in order to design a **skeleton JEPA for sign language**.

Evidence tags:
- **[verified-raw: …]**: I grepped the number from the raw paper text (arXiv PDF or ECVA PDF converted with PyMuPDF), from an author project page, or from code. Raw files are in `scratchpad/litreview/raw/L3/` (`txt/*.txt`, `sjepa_eccv.txt`, `slim_v3.txt`, `*_code/`).
- **[from-summary]**: the number comes from a secondary summary and is not in a primary source I could reach.
- **"as reported in X"**: the number is a baseline row copied from paper X's own comparison table, verified raw in X.

All NTU numbers are top-1 %. Column order is always NTU-60 xsub / xview, then NTU-120 xsub / xset. Unless a row is marked 3s (joint+bone+motion ensemble), it is joint-only.

---

## 0. Bottom line

1. **The two papers you were unsure about both exist.**
   - **Path-JEPA** is ECCV 2026. Crossref confirms LNCS vol. 17054, pp. 110–128, DOI 10.1007/978-3-032-37258-1_7.
   - **SLiM** (spelled "SLiM", for "Skeleton Less is More") is NeurIPS 2026 according to the authors' project page. It is arXiv 2603.10648.
   - Both are skeleton action-recognition SSL papers on NTU and PKU data. Neither is about sign language.
2. **The 2024–2026 trend is clear: stop predicting raw coordinates and predict teacher features instead** (S-JEPA → GFP → SLiM → Path-JEPA). Two caveats apply:
   - (a) Feature *MSE* without a predictor can collapse. SLiM measured an effective rank of 1.12.
   - (b) Coordinate targets can still win when they are well normalized and well weighted. AMR (CVPR 2026) reached 87.4 on xsub with "temporally normalized" joint coordinates.
3. **Structured masking is now consistently better than independent-token masking on skeletons:**
   - SLiM's semantic tube masking: +3.5 pts xsub over joint masking.
   - Path-JEPA's kinematic-subtree propagation: a "leak-free" mask.
   - SkeletonMAE (Yan): masking up to 3 body parts helps.
   - MAMP and MaskSem: motion-guided or CAM-guided sampling, +1.2 each.

   The reason is that independent joint masking at 90% is still solvable by spatial or temporal interpolation.
4. **S-JEPA (ECCV 2024) has no released code.** The official repo `Moo-osama/S-JEPA` is empty, so its internals below come from the camera-ready paper plus the MAMP code it inherits from.
   - **The camera-ready contains no 2D-input → 3D-target "depth trick".** Both branches see 3D joints. The "depth" claim only means the targets are features of the full 3D sequence. The 2D→3D lifting pretext belongs to MotionBERT.
5. **For sign language:**
   - Copy these: the latent-target JEPA with an EMA teacher, the asymmetric geometric "view" on the student, target masking at the teacher *output*, and motion-aware or part-structured masking.
   - Re-derive these for 2D, many-joint (body + 2×21 hands + face), noisy keypoints: the 3D rotation augmentations, 25-joint body-part groups, `T=120 / l=4` tokenization, and the NTU normalizations.

---

## 1. Must-cover paper 1 — S-JEPA (internals)

**Citation.** M. Abdelfattah, A. Alahi (EPFL). *S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition*. ECCV 2024, LNCS pp. 367–384, DOI 10.1007/978-3-031-73411-3_21.
- It is not on arXiv. arXiv API searches found nothing.
- Project page: sjepa.github.io. It has no code link.
- Code: the GitHub repo `Moo-osama/S-JEPA` (the first author's account) was created 2024-03-07 and **is empty** ("Git Repository is empty", GitHub API).
- Unofficial re-implementations exist (`trungphamss2/SJEPA_VSL`, `trungphamss2/MS-SJEPA`). These are student projects; one quotes incorrect "paper" numbers. Do not use them as references.

**Pipeline** (paper §3 and §4.2; all [verified-raw: sjepa_eccv.txt]):

| Component | S-JEPA choice |
|---|---|
| Input | 3D joints, NTU 25 joints. Random temporal trim ratio in [0.5, 1] at train time and 0.9 at test time, then bilinear resize to **T=120**. Same feeder as MAMP: NTU sequences are translated so that joint-2 (spine) of the first frame is the origin [verified-raw: MAMP code `seq_transformation.py`]. |
| Tokenization | Non-overlapping temporal **segments l=4** per joint. Each (joint, 4-frame segment) is linearly embedded, giving **N = 30×25 = 750 joint-segment tokens** (per-joint tokens, not per-frame tokens). Separate learnable spatial and temporal positional embeddings. |
| Encoders | "Vanilla transformer". View encoder and EMA target encoder both have **Le=8, Ce=256, 8 heads, FFN 1024**. |
| Predictor | Vanilla transformer, **Lp=5, Cp=256**. It takes the view-encoder outputs with learnable mask tokens inserted at the target indices. |
| Masking | **MAMP motion-aware masking, r=0.9**. Tokens with large frame-to-frame motion are sampled for masking more often. Visible tokens only go through the view encoder (MAE-style). |
| Targets | The **target encoder sees the full, unmasked, un-augmented 3D sequence**. Targets are taken from its **output** at the masked indices. Masking the target encoder's *input* instead drops accuracy to **78.1 vs 85.3** at 800 epochs (Table 6). |
| "View" asymmetry | Only the student branch gets geometric transforms: rotation about the skeleton's **own vertical axis** (estimated from pelvis and shoulders, Rodrigues formula), plus translation and spatial flip. Removing them gives 84.6 / 78.9 instead of 85.3 / 79.6 (Table 5). |
| Loss | Softmax over the 256 feature dimensions. Predictor temperature τp=0.1; target temperature τt=0.06 ("sharpening"). Cross-entropy H(p_t, p_p), DINO-style. **MSE instead of CE: 83.7 / 76.2** (Table 5). No projection or prototype head is described. |
| Collapse handling | **EMA target encoder** with λ cosine-scheduled from 0.9999 to 1.0. **Without EMA the model collapses: 1.6 / 0.83.** **Centering**: target minus an EMA batch center, β=0.9. Without centering: 84.5 / 79.4. Sweeping β from 0.5 to 1.0 gives xsub 84.3–85.3; β=0.9 is best. |
| Optimization | **1200 epochs**, effective batch 256. AdamW with betas (0.9, 0.95), wd 0.05. LR 1e-3 after a 20-epoch warmup, cosine down to 5e-4. 8×A100. |
| Evaluation encoder | The **target (EMA) encoder** is used. It beats the view encoder at every λ; at λ=0.9999 it is 85.3 vs 84.8. |
| Design ablation | Le 7–10 and Ce 64–512 all land at 84.0–85.3. Ce=64 drops to 84.0. Lp=5 is best. |

**Results** [verified-raw: sjepa_eccv.txt Tables 1–4]:
- Linear: **85.3 / 89.8 / 79.6 / 79.9**; PKU-I 92.2; PKU-II 53.5.
- Fine-tune: 93.1 / 97.6 / 90.3 / 91.3.
- Semi-supervised, NTU-60: 1% → 67.5 (xsub) / 69.1 (xview); 10% → 88.4 / 91.4.
- Transfer to PKU-II (pretrained on NTU-60 / NTU-120 / PKU-I): 71.4 / 74.2 / 70.9.

**Cost.** Measured by later papers on identical hardware, S-JEPA is ~4.5× slower to pretrain than MAMP:
- 90h57m vs 20h27m on one RTX 4090 [verified-raw: gfp.txt Table 1].
- 139.8 h vs 29.9 h on one L20 [verified-raw: amr.txt Table 1].

The extra cost comes from the full-token target encoder and the 1200 epochs.

**On the "depth / 2D-to-3D target trick".** The camera-ready never feeds 2D to either branch. The view branch gets rotated 3D, and the target branch gets clean 3D. "Depth" appears only as motivation: predicting 3D-encoder features is said to carry "depth and context" information. 2D→3D lifting is cited as **MotionBERT's** pretext (§1–2). No supplementary is listed on ECVA, and no code exists. Any 2D-student / 3D-teacher variant is therefore **not verifiable from S-JEPA**. It is, however, a natural design for sign language (see §6).

**What actually makes S-JEPA work** (from its own ablations):
1. EMA teacher. Without it the model collapses.
2. Targets taken from the teacher *output*, so each target is contextualized: +7.2 pts.
3. Softmax-CE with centering and sharpening instead of MSE: +1.6 / +3.4.
4. A geometric "view" on the student only: +0.7.

---

## 2. Must-cover paper 2 — Path-JEPA (ECCV 2026): found

**Citation.** Ashutosh Singh (Northeastern) and Ashish Singh (UMass Amherst). *Path-JEPA: Path Signature Based Predictive Learning for Skeleton Action Recognition*. ECCV 2026, LNCS vol. 17054, pp. 110–128, DOI 10.1007/978-3-032-37258-1_7.
- Crossref returns exactly this title, authors, pages and container "Computer Vision – ECCV 2026" [verified-raw: Crossref API].
- Project page and repo: `github.com/singh17ashu/Path-JEPA`. It contains only a README and index.html; **code is "coming soon"**.
- No arXiv version was found: the arXiv API has no title match, and the page's "arXiv" button has no link.
- The Springer page is behind a JS challenge, so I could not read the full paper.

**What it is.** Path-JEPA is about **path signatures from rough-path theory**, used as *prediction targets* for a skeleton JEPA. It is **not** a graph or kinematic-path model in the usual sense, although kinematic chains are one of its token families. The description below combines the author abstract [verified-raw: README / index.html] with the detailed ECCV2026 note on en.papernotes.org [from-summary; that page is LLM-generated from the paper text].
- **Path lifting.** Each joint trajectory is piecewise-linearly interpolated into a continuous path. A **lead-lag** transform adds a time-delayed copy (R^6 per 3D joint).
- **Signature features.** Truncated **log-signatures** at level 2 encode displacement (level 1) and Lévy signed area (level 2 cross terms). For d'=6 this gives 22 coefficients [from-summary].
- **Token families.**
  - **J-PSF**: joint paths.
  - **P-PSF**: pair or edge paths, i.e. the relative path X_v − X_u for bone-connected joints. These are translation-invariant.
  - **C-PSF**: **kinematic-chain paths** such as shoulder→elbow→wrist.

  Each token concatenates a sequence-global log-signature and a local-window log-signature, over multi-scale windows.
- **Signature-augmented masking.** Seed joints are sampled with probability proportional to motion (MAMP-like). Each seed is **expanded to its whole anatomical subtree**. The mask is then **propagated to every edge or chain token that touches a masked joint**, so relational tokens cannot leak the masked trajectory [author abstract verified-raw; mechanism details from-summary].
- **Architecture** [from-summary]:
  - Student: 6-layer transformer, width 512, 8 heads, on visible tokens.
  - Predictor: 4-layer, learnable queries cross-attending to student context.
  - Teacher: EMA (0.996→1), with centered targets (running mean, decay 0.9).
  - Loss: **cosine** distance. 300 pretraining epochs.
- **Linear results** [verified-raw: author project page; identical numbers in papernotes]: **87.6 / 92.9 / 81.5 / 81.8**; PKU-I 93.4; PKU-II 56.7. Compare S-JEPA at 85.3 / 89.8 / 79.6 / 79.9.
- **Fine-tune** [from-summary]: 94.4 / 98.2 / 90.8 / 91.9. **Semi-supervised 1% xsub** [from-summary]: 72.2.
- **Frame-rate robustness** on NTU-60 xsub, linear [verified-raw: project page]:

  | Method | 15 fps | 20 fps | 30 fps | 45 fps | 60 fps |
  |---|---|---|---|---|---|
  | Path-JEPA | 87.1 | 87.5 | 87.7 | 87.8 | 88.1 |
  | S-JEPA | 82.4 | 82.9 | 85.3 | 84.7 | 84.5 |
  | MAMP | 81.8 | 83.5 | 84.9 | 82.7 | 81.2 |

  Path-JEPA stays within a 1.0-point band. (Its 30 fps value of 87.7 differs slightly from the 87.6 headline.)
- **Tokenization ablation** on xsub [from-summary]:

  | Tokens used | xsub |
  |---|---|
  | Global joint only | 84.4 |
  | Local joint only | 83.3 |
  | Global + local joint | 85.8 |
  | + edge | 86.4 |
  | + chain | 87.6 |

**Relevance to sign.** Very high:
1. Signatures are reparameterization-invariant, which addresses signing-speed variation.
2. Edge and chain paths (wrist→MCP→PIP→DIP→tip) are exactly the finger kinematic chains.
3. Leak-free mask propagation is essential once bone or relative-position tokens are used.
4. The method works fine on 2D (d=2 gives a lead-lag d'=4).

Caveat: interval signatures smooth out timing, so very short handshape transitions may need small windows.

---

## 3. Must-cover paper 3 — SLiM (NeurIPS 2026): found ("SLim" = SLiM)

**Citation.** J. Do, Y. Chen, G. Youk, M. Kim (KAIST VICLab). *Less is More: Compact-Token Masked Feature Prediction for Skeleton Representation Learning* ("SLiM", Skeleton Less is More). arXiv **2603.10648** (v1 2026-03-11, v3 2026-08-03).
- **Venue NeurIPS 2026** comes from the authors' project page `kaist-viclab.github.io/SLiM_site` [verified-raw]. The arXiv comment gives no venue.
- Code: **github.com/KAIST-VICLab/SLiM**, a DINOv2-derived codebase. Weights: huggingface.co/JeonghyeokDo/SLiM.
- It is **not sign language**; NTU and PKU only. Its stated future work is "noisy 2D poses, missing joints, heterogeneous skeleton topologies".

**Method** [verified-raw: slim_v3.txt §3–4 and App. A–E; code files `slim/data/masking.py`, `slim/configs/default_config.yaml`]:
- **Encoder.** ViT with 8 layers, D=256, 8 heads, a CLS token, **temporal RoPE**, and learnable per-joint ("skeletal") positional embeddings.
- **Compact tokens.** Clip T=64, patch (P_T=8, P_J=1), giving **8×25 = 200 tokens**. This is 3.59 GFLOPs at inference vs 28.32 for 30×25-token MAEs (7.89×).
- **Teacher.** EMA teacher–student (momentum 0.994→1), with two objectives sharing one encoder:
  - **MFP**, masked feature prediction (iBOT-style): CE between the teacher prototype distribution on the clean global view and the student on the masked view, at masked tokens only. The student uses **in-place mask tokens**, so there is no separate predictor.
  - **GLCL**, global–local DINO on the CLS token: 2 global views (64 frames covering 50–100% of the clip) and 6 local views resampled at 32 / 16 / 8 frames *from the same interval*.
  - Heads: 3-layer MLP (2048 hidden, 256 bottleneck), **K=65,536 prototypes**, λ=1.
  - Extras present in the code config but not stressed in the paper: **Sinkhorn-Knopp** teacher centering, **KoLeo** weight 0.1, 4 register tokens, SwiGLU FFN, and `mask_sample_probability 0.5`. The default config uses 8 local crops of 16 frames, which differs from the paper's 6 multi-scale local views.
- **Semantic Tube Masking (STM).**
  - Five groups: trunk/head, left arm, right arm, left leg, right leg.
  - A proposal picks a group, a contiguous joint span, and a target area A∈[8, 0.5N]. Its temporal extent is h = round(A/w) ("constant volume": small parts are masked longer).
  - Target ratio ~U(0.5, 0.9); the realized ratio has mean 0.695.
  - Mask tokens replace content *before* the joint positional embedding is added, so joint identity is kept.
  - STM is also used as an augmentation on local views (p=0.5).
- **Skeleton-Aware Augmentations.** 360° yaw with ±30° tilt; a true mirror (x-negation plus left/right index swap); bone-length scaling U(0.85, 1.15) that keeps bone directions.
- **Training.** 150 epochs, batch 768, 4×A6000, AdamW base LR 2e-4.

**Results** [verified-raw]:
- **Linear**: 87.9 / 93.2 / 81.2 / 83.6; PKU-II 59.7.
- **Semi-supervised NTU-60**: 1% → 72.1 / 75.8; 10% → 88.8 / 91.9.
- **kNN retrieval**: 72.5 / 89.9.
- **Transfer to PKU-II** (linear): 72.7 from NTU-60, 75.3 from NTU-120.

**Ablations most relevant to us** (100-epoch models, NTU-60) [verified-raw: slim_v3.txt Tables 5, 7, 10, 11, 12]:
- **Structured vs random masking**: STM 87.7 / 92.7 vs conventional joint masking on the global view 84.2 / 90.4 (**+3.5 xsub**).
- **Mask ratio**: U(0.5–0.9) 87.7 vs fixed 90% 86.8 vs U(0–0.5) 84.5.
- **Target ablation** (matched encoder, masking and GLCL):

  | MFP target | Loss | Extra module | xsub / xview |
  |---|---|---|---|
  | Raw coordinates | MSE | 5-block decoder | 84.30 / 89.62 |
  | Teacher features | MSE | none | 84.53 / 89.32 (the MFP branch collapsed: effective rank 1.12) |
  | Teacher features | MSE | predictor head | 85.07 / 90.12 |
  | **Teacher prototype distribution** | **CE** | none | **87.66 / 92.68** |

  Conclusion: the latent target alone does not explain the gain. The *form* of the target (prototype assignment with an EMA teacher and centering) matters.
- **Token budget**: MAMP falls from 87.00 to 80.90 average when going from 30 to 8 temporal tokens; SkeletonMAE falls from 76.25 to 70.50. SLiM goes 90.90 → 90.55.
- **Missing and noisy joints, tested at inference only**:
  - Declared-missing joints routed to the mask token (8 of 25 removed for the whole clip): 87.23 vs 87.9 clean.
  - **Zero-filled unflagged joints**: 49.63.
  - Gaussian noise of 0.05 × bone length: 87.62.
  - Pretraining on degraded data helps only the matched corruption, and costs −2.26 on clean data.

**Relevance to sign.** STM is the closest existing recipe to "mask a whole hand for k frames". The mask-token route for missing joints maps directly onto RTMPose/MediaPipe confidence. The compact temporal patch (8 frames per token), however, is risky for fast fingerspelling.

---

## 4. Per-paper cards

### 4.1 Masked prediction (generative or latent)

**MAMP** — Mao et al., *Masked Motion Predictors are Strong 3D Action Representation Learners*, ICCV 2023, arXiv 2308.07092, code github.com/maoyunyao/MAMP. Code read: `model_mamp/transformer.py`, the NTU-60 config, `feeder`.
- **Input**: 3D joints, T=120 (crop 50–100% with a minimum of 64 frames, bilinear resize). Random rotation θ∈±0.3 rad about xyz. No bone or velocity input streams.
- **Tokens**: Conv2d with kernel (4,1), giving **30×25 joint-segment tokens** of dim 256.
- **Encoder**: 8 layers, 256 wide. **Decoder**: 5 layers, 256 wide (3 for NTU-60 in one config).
- **Target**: **frame-difference motion** (stride 1) of the 4-frame patch, i.e. 12-d. It is **per-patch normalized** (`norm_skes_loss`: subtract the mean and divide by the std over the patch's 12 values). MSE on masked tokens only.
- **Motion-aware masking**: token motion = mean |Δx| over the patch, then I/(max(I)·τ) with **τ=0.80**, softmax, and **Gumbel top-k** to choose **90%** of tokens to mask.
- **Training**: 400 epochs, LR 1e-3 → 5e-4, wd 0.05.
- **Results** [verified-raw: mamp.txt and S-JEPA Table 1]:
  - Linear 84.9 / 89.1 / 78.6 / 79.1; PKU-I 92.2; PKU-II 53.8.
  - Fine-tune 93.1 / 97.5 / 90.0 / 91.3.
  - Semi-supervised 1% 66.0 / 68.7; 10% 88.0 / 91.5.
- **Key ablations** [verified-raw: mamp.txt Tables 7–9, Fig. 4]. Input → target, NTU-60 / NTU-120 xsub:

  | Input → target | NTU-60 | NTU-120 |
  |---|---|---|
  | Joint → joint | 74.8 | 72.5 |
  | **Joint → motion** | **84.9** | **78.6** |
  | Motion → joint | 76.5 | 71.0 |
  | Motion → motion | 75.9 | 70.5 |

  - Random vs motion-aware masking: 83.7 / 77.3 vs 84.9 / 78.6.
  - Mask ratio sweep on xsub: 70% 76.6, 75% 78.7, 80% 78.7, 85% 82.2, **90% 84.9**, 95% 82.2.
  - Segment length l=4 is best (84.9 vs 84.6 at l=2 and 83.3 at l=6).

**SkeletonMAE (Wu et al.)** — *Spatial-Temporal Masked Autoencoders for Self-supervised Skeleton Action Recognition*, ICME 2023 workshop, arXiv 2209.02399.
- MAE that reconstructs masked joint coordinates with an **STTFormer** backbone.
- "Spatial-Temporal Masking" at both the joint level and the frame level, with separate ratios [verified-raw: skmae_wu.txt].
- **Results** as reported in S-JEPA Table 1 / 2 [verified-raw]:
  - Linear 74.8 / 77.7 / 72.5 / 73.5; PKU-II 36.1.
  - Fine-tune 86.6 / 92.9 / 76.8 / 79.1 with STTFormer, and 88.5 / 94.7 / 87.0 / 88.9 when re-run with a vanilla transformer.
- This is the canonical "joint → joint" baseline that MAMP beats by +10.1 linear.

**SkeletonMAE (Yan et al.)** — *Graph-based Masked Autoencoder for Skeleton Sequence Pre-training*, ICCV 2023, arXiv 2307.08476. Part of the "SSL/STRL" pipeline.
- A GIN-based graph MAE that **masks whole body parts** and reconstructs masked joint features and their edges from adjacent joints, using topology priors.
- Fine-tune-style results [verified-raw: skmae_yan.txt Table 3]: 92.8 / 96.5 / 84.8 / 85.7.
- **Body-part masking ablation** [verified-raw: Table 4c]. Mean accuracy as the number of masked parts goes 0 → 5:
  - GIN: 88.6, 89.5, 90.2, **91.2**, 90.3, 91.2.
  - GCN: 87.6, …, **90.6** at 3 parts.
  - Masking 3 parts is best or tied.
- This is the only pre-2024 direct evidence that more structured masking helps.

**MacDiff** — Wu et al., *Unified Skeleton Modeling with Masked Conditional Diffusion*, ECCV 2024, arXiv 2409.10473, code github.com/LehongWu/MacDiff.
- A transformer encoder on masked skeletons conditions a diffusion decoder. The paper reports an encoder of 8 layers and a decoder of 5 layers, with an MLP dimension of 1024.
- Linear [verified-raw: macdiff.txt]: 86.4 / 91.0 / 79.4 / 80.2; PKU-I 92.8. Transfer to PKU-II 72.2 / 73.4, as reported in SLiM Table 9.

**IGM** — *Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition*, ECCV 2024, arXiv 2410.20349.
- Generative model with an idempotence constraint; transformer encoder.
- Linear [verified-raw: igm.txt; the paper's column order is xview, xsub]: 86.2 / 91.2 / 80.0 / 81.4.
- (The task brief guessed ICLR 2025; it is in fact **ECCV 2024**.)

**GFP** — Sun et al., *Towards Efficient General Feature Prediction in Masked Skeleton Modeling*, ICCV 2025, arXiv 2509.03609. No code link in the paper.
- MAMP-style encoder: 8 layers, 256 wide, motion-aware masking, l=4, 400 epochs.
- The decoder predicts **hierarchical features**: local 5-, 10- and 30-frame pooled joint features plus a global feature. These come from a co-trained lightweight **target-generation MLP network (TGN)** whose input is **motion** (Δx).
- Collapse is prevented with **VICReg variance + covariance** terms rather than EMA. Weights λ=5, α=5, β=1.
- Pretraining is 6.2× faster than MAMP.
- **Results** [verified-raw: gfp.txt]:
  - Linear 85.9 / 92.0 / 79.1 / 80.3; PKU-II 56.2.
  - Semi-supervised 1% 71.8 / 72.9, as reported in SLiM Table 2.
  - kNN retrieval 70.9 / 87.1.
- **Ablations**:
  - Replacing learned targets with flattened low-level targets ({5,10,30,g}*): 86.8 vs **92.0** xview.
  - TGN input: motion 85.9 / 92.0 vs joints 85.0 / 90.9 vs masked joints 84.2 / 90.3.

**AMR** — Sun et al., *Exploring Adaptive Masked Reconstruction…*, CVPR 2026, arXiv 2606.11450. The code repo `AshenOne1005/AMR` holds only a README.
- A **decoupled cross-attention decoder** (learnable queries, no self-attention among mask tokens) reconstructs **larger patches**: 125 instead of 750.
- A **focal reconstruction** loss weights each window by its motion energy, w = 1/(1+k·e_t/e_s) with k=0.01 and multi-scale windows of 4 / 8 / 12 frames.
- Target: **temporally normalized raw joint coordinates**, following NAT (Gong et al., AAAI 2025). Loss on all patches. Motion-aware masking from MAMP.
- **Results** [verified-raw: amr.txt]:
  - Linear 87.4 / 92.3 / 81.1 / 81.9; PKU-II 60.3.
  - 3.7 h vs S-JEPA 139.8 h on an L20.
- **Ablations**:
  - Joint target 87.4 / 81.1 vs motion target 86.7 / 80.6. This reverses MAMP's finding once coordinates are normalized.
  - Cross-attention decoder vs adding self-attention: 87.4 vs 87.2 (no gain).
  - Also reported as a baseline: NAT 85.1 xsub.

**MaskSem** — *Semantic-Guided Masking for Learning 3D Hybrid High-Order Motion Representation*, IROS 2025, arXiv 2508.12948, code github.com/JayEason66/MaskSem (built on the MAMP codebase).
- **Grad-CAM (on relative motion)–guided masking**: the CAM replaces motion intensity in MAMP's Gumbel sampler. Code: `act_aware_random_masking`.
- Target: **velocity + acceleration**. Ratio 90%. Uses a MoCo queue of 65,536.
- Linear [verified-raw: masksem.txt Table I]: 85.9 / 90.8 / 77.5 / 79.3; PKU-I 91.2; PKU-II 55.8.
- **Ablation** on xview, linear [verified-raw: Table V]:

  | Masking | Target | xview |
  |---|---|---|
  | Motion-aware | velocity | 89.1 |
  | **Semantic** | velocity | **90.3** |
  | Semantic | acceleration only | 85.8 |
  | Semantic | velocity + acceleration | 90.8 |

**ASMa** — *Asymmetric Spatio-temporal Masking*, arXiv 2602.06251.
- Two ST-GCN encoders, each with an asymmetric mask pairing:
  - One masks high-degree joints and low-motion frames.
  - The other masks low-degree joints and high-motion frames.
- A Barlow-Twins-style feature alignment module, then distillation into a small model.
- 3-stream fine-tune [verified-raw: asma.txt Table 2]: 92.0 / 96.8 / 87.9 / 88.7.
- Linear with an ST-GCN backbone (Table 4) is low: 75.84 xsub. It is a GCN/contrastive-era design, weaker under linear probing.

### 4.2 Contrastive and other non-masked methods

The contrastive helper did not finish, so these cards are short. Linear numbers are as reported in the S-JEPA / SLiM / GFP comparison tables [verified-raw in those files] unless an own-paper source is named.

| Method (venue, arXiv) | Encoder / input | Core idea | Linear NTU-60 xs/xv, NTU-120 xs/xset |
|---|---|---|---|
| CrosSCLR (CVPR'21, 2104.14466) | ST-GCN, J+B+M | MoCo with cross-view (stream) positive mining | 3s 77.8 / 83.4 / 67.9 / 66.7 |
| AimCLR (AAAI'22, 2112.03590) | ST-GCN | Extreme augmentations, energy-based attention dropping, nearest-neighbour mining | 3s 78.9 / 83.8 / 68.2 / 68.8; J 74.3 / 79.7 / 63.4 / 63.4 |
| SkeleMixCLR (2207.03065, submitted to TMM) | ST-GCN | Spatio-temporal mixing of two skeletons (part/segment cut-mix) as positives | own table, J (SkeleMixCLR+): 80.7 / 85.5 [verified-raw: skelemix.txt] |
| CMD (ECCV'22, 2208.12448) | GRU / transformer | Cross-modal (J/B/M) mutual distillation of neighbour similarity | J 79.4 / 86.9 / 70.3 / 71.5 (S-JEPA table; SLiM lists xsub 79.8) |
| Hi-TRS (ECCV'22, 2207.09644) | Hierarchical transformer | Frame / clip / video-level hierarchical pretext | Fine-tune 3s 90.0 / 95.7 / 85.3 / 87.4 |
| HiCLR (AAAI'23, 2211.13466) | ST-GCN / transformer | Growing augmentations with hierarchical consistency | see paper (not re-extracted) |
| PSTL (AAAI'23, 2302.09018) | ST-GCN, Barlow Twins | **Central spatial masking** (high-centrality joints dropped more often) and **motion-attention temporal masking** (fast frames dropped) | 3s 79.1 / 83.8 / 69.2 / 70.3; PKU-II 3s 52.3 [verified-raw: pstl.txt] |
| ActCLR (CVPR'23, 2303.10904) | ST-GCN | Unsupervised "actionlet" (moving region) gets different augmentations from the static part | 3s 84.3 / 88.8 / 74.3 / 75.7; J 80.9 / 86.7 / 69.0 / 70.5 |
| HaLP (CVPR'23) | — | Hallucinated latent positives | J 79.7 / 86.8 / 71.1 / 72.2 |
| HiCo (AAAI'23) | GRU, hierarchical | Hierarchical contrast | J 81.1 / 88.6 / 72.8 / 74.1 |
| UmURL (ACM MM'23, 2311.03106) | Transformer, unified multi-modal | One-stream early fusion of J/B/M with intra/inter-modal consistency | NTU-120 73.5 / 74.3; retrieval 71.3 / 88.3 (GFP table) |
| PCM3 (ACM MM'23, 2308.03975) | GRU | Contrastive learning plus masked motion modeling with prompts | own: 83.9 / 90.4 / 76.5 / 77.5; PKU-II 51.5 [verified-raw: pcm3.txt] |
| SkeAttnCLR (2023) | — | Local and global attention contrast | 3s 82.0 / 86.5 / 77.1 / 80.0 |
| STARS (2407.10935) | Transformer (MAMP init) | MAMP pretraining then contrastive "self-supervised tuning" | own single-stream: 87.1 / 90.9 / 79.9 / 80.8; PKU-I 91.2; PKU-II 52.7 [verified-raw: stars.txt] |
| USDRL (AAAI'25, 2412.09220) | Transformer | Dense multi-grained feature decorrelation (no negatives) | 85.2 / 91.7 / 76.6 / 78.1; PKU-II 54.4 |
| HSP (CVPR'25) | — | 2D + 3D hierarchical semantic preservation | 80.7 / 88.0 / 71.0 / 73.2 |
| HeterSkel (CVPR'25, 2506.03481) | Transformer | Unifies heterogeneous skeleton formats | own "J" row 80.2 / 88.0 / 70.7 / 73.5; J+C 86.1 / 92.7 / 75.8 / 80.0 [verified-raw: heterskel.txt; meaning of C not re-checked] |
| MaskCLR (CVPR'24, same first author as S-JEPA) | Transformer | Attention-guided contrastive learning for robust representations | Fine-tune 93.5 / 97.5 / 90.5 / 91.9 (as reported in Path-JEPA note [from-summary]) |

"BIGRL" could not be identified, and no time was spent on it. Not found ≠ does not exist.

### 4.3 Human-motion pretraining (beyond NTU action SSL)

**MotionBERT** — Zhu et al., ICCV 2023, arXiv 2210.06551.
- **DSTformer**: dual-stream spatial/temporal MHSA with adaptive fusion; depth 5, 8 heads, width 512; T=243.
- Pretext: **2D→3D lifting** from **corrupted 2D**. Orthographically projected AMASS / H36M plus 2D from PoseTrack and InstaVariety. Corruption: **15% of joints zeroed**, frame-level masks, and mixed Gaussian/uniform noise.
- Loss: 3D position + velocity, plus a weighted 2D re-projection loss for 2D-only data.
- Curriculum: 30 epochs on 3D only, then 60 epochs on 3D+2D.
- **Results** [verified-raw: motionbert.txt]:
  - NTU-60 action uses **HRNet 2D keypoints**. Fine-tune 93.0 / 97.2; scratch 87.7 / 94.1.
  - Frozen backbone plus trained head: xview 87.3. NTU-120 one-shot: 67.4.
- **Pretraining ablation** on xsub fine-tune:

  | Setting | xsub |
  |---|---|
  | No pretraining | 87.7 |
  | Pretrained | 89.4 |
  | + noise | 90.7 |
  | + mask | 91.9 |
  | + in-the-wild 2D | 93.0 |

  **Corrupting the 2D input is worth +2.5.** This is the most sign-relevant evidence on noisy 2D keypoints.

**PoseBERT** — Baradel et al., TPAMI 2022, arXiv 2208.10211, code github.com/naver/posebert.
- A BERT-style transformer trained only on MoCap (SMPL / AMASS), **with a MANO hand variant**.
- **Frame-level masking** (default 12.5%), with a mask token or random pose, plus Gaussian noise.
- Used for refinement, infilling and forecasting on top of any frame-wise estimator [verified-raw: posebert.txt].
- Relevant as a hand-pose temporal prior; it is not a representation learner for recognition.

**Sapiens** — Khirodkar et al., ECCV 2024 (oral), arXiv 2408.12569.
- Image MAE on **Humans-300M** at 1024 px, with mask ratios explored from 75% to 95%.
- Its pose head predicts **308 keypoints including 243 face keypoints** [verified-raw: sapiens.txt].
- This is an *input-side* asset for sign (dense face and hand keypoints), not skeleton SSL.

**Human-JEPA** — Wei, Sun, Zhao (Oulu), arXiv 2608.21160.
- Built from V-JEPA 2.1 on **RGB video**: anchored forecasting with past→future masks.
- **Not a skeleton model.** I list it only because "skeleton JEPA" searches surface it.

**Other JEPA-named hits** (WiFi-JEPA 2607.11064, T-JEPA, Point-JEPA, ACT-JEPA) are not skeleton-sequence SSL.

**Nearest other recent items**:
- S2I (2603.05963): skeleton→image encoding to use vision-pretrained models (submitted to IJCV).
- "Beyond Binary Contrast" (2604.17914, CVPR 2026 per papernotes).
- PRISM (CVPR 2026, per papernotes; not read).

---

## 5. Comparison table (joint-only linear evaluation unless noted)

| Method | Venue | Tokens / encoder | Masking | Target | Collapse guard | NTU60 xs | xv | NTU120 xs | xset | Tag |
|---|---|---|---|---|---|---|---|---|---|---|
| SkeletonMAE (Wu) | ICMEW'23 | STTFormer | joint + frame random | coords | n/a | 74.8 | 77.7 | 72.5 | 73.5 | raw (S-JEPA T1) |
| MAMP | ICCV'23 | 30×25 joint-seg, ViT-8/256 | motion-aware 90% | Δx (patch-norm) | n/a | 84.9 | 89.1 | 78.6 | 79.1 | raw |
| S-JEPA | ECCV'24 | 30×25, ViT-8/256 + pred-5 | motion-aware 90% | EMA-teacher output, CE + center | EMA + centering | 85.3 | 89.8 | 79.6 | 79.9 | raw |
| MacDiff | ECCV'24 | ViT enc-8 / diff-dec-5 | masked | diffusion denoising | n/a | 86.4 | 91.0 | 79.4 | 80.2 | raw |
| IGM | ECCV'24 | transformer | — | idempotent generative | — | 86.2 | 91.2 | 80.0 | 81.4 | raw |
| USDRL | AAAI'25 | transformer | — | decorrelation (no masking) | VICReg-like | 85.2 | 91.7 | 76.6 | 78.1 | raw (SLiM T1) |
| MaskSem | IROS'25 | MAMP ViT | Grad-CAM-guided 90% | velocity + acceleration | n/a | 85.9 | 90.8 | 77.5 | 79.3 | raw |
| GFP | ICCV'25 | MAMP ViT + TGN | motion-aware | hierarchical learned features | VICReg | 85.9 | 92.0 | 79.1 | 80.3 | raw |
| AMR | CVPR'26 | ViT + cross-attn decoder | motion-aware | normalized coords, focal-weighted | n/a | 87.4 | 92.3 | 81.1 | 81.9 | raw |
| STARS | arXiv'24 | MAMP + contrastive tuning | motion-aware | Δx then contrastive | — | 87.1 | 90.9 | 79.9 | 80.8 | raw |
| **SLiM** | NeurIPS'26 | **8×25**, ViT-8/256, RoPE | **semantic tube 50–90%** | iBOT prototypes + DINO CLS | EMA + SK centering + KoLeo | **87.9** | **93.2** | 81.2 | **83.6** | raw |
| **Path-JEPA** | ECCV'26 | signature tokens (J/P/C × global/local), ViT-6/512 | motion seed → subtree, propagated | EMA latent of signature tokens, cosine | EMA + centering | 87.6 | 92.9 | **81.5** | 81.8 | raw (author page) |
| ActCLR (3s) | CVPR'23 | ST-GCN | actionlet augmentation | InfoNCE | queue | 84.3 | 88.8 | 74.3 | 75.7 | raw (S-JEPA T1) |
| MotionBERT (FT, 2D HRNet) | ICCV'23 | DSTformer | 15% joints + frames + noise | 3D lift | n/a | FT 93.0 | FT 97.2 | — | — | raw |

---

## 6. Structured vs random masking, and latent vs coordinate targets: the evidence in one place

**Structured or informed masking beats independent random masking.** All of the following are on NTU-60 xsub linear unless noted.

| Evidence | Structured / informed | Random / independent | Gain |
|---|---|---|---|
| SLiM, semantic tube masking (100-epoch ablation) | 87.7 | 84.2 (joint masking) | +3.5 |
| MAMP, motion-aware sampling | 84.9 (NTU-120: 78.6) | 83.7 (NTU-120: 77.3) | +1.2 / +1.3 |
| MaskSem, Grad-CAM-guided (xview) | 90.3 | 89.1 (motion-aware) | +1.2 |
| SkeletonMAE-Yan, GIN, masked body parts | 91.2 at 3 parts | 88.6 at 0 parts | +2.6 |

- Path-JEPA's subtree-expanded, propagated mask is presented as necessary to stop leakage through relational tokens. No random-mask number is public.
- ASMa argues that motion-only or degree-only masking is *biased* and pairs complementary masks. This warns against "always mask the moving joints".

**Mask ratio.** 90% is the fixed-ratio optimum for MAMP and MaskSem. SLiM's sampled ratio U(0.5, 0.9) beats a fixed 90% (+0.9).

**Latent vs coordinate targets.**

| Comparison | Latent / better | Coordinate / worse | Gap |
|---|---|---|---|
| S-JEPA vs MAMP (same backbone and masking) | 85.3 | 84.9 | +0.4 linear; about +0.8–1.0 on transfer |
| GFP: learned hierarchical vs low-level | 92.0 (xview) | 86.8 | +5.2 |
| SLiM matched target study: prototype CE vs raw coordinates | 87.66 | 84.30 | +3.4 |

- In SLiM's study, plain feature MSE (84.53) ≈ coordinates and collapses without a predictor.
- **Counter-evidence**: AMR's normalized-coordinate MAE reaches 87.4 xsub, above S-JEPA, GFP and MaskSem.
- **MAMP**: the choice of coordinate target matters a lot. Motion beats joints by +10.1.

**Net**: latent targets help when they are (i) taken from a contextual teacher *output*, (ii) protected against collapse (EMA + centering + CE / prototypes, or VICReg), and (iii) combined with a hard, structured mask. Otherwise a well-normalized, motion-weighted reconstruction is as good or better, and cheaper.

---

## 7. Lessons for a sign-language skeleton JEPA

**What transfers**
1. **JEPA core.**
   - EMA target encoder on the clean full sequence.
   - Targets from the teacher's *output* at masked positions (S-JEPA: +7.2 over input masking).
   - Collapse guards: centering plus sharpened softmax-CE or prototype heads, which are safer than raw MSE (SLiM: feature MSE collapsed to rank ≈1). Alternatively VICReg on the targets (GFP).
   - Evaluate the EMA encoder.
2. **Structured masking in the sign anatomy.** Define groups as: each hand (21 joints, with sub-chains per finger), each arm (shoulder–elbow–wrist), face regions (mouth, eyes/brows), and torso. Then:
   - Use SLiM-style constant-volume tubes, so small groups (one finger) are masked for longer than large groups (a whole arm).
   - Expand masks along the kinematic subtree, as Path-JEPA does. Masking the wrist should mask the hand.
   - **Propagate masks to any derived token** (bone, relative-to-wrist, edge or chain features) to avoid leaks.
   - Sample the ratio from roughly U(0.5, 0.9) instead of a fixed 90%.
   - Mix motion-informed seeds with uniform seeds, as ASMa warns. Motion-only masking over-targets moving hands and never asks the model to infer the static non-dominant hand.
3. **Motion-aware information.** Motion as a target (MAMP), as the TGN input (GFP), or as loss weights (AMR) all help. For sign, compute motion per hand in a **wrist-relative frame**, so finger articulation is not swamped by arm translation.
4. **Missing-joint protocol.** SLiM shows that routing declared-missing joints to the mask token costs ≈0.7 points, while zero-fill costs up to 38. Map RTMPose/MediaPipe confidence (or MediaPipe's missing-hand flag) to mask tokens at both train and test time. **Never zero-fill.**
5. **Corruption pretraining for noisy 2D.** MotionBERT gets +2.5 from noise and mask corruption of 2D inputs. A 2D-student / clean-or-smoothed-teacher asymmetry is a reasonable JEPA analogue of its 2D→3D lifting.
   - This asymmetry is the plausible reading of the "S-JEPA 2D→3D depth" idea, but **it is not in S-JEPA**.
   - Options for the teacher input: lifted 3D hands (e.g. from a hand-mesh or lifting model) while the student sees raw 2D, or a temporally filtered teacher input while the student sees jittered 2D.
6. **Temporal-rate robustness.** Path-JEPA's ±0.5-point band from 15 to 60 fps addresses signer speed and dataset fps variation directly. Two cheap substitutes: log-signature level-2 features of hand and finger chains as *targets*, or multi-rate local views (SLiM GLCL at 32/16/8 frames).

**What does not transfer**
1. **3D-specific augmentations.** S-JEPA's rotation about the body vertical axis and SLiM's 360° yaw assume metric 3D. On 2D keypoints, use 2D affine and scale, a small shear (a camera-tilt proxy), and **mirroring only with care**: left/right handedness is lexical in some sign languages, but dominant-hand normalization is common.
2. **NTU token geometry.**
   - 25 joints with 4-frame (MAMP/S-JEPA) or 8-frame (SLiM) temporal patches. With about 130 sign joints (body + 2×21 + a face subset), per-joint tokens explode: 130 × 30 = 3,900 tokens.
   - Group tokens per hand or finger (Path-JEPA-style chain tokens).
   - 8-frame patches at 25–30 fps (~0.3 s) are too coarse for fingerspelling and handshape transitions. SLiM itself warns that subtle high-frequency actions may need denser temporal tokens.
3. **NTU preprocessing** (origin at spine of the first frame, two-person padding) is irrelevant for sign. Use per-frame shoulder-centered and shoulder-width-scaled normalization, plus wrist-relative hand coordinates.
4. **Coarse body-part groups** (5 groups, with legs) do not exist in sign framing. The informative structure is finger-level, and the face carries non-manual grammar.
5. **NTU linear-probe gains of 1–3 points** may not predict sign gains. NTU actions are coarse full-body motions with strong body-part shortcuts. Sign classes differ mainly in handshape, location and small movements, and signer variation (hand size, speed, style) matters more. Validate on signer-independent splits, and use retrieval or kNN probes (SLiM, GFP) in addition to linear.
6. **Compute.** S-JEPA's 1200 epochs with a full-token teacher (~4.5× MAMP cost) is a poor fit for a shared single GPU. SLiM-style compact tokens or AMR/GFP-style light targets cut cost by 6–8×.

**A concrete starting recipe**:
- **Student**: MAMP/S-JEPA ViT (8×256).
- **Tokens**: per-hand and per-chain tokens with 2–4-frame temporal patches.
- **Masking**: tubes over sign-anatomy groups with subtree expansion and leak propagation. Ratio ~U(0.5, 0.9). Missing keypoints go through the mask token.
- **Teacher**: EMA on the clean (filtered) sequence; targets at the teacher output.
- **Loss**: centered, sharpened CE or a prototype head. Optionally an auxiliary global CLS DINO loss over multi-rate local views.
- **Evaluation**: frozen-EMA linear, kNN, and signer-independent splits.
