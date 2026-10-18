# B — Skeleton SSL and masking strategies: an ablation-centred, verified update (2026-10-14)

**Purpose.** This file supports the experiment design for **skeleton-based SSL on COCO-WholeBody 2-D keypoints from ASL Citizen**. The encoder is frozen and evaluated by dictionary retrieval (kNN) and a linear probe.

**Relation to earlier reviews.** It *extends and corrects* `docs/proposal12/literature/L3_skeleton_ssl.md` and `L2_masking_structured.md`. Papers that those files cover well get only a one-line pointer here.

**Evidence tags.**
- **[V]**: I grepped the number myself in this session from raw paper text. The table or figure is named next to it.
- **[V\*]**: the number comes from *another* paper's comparison table, which I grepped. The original paper was not read.
- **[S]**: the fact comes only from a search summary or an LLM-generated note.
- **[K]**: background knowledge, not checked here.

**Raw text location.** All paths are under `scratchpad/lit/raw/`:
- `new/*.txt`: PDFs fetched today and converted with PyMuPDF.
- `prev/L3txt/*.txt`, `prev/L2txt/*.txt`, `prev/sjepa_eccv.txt`, `prev/slim_v3.txt`, `prev/pathjepa_*`: raw files from the 2026-10-11 session, copied here and re-grepped today.
- `q/*.xml`: arXiv API query results.

**Conventions.** NTU numbers are top-1 % in the order NTU-60 xsub / xview, then NTU-120 xsub / xset. Inputs are joint-only unless "3s" is stated. "Linear" means a frozen-encoder linear probe; "kNN" means 1-NN retrieval accuracy with cosine similarity.

---

## 1. Bottom-line design rules, with evidence strength

Strength scale:
- **Strong**: at least two independent papers agree, or one large matched ablation shows an effect above 3 points.
- **Moderate**: one matched ablation in the 1–3 point range, or several unmatched hints.
- **Weak / gap**: indirect evidence or none.

Nearly all evidence comes from **3-D Kinect NTU data with 25 joints**, so every rule carries a transfer risk for 2-D sign data (see §5).

### R1. Predict contextual teacher outputs, with a collapse-safe loss — Moderate–Strong

Evidence:
- **S-JEPA, target taken at the teacher output vs the teacher input:** 85.3 vs 78.1, i.e. **+7.2** (800-epoch ablation) [V: sjepa_eccv Table 6].
- **SLiM matched target study** (same encoder, masking and GLCL; 100 epochs) [V: slim_v3 Table 10]:

  | Target | Loss | Module | NTU-60 xsub |
  |---|---|---|---|
  | Coordinates | MSE | 5-block decoder | 84.30 |
  | Teacher features | MSE | none | 84.53 (collapsed: effective rank 1.12) |
  | Teacher features | MSE | predictor | 85.07 |
  | **Teacher prototype distribution** | **CE** | none | **87.66 ± 0.11** |

- **S-JEPA, CE vs MSE:** 85.3 vs 83.7 [V: Table 5].

**Caveat:** a well-normalised coordinate target can still compete. AMR with NAT-normalised joints reaches 87.4 [V: amr Table 1/8].

**Rule:** do not run a latent-MSE JEPA without a predictor plus a centering or prototype guard. Plan an *iBOT/DINO-style CE head on EMA-teacher outputs* as the default, and keep a normalised-coordinate MAE as a strong baseline.

### R2. Never use raw absolute joints as a reconstruction target — Strong; motion vs normalised coordinates — Weak

- MAMP, joint→joint vs joint→motion: 74.8 vs **84.9** (+10.1) [V: mamp Table 7].
- AMR reverses the ranking once coordinates are temporally normalised: joint 87.4 vs motion 86.7 [V: amr Table 8].
- The +10.1 is therefore mostly "un-normalised coordinates are a bad target", not "motion is magic".

### R3. Make masking hard to interpolate, and keep it stochastic — Strong

**Skeleton2vec** (200-epoch ablations) [V: 2401.00921 Table 5 and Fig. 4]:
- Random 79.4 / 85.1 vs tube 83.0 / 87.2, i.e. **+3.6 / +2.1**. Motion-aware tube: 83.5 / 87.7.
- Tube length is measured in 3-frame tokens (xsub):

  | α (tokens) | 1 | 2 | 3 | **5** | 10 | 15 | 30 (whole clip) |
  |---|---|---|---|---|---|---|---|
  | xsub | 79.4 | 81.7 | 82.6 | **83.0** | 81.7 | 80.2 | 78.2 |

  The optimum is about 15 frames (~0.5 s at 30 fps). **Whole-clip tubes are worse than random.**

**HA-CM:** removing the Gumbel randomness from motion-aware masking (deterministic top-k) collapses MAMP from **84.9 to 66.7** [V: 2409.17951 Table VI].

**Skeleton2vec, strength of motion bias β:** 0 → 83.0, **0.1 → 83.5**, 0.2 → 82.1, 0.3 → 79.5 [V: Table 6a]. Over-biasing towards moving joints costs 4 points.

### R4. The value of structured masking depends on the target — Strong (this corrects L2/L3)

Structured masking helps when the loss sits on **masked tokens**:

| Paper | Structured | Random / baseline | Gain |
|---|---|---|---|
| Skeleton2vec (tube vs random) | 83.0 | 79.4 | +3.6 |
| SLiM (semantic tube vs joint masking on the global view) | 87.7 | 84.2 | +3.5 [V: Table 5a] |
| PSTL (CSM+MATM vs random) | 77.3 | 75.6 | +1.7 [V: pstl Table 3] |
| MAMP (motion-aware vs random) | — | — | +1.2 [V] |

It *hurts* in MacDiff, whose diffusion decoder is conditioned on a **pooled global representation** and denoises the whole sequence [V: macdiff Table 7]:

| Masking | random | spatial-temporal | motion-aware | temporal | tube |
|---|---|---|---|---|---|
| NTU-60 xsub | **86.4** | 85.3 | 85.5 | 84.1 | 83.1 |

**Rule:** treat mask structure as a hypothesis to test per target type, not as a default win.

### R5. High mask ratios for per-joint tokens; sample the ratio — Strong

| Paper | Ratio sweep (xsub linear) | Source |
|---|---|---|
| MAMP | 70% 76.6, 85% 82.2, **90% 84.9**, 95% 82.2 | [V] |
| Skeleton2vec | 80% 83.1, **90% 83.5**, 95% **77.1** | [V: Table 6b] |
| MacDiff | 0% 79.3, 50% 82.7, 80% 83.8, **90% 86.4**, 95% 83.6 | [V: Table 7] |
| SLiM | U(0.5, 0.9) **87.7** vs fixed 90% 86.8 vs U(0, 0.5) 84.5 | [V: Table 5a] |

There is a cliff at 95% in two of the four papers.

### R6. Anti-collapse — Strong for EMA necessity, Moderate for regulariser alternatives

- **S-JEPA without EMA collapses to 1.6 / 0.83** [V: Table 5].
- The **EMA momentum λ is a high-variance knob**. Linear xsub, target encoder vs view encoder [V: Fig. 3b]:

  | λ | 0.99 | 0.999 | **0.9999** | 0.99999 |
  |---|---|---|---|---|
  | Target encoder | 80.0 | 83.6 | **85.3** | 80.9 |
  | View encoder | 79.2 | 81.9 | 84.8 | 80.8 |

  The EMA encoder is always the better one to evaluate.
- **Regulariser routes without EMA:**
  - GFP uses VICReg variance + covariance, reaching 85.9 in 400 epochs [V].
  - USDRL replaces InfoNCE with VICReg-style decorrelation: linear 74.8 → 84.2 and retrieval 63.8 → 73.7 (STTR backbone) [V: 2508.12586 Table 10].
- **Gap:** no skeleton paper tests SIGReg or a frozen-teacher target.

### R7. For retrieval, pure masked reconstruction is the wrong objective; add a global or instance term — Strong

- MAMP: linear 84.9 but **kNN 62.0 / 70.0** [V: 2406.02978 Table 3; also V\* in GFP, SLiM, IGM].
- **STARS:** 20 epochs of NNCLR tuning on top of MAMP lift kNN (K=1) from 63.1 / 80.3 to **79.9 / 88.6** (+16.8 xsub). The linear gain is only about +2.2 (87.1 vs 84.9) [V: stars Tables 1/2/11; L3].
- Methods with a global term recover kNN:

  | Method | kNN xsub / xview | Source |
  |---|---|---|
  | GFP | 70.9 / 87.1 | [V] |
  | SLiM | 72.5 / 89.9 | [V] |
  | IGM | 69.3 / 72.6 | [V] |
  | USDRL (DSTE) | 75.0 / 89.3 | [V: Table 4] |
  | PCM3 | 73.7 / 88.8 | [V] |

### R8. Combine local masked prediction with global multi-view or multi-rate invariance — Moderate

SLiM, NTU-60 xsub [V: Table 4]:

| Objective | xsub |
|---|---|
| MFP (masked feature prediction) alone | 85.3 |
| Plain CL alone | 73.6 |
| GLCL alone | 75.3 |
| MFP + CL | 86.2 |
| **MFP + GLCL** | **87.7** |

PCM3++ shows the same direction: using masked or predicted views as positives raises xview 88.2 → 90.5 [V: 2406.02978 Table 5].

### R9. Augmentations are as decisive as masking in invariance branches; use weak→strong asymmetry — Strong

| Paper | Ablation | Result | Source |
|---|---|---|---|
| BYOL-skeleton | conservative-only vs aggressive-only augmentation | **51.1 vs 79.6** | [V: 2202.02232 Table 2] |
| BYOL-skeleton | full (asymmetric pipelines + multi-view sampling) | 86.8 | [V: Table 2] |
| SkeletonCLR | no augmentation vs shear + crop | 33.3 vs 68.3 | [V: crossclr Table 3] |
| HiCLR | adding naive random masking as a contrastive augmentation | 72.9 → **56.7** | [V: hiclr Table 3] |
| HiCLR | hierarchical weak→strong arrangement | 77.6 | [V: Table 3] |
| AimCLR | extreme augmentations applied directly | 75.0 → 71.3 | [V: aimclr Table 1] |
| SLiM | removing rotation | −6.1 | [V: Table 5b] |
| SLiM | bone-length-preserving vs naive scaling | +1.3 | [V: Table 5b] |
| BYOL-skeleton | removing the Savitzky–Golay low-pass filter augmentation | −2.4 | [V: Table 2] |
| S-JEPA | student-only geometric view | +0.7 | [V: Table 5] |

### R10. Missing keypoints go through the mask token, never zero-fill or parent-fill — Strong

- SLiM, 8 of 25 joints missing for the whole clip, NTU-60 xsub [V: slim_v3 Table 11]:

  | Representation of missing joints | Accuracy |
  |---|---|
  | Declared mask token | 87.23 |
  | Parent-joint fill | 66.51 |
  | Zero fill | 49.63 |

- Masking used as an augmentation buys occlusion robustness. With 10 joints masked at test time, PSTL keeps **73.8** while AimCLR drops to **54.3** [V: pstl Table 4].

### R11. Compact temporal tokens only work with latent targets; low-value joints can be pruned — Moderate

- **Temporal tokens 30 → 8:** MAMP average accuracy 87.00 → 80.90; SLiM 90.90 → 90.55 [V: slim_v3 Table 7].
- **Joint pruning (HA-CM):** removing torso joints (25 → 18) gives 85.1 vs 84.9 at 0.6× the training time; 16 joints drops to 81.5. "Removing head joints causes a sharp accuracy drop" [V: Table V].

### R12. Statistics: expect ±0.5–1 point of uncontrolled noise; no skeleton paper reports pretraining-seed variance — Strong as a caution

- SLiM states explicitly that its main results come from a **single pretraining run**. Its ± values come only from linear probes (±0.1) or corruption draws [V: slim_v3 §"Scope of reported variability"].
- BYOL-skeleton's 1%-label fine-tuning has SD of 0.6–1.5 across label subsets [V: Table 5].
- Rankings between *target types* stayed stable over the schedule in GFP's curve. Hyperparameter rankings (EMA momentum), however, flip with schedule length (§3f).

---

## 2. Paper cards

### 2.1 New cards (absent from L2/L3, or present only as a table row)

**Skeleton2vec** — Xu, Huang, Wang, Hu, Deng. arXiv **2401.00921**. The arXiv comment says "Submitted to CVPR 2024"; I found no venue, so treat it as a preprint [V: q xml]. Code: github.com/Ruizhuo-Xu/Skeleton2vec.
- **What it is:** a *data2vec-2.0-style* skeleton model. It predates S-JEPA as a latent-target skeleton method but is not cited in the S-JEPA comparisons.
- **Input:** NTU 3D, DG-STGCN uniform sampling with T=90, random rotation. **Test score = average over 10 sub-sequences.** This is a protocol difference from MAMP and S-JEPA (single clip).
- **Tokens:** per-joint segments with l=3, i.e. 30×25 tokens.
- **Architecture:** MAMP encoder (8 layers, width 256) plus a **3-layer decoder**.
- **Teacher:** EMA with τ0=0.9999. The target is the **mean over all teacher layers of instance-normalised FFN outputs, then LayerNorm**, taken at masked tokens. Loss is MSE (data2vec 2.0).
- **Masking:** motion-aware *tube* masking at 90%. Within each tube of α=5 tokens (15 frames) the same joints are masked; selection uses p = U(0,1) + β·(normalised motion) with β=0.1.
- **Training:** 600 epochs, batch 128, 2× RTX 4090.
- **Results** [V: Tables 1–4]:
  - Linear 85.7 / 90.3 / 79.7 / 81.3; PKU-II 55.6.
  - Fine-tune 93.1 / 97.8 / 89.5 / 91.1.
  - Semi-supervised NTU-60: 1% → 75.7 (xsub) / 76.2 (xview); 10% → 89.2 / 92.9.
  - Transfer to PKU-II: 73.0 / 75.1.
- **Ablations** (200 epochs, linear xsub/xview) [V]:
  - Masking: random 79.4 / 85.1; tube 83.0 / 87.2; motion-aware tube 83.5 / 87.7.
  - Tube length and β: as in R3.
  - Ratio: 80/85/90/95% → 83.1 / 83.3 / 83.5 / 77.1.
  - Schedule (τ0=0.9999): 100→79.2, 200→83.5, 300→84.9, 400→85.3, 500→85.5, 600→85.7 [V: Fig. 3 values].
  - Text: smaller τ0 (0.99, 0.999) "leads to a rapid performance improvement in the early stages… [then] a decline", and 0.99999 "significantly slow[s] down convergence" [V].
- **Relevance:** it is the closest *working* precedent for "latent targets + temporally persistent masks". It is also the cleanest random-vs-tube ablation on skeletons.

**BYOL for skeletons** — Moliner, Huang, Åström. **CVPR-W 2022** (L3D-IVU), arXiv **2202.02232** [V: q xml].
- **Setup:** ST-GCN encoder. Input is **joint + bone concatenated** over 150 frames, centred and canonically rotated. 1600 epochs, batch 512, BYOL momentum 0.99→1.
- **Two asymmetric augmentation pipelines:**
  - *Aggressive*: Savitzky–Golay low-pass filter (order 2, window 15, p=0.5); speed resampling 0.7–1.3; shear in [−1, 1]; cyclic temporal shift; **left/right limb drop** (p=0.5).
  - *Conservative*: resampling only.
- **Multi-viewpoint sampling (MVS):** two NTU cameras recording the same action form the positive pair.
- **Linear results:** **86.8 / 91.2 / 77.1 / 79.2**; PKU-II 55.25 [V: Table 1]. This is higher on NTU-60 than MAMP (84.9) and S-JEPA (85.3), yet it is absent from their tables.
- **Ablation** (linear CS / CV) [V: Table 2]:

  | Setting | CS / CV |
  |---|---|
  | BYOL, conservative only | 51.1 / 55.1 |
  | BYOL, aggressive only | 79.6 / 81.0 |
  | Full minus shear | 78.0 / 81.1 |
  | Full minus low-pass filter | 84.4 / 90.2 |
  | Full minus temporal shift | 84.7 / 89.4 |
  | Full minus resampling | 85.6 / 90.6 |
  | Full minus L/R drop | 85.6 / 89.7 |
  | Without MVS (asymmetric, all augmentations) | 81.8 / 82.3 |
  | MVS + aggressive only | 83.8 / 88.7 |
  | **Full** | **86.8 / 91.2** |

  Column assignment for the single-✓ rows is inferred from row semantics. MVS is worth about +5 CS, and **ASL Citizen has no multi-view recordings**.
- **Semi-supervised:** 1× model at 1% labels: 75.8 ± 1.0; 79.4 ± 0.6 after distillation. Labels are sampled **class-balanced** [V: Table 5, text].
- **Relevance:** low-pass filtering and limb dropping are directly usable for jittery 2-D keypoints and missing hands. It also shows how much an EMA teacher-student depends on augmentation strength.

**HA-CM** — Yin, Cao et al. arXiv **2409.17951**; the comment says "IEEE Trans" [V: q xml]. Code: github.com/YinxPeng/HA-CM-main.
- **What it is:** a MAMP-based cross-masking method. Hyperbolic joint-hierarchy masking and attention-guided temporal masking alternate on odd and even samples, and a cross-contrast loss is added.
- **Results:** linear 86.3 / 91.2 / 78.9 / 80.2; PKU-II 50.9. Fine-tune 92.4 / 97.7 / 89.1 / 90.5 [V: Tables I–II].
- **Valuable ablations:**
  1. **Gumbel noise off:** MAMP 84.9 → **66.7**; HA-CM 85.9 → 68.1 [V: Table VI].
  2. **Redundancy, on MAMP** [V: Table V]:
     - Joints 25→20 / 18 / 16: 84.8 / 85.1 / 81.5.
     - Clip length 120→92 / 72 / 64: 84.3 / 83.8 / 82.0.
     - Pooling size l=3 / 5 / 6: 84.9 / 84.4 / 84.2.
     - Best efficient setting: 18 joints, 72 frames, l=3 → 85.2 in 6.6 h (vs 14.3 h).
  3. **Mask criteria** (uniform masking): motion intensity 85.2, temporal-attention variants 85.6 / 84.6, spatial-hierarchy 84.0. Cross-masking 85.2–86.3 [V: Table VII]. Mask criteria move results by only about ±1.
- **Also reported, from its tables [V\*]:**
  - MMFR (Zhu, Shu, Tang, TCSVT'24): linear 84.2 / 89.5 / 77.1 / 78.8; FT 91.9 / 96.5 / 87.4 / 90.4.
  - SCD-Net (AAAI'24, contrastive): linear J **86.6 / 91.7 / 76.9 / 80.1**; semi 1% 69.1 / 66.8.

**STARS** — arXiv 2407.10935. L3 lists only its linear numbers; this card adds the frozen-feature results.
- **Method:** MAMP pretraining, then **20 epochs** of NNCLR contrastive tuning of the second half of the encoder (layer-decay α=0.2), with a BatchNorm projector.
- **kNN, K=1** [V: Tables 2/11/14/16/17]:
  - MAMP (their reproduction): **63.1 / 80.3 / 51.8 / 56.1**. STARS: **79.9 / 88.6 / 67.6 / 67.7**.
  - Few-shot, 1-shot: MAMP 47.6 → STARS 63.5.
  - MAMP + NNCLR jointly: 74.6 / 86.5. NNCLR from scratch on the same transformer: 37.6–41.2 xsub.
  - Naive MAE (joint target): 44.1 / 43.7 kNN; 53.5 / 55.2 after tuning.
  - Full-layer tuning (α=1.0): 74.3 vs α=0.2: 83.7 (kNN K=10).
  - Batch size 16 → 128: kNN 76.7 → 79.9.
- **Relevance:** directly relevant to dictionary retrieval. The best skeleton kNN on record comes from *MAE pretraining followed by short contrastive tuning*. Note that SLiM's "best retrieval" claim (72.5) omits STARS (79.9 xsub).

**Self-Supervised Skeleton-Based Action Representation Learning: A Benchmark and Beyond** — Zhang, Lin, Yang, Liu (PKU). **IJCV 2025**, arXiv 2406.02978 [V: q xml].
- **What it is:** a survey plus a *re-run* benchmark using official weights, and the PCM3++ method.
- **Multi-task benchmark** [V: Table 3]. Columns: retrieval (NTU-60 xsub/xview, NTU-120 xsub/xset); linear accuracy with occlusion on NTU-60 xview (spatial/temporal); PKU detection mAP@0.1/0.5; few-shot on unseen NTU-120 classes (5-way 1-shot / 5-shot).

  | Method | Retrieval | Occluded | Detection | Few-shot |
  |---|---|---|---|---|
  | CMD | 70.6 / 85.4 / 58.3 / 60.9 | 72.7 / 79.5 | 73.7 / 68.4 | 64.9 / 82.8 |
  | PCM3 | 73.7 / 88.8 / 63.1 / 66.8 | 87.0 / 86.1 | 73.3 / 68.2 | 65.2 / 82.9 |
  | ActCLR | 69.6 / 76.1 / 51.4 / 54.9 | 48.3 / 53.3 | 41.9 / 35.1 | **68.0** / 84.3 |
  | AimCLR | 58.8 / 67.9 / 43.7 / 42.4 | 48.5 / 54.7 | 43.9 / 35.1 | 66.7 / 83.8 |
  | MAMP† | 62.0 / 70.0 / – / – | 56.5 / 81.0 | **78.8 / 72.4** | **56.2** / 76.3 |
  | UmURL\* | 72.0 / 88.9 / 59.5 / 62.2 | 71.1 / 77.4 | 43.4 / 38.3 | 64.9 / 77.2 |

  † MAMP uses more input frames. \* UmURL uses a hard-coded input length.
- **Reading:** the method with the best linear accuracy (MAMP) is *worst* on few-shot and among the worst on retrieval, but *best* on detection. GCN contrastive models collapse under spatial occlusion (about 48).
- **Pretraining data scale**, PCM3++ evaluated on NTU-60 xsub recognition / NTU-60 retrieval / few-shot 5-1 [V: Table 4]:

  | Pretraining data | Recognition | Retrieval | Few-shot 5-1 |
  |---|---|---|---|
  | NTU-60 | 84.8 | 75.4 | 66.7 |
  | NTU-120 | 85.1 | 75.3 | 69.6 |
  | PKU-MMD (about half the size of NTU-60 [K]) | **77.6** | 65.1 | 58.9 |
  | NTU-120 + PKU | 85.8 | 76.2 | 69.4 |

  - Halving the data costs 7 points.
  - Adding *new classes* does not help in-domain tasks but helps unseen-class few-shot (+2.9).
- **PCM3++ ablation** [V: Tables 5–6]:
  - Masked/predicted views as contrastive positives: xview 88.2 → 89.3 → 90.0. With contrastive gradients into the masked branch: 90.5.
  - Clip-level contrast: detection 73.3 → 75.5.

**USDRL — "Foundation Model for Skeleton-Based Human Action Understanding"** — Wang, Weng et al. **TPAMI** (arXiv 2508.12586). This is the journal extension of AAAI'25 USDRL.
- **Method:** a transformer dense spatio-temporal encoder, *multi-grained feature decorrelation* (variance, auto-covariance and cross-correlation over temporal, spatial and instance levels) and multi-view / multi-modal consistency. There are no negatives and no masking.
- **Key ablation** (NTU-60 xsub, linear / retrieval) [V: Table 10]:
  - STTR backbone: negative-based CL 74.8 / 63.8 vs full decorrelation 84.2 / 73.7.
  - DSTE backbone: 77.3 / 64.4 vs 85.2 / 75.0.
  - Encoder layers 1 / 2 / 3: 84.5 / 85.2 / 84.9 linear [V: Table 11].
- **2-D input:** UAV-Human 2-D skeletons, unsupervised linear CS-v1 36.3 vs supervised ST-GCN 30.2 [V: Table 3]. This is one of the very few 2-D skeleton SSL numbers.
- **Retrieval:** 75.0 / 89.3 / 63.3 / 66.7 [V: Table 4].

**MacDiff** — ECCV'24, arXiv 2409.10473. L3 gives only its results; this card adds the ablations.
- **Architecture:** an encoder on 90%-randomly-masked tokens. Its pooled global representation, together with the visible local tokens, conditions an AdaLN diffusion decoder that denoises the *whole* normalised sequence (ε-prediction, T=1000). 500 epochs [V: §3, §4.1].
- **Ablations** [V: Tables 7–10]:
  - Masking strategy: see R4.
  - Ratio: see R5.
  - Noise schedule: inverse-cosine 86.4 vs linear 83.4.
  - Decoder depth 2 / 3 / 4 / 5: 84.8 / 86.4 / 86.0 / 85.9.
  - Diffusion-based augmentation for 1% semi-supervised: 65.6 → 72.0.
- **Lesson:** with a *global-conditioned, whole-sequence* generative target, interpolation shortcuts no longer matter, so random masking wins.

**IGM** — ECCV'24, arXiv 2410.20349. This card adds robustness numbers.
- **kNN:** MAMP 62.0 / 70.0 vs IGM 69.3 / 72.6 [V: Table 2]. Module ablation: kNN 64.7 → 69.3, linear 83.3 → 86.2 [V: Table 6].
- **Corruption test (NTU-C xsub)** [V: Table 4]:

  | Method | Joint noise (p=1, σ²=0.1) | Right-arm occlusion |
  |---|---|---|
  | **MAMP** | **2.4** | **10.7** |
  | AimCLR | 6.3 | 28.1 |
  | IGM (with generative test-time denoising) | 58.7 | 56.9 |

- **Caveats:** σ²=0.1 m² is enormous noise. The arXiv ablation text contains the LLM leftover "Here's the modified text for the ablation experiments", which is a quality red flag.

**PSTL** — AAAI'23 oral, arXiv 2302.09018. This card adds ablations.
- **Method:** Barlow Twins over a triplet: unmasked anchor, Central Spatial Masking (high-degree joints masked more often) and Motion-Attention Temporal Masking (high-motion frames masked).
- **Masking ablation:** CSM+MATM 77.3 vs RSM+RTM 75.6 [V: Table 3].
- **Test-time joint shading** (number of masked joints 0 → 10, NTU-60 xsub linear) [V: Table 4]:

  | Method | 0 | 10 |
  |---|---|---|
  | PSTL | 77.3 | 73.8 |
  | AimCLR | 74.3 | 54.3 |
  | SkeletonCLR | 68.3 | 50.7 |

- **Body-part shading:** 0 → 4 parts: PSTL 77.3 → 47.3; AimCLR 74.3 → 37.9 [V: Table 5].

**Contrastive-family ablations, compact** (all [V]):
- **SkeletonCLR / CrosSCLR** (CVPR'21) [crossclr Tables 3–4]:
  - No augmentation 33.3 / 26.2; shear 0.5 + crop γ=6 gives 68.3 / 76.4.
  - Cross-view (joint+motion) mining: 68.3 → 72.9 linear on joint.
- **AimCLR** (AAAI'22, 3s, NTU-60) [aimclr Table 1]:
  - Normal augmentation 75.0 / 79.8; extreme augmentation directly 71.3 / 77.8.
  - Normal + extreme inside AimCLR's asymmetric (D3M) framework 77.4 / 82.5; + EADM 78.2 / 82.8; + NNM 78.9 / 83.8.
- **ActCLR** (CVPR'23, NTU-60 xview, joint) [actclr Tables 7–8]:
  - Skeleton-mix applied to the *non-actionlet* (static) region vs the full area: kNN 78.04 vs 73.24; linear 86.79 vs 83.05.
  - No transformations: 67.50 / 79.98.
  - **Sign analogue:** augment the static parts (torso, non-dominant hand at rest), not the active hand.
- **CMD** (ECCV'22, NTU-60 xsub) [cmd Table 5]:
  - Bidirectional joint⇄motion distillation lifts joint linear 76.1 → 79.8.
  - Joint kNN 63.4 → 64.8 for one-way motion→joint distillation; joint⇄bone gives kNN 66.5.
- **UmURL** (ACM MM'23) [umurl Tables 7–9]:
  - Early fusion of streams: J 81.7, J+M 83.7, J+B 83.3, J+M+B 84.2.
  - Modality-specific embeddings/projectors beat shared ones (84.2 vs 83.2–83.4).
- **HiCLR** (AAAI'23) [hiclr Tables 3 and 5]:
  - Baseline with [weak, normal] augmentations 72.9; adding naive random masking 56.7; adding AdaIN **13.2** (near collapse).
  - The same augmentations inside the hierarchical (weak→strong) scheme: 77.2–77.6.
  - kNN on the joint stream: 67.3 / 75.3 vs AimCLR 62.0 / 71.5.
- **HYSP** (ICLR'23, BYOL-type, hyperbolic) [2303.06242 Table 5]: linear xview *improves* with smaller batches (512 → 32: 78.3 → 82.6). This is the opposite of SLiM (384 → 768: +0.95 average [V: slim Table 7]) and STARS (bigger is better). Batch sensitivity is method-specific.

**IosPSTL / OPSTL** — IJCNN 2025, arXiv 2309.12029 [V: Tables I, VI].
- Pretraining and testing on occluded NTU (realistic furniture occlusion), linear xsub: SkeletonCLR 56.74, AimCLR 58.90, PSTL 59.52, IosPSTL 67.11. Clean performance is about 77.
- With 20% random coordinate occlusion, NTU-120 xset drops to **4.22** without imputation. Persistent missing data destroys linear separability unless it is handled explicitly.

**LA2DS — "Learning by Aligning 2D Skeleton Sequences"** — ECCV'24, arXiv 2305.19480 [V: Tables 3–4].
- Self-supervised temporal alignment using **2-D skeleton heatmaps** in a spatio-temporal transformer.
- On H2O (hand–object, 21-joint hands), dropping one finger per frame costs about 1 point vs about 8 for a 3-D-coordinate model.
- Train-clean / test-noisy: 68.59 vs 65.15.
- Weak evidence that rendering 2-D keypoints as Gaussian heatmaps adds robustness to missing fingers.

**AG-MAE** — Ikne, Allaert, Wannous. 3DV 2025 [S]. An anatomically constrained spatio-temporal MAE for online hand gesture recognition from 3-D hand keypoints (SHREC'21). Code: github.com/lambda-xyz-01/AGMAE. HAL is behind a bot wall, so no numbers were verified. This is the only hand-skeleton-specific masked SSL I found outside sign language; SignBERT+ and BEST are covered in L2/L4.

**Minor 2025–2026 contrastive papers** [S, abstracts only]: TranCLR ("Beyond Binary Contrast", 2604.17914), M3GCLR (2603.09367), VCL (2601.07666), DoGCLR (2511.14179), QCLR (ACCV'26, adversarial robustness, 2610.13281). None changes the design picture.

**"Frozen in a Frame: The Velocity Blind Spot in JEPA World Models"** — arXiv 2610.04585 [S].
- Single-frame JEPA world-model embeddings carry no velocity information: linear velocity probes are at chance. The proposed fix splits the code into an explicit pose code plus a finite-difference motion code.
- This is tangential, because our encoder sees clips. It is still a cheap diagnostic worth copying: a linear velocity or acceleration probe on frozen sign features.

### 2.2 Corrected or extended cards for papers already in L3/L2

**S-JEPA** (ECCV'24). I read the camera-ready in full (`prev/sjepa_eccv.txt`). Additions to L3:
1. **The "2-D→3-D depth target" claim stays rejected.** The abstract reads: "Given a partial skeleton sequence, predict the latent representations of the missing joints of the same sequence". Both branches embed 3-D skeletons. "Depth" is motivation only (Fig. 1: "Predict high-level 3D features") [V]. The project page agrees [V: sjepa_site].
2. **Epoch inconsistency:** §4.2 says 1200 epochs, but Table 6 reports output-masking **at 800 epochs = 85.3**, the same as the headline [V]. Either the headline is an 800-epoch result or the table mislabels it. Do not budget 1200 epochs on S-JEPA's authority.
3. **λ sensitivity** (80.0–85.3) and **β sweep** [V: Fig. 3]:

   | β | 0.5 | 0.6 | 0.7 | 0.8 | **0.9** | 1.0 |
   |---|---|---|---|---|---|---|
   | xsub | 84.3 | 84.6 | 85.1 | 85.1 | **85.3** | 84.1 |
   | xview | 88.9 | 89.2 | 89.4 | 89.5 | 89.8 | 88.8 |

   β=1.0 freezes the centre (84.1), while Table 5 reports "no centering" as 84.5, so the two are inconsistent. Centering is worth only +0.2 to +0.8.
4. **Statistics:** no projection or prototype head; the softmax is over 256 feature dimensions; centering is applied to targets only, and "Centering the predictor output as well is not necessary" [V]. Linear probe: 300 epochs, SGD with LR 0.1. Semi-supervised results are means of 5 runs with no SD.
5. **Fine-tune comparison:** S-JEPA 93.1 / 97.6 / 90.3 / 91.3 vs MotionBERT (fine-tune, 2-D HRNet input) 93.0 / 97.2 / 84.8 / 86.4 [V: Table 2].

**SLiM** (arXiv 2603.10648; NeurIPS'26 per project page). Additions:
- **Objective ablation:** R8.
- **Augmentation ablation** [V: Table 5b]:

  | Change from full model | xsub |
  |---|---|
  | No rotation | 81.6 |
  | Naive rotation | 84.8 |
  | No mirroring | 85.3 |
  | Naive mirroring | 85.9 |
  | No scaling | 84.4 |
  | Naive scaling | 86.4 |
  | **Full** | **87.7** |

- **The +3.5 STM gain is partly an augmentation effect** [V: Table 5a]:
  - STM on the global view alone, with no local-view masking: 83.3.
  - The local-view STM role adds +2.0 (85.7 → 87.7).
- **Schedule:** 50 epochs 86.12 [V: Table 12 clean control]; 100 epochs 87.66; 150 epochs 87.9.
- **Batch:** 384 vs 768 → 89.60 vs 90.55 average.
- **Variance:** single pretraining run; probe SD about 0.1.

**Path-JEPA** (ECCV'26).
- **Status unchanged:** the repo still holds only LICENSE, README and index.html (last push 2026-10-05; GitHub API) [V], and there is no arXiv version.
- Headline linear results and the fps table were re-verified on the author page [V].
- **New observation:** in the fps table the baselines' 30-fps entries equal their published headline numbers *exactly* (SkeletonMAE 74.8, MAMP 84.9, S-JEPA 85.3), while Path-JEPA's own 30-fps value (87.7) differs from its headline (87.6). The baseline fps curves may be partly copied rather than re-run, so treat the robustness claim cautiously.
- **Tokenisation ablation** [S: papernotes]:

  | Tokens | xsub |
  |---|---|
  | Global joint signatures only | 84.4 (below S-JEPA's 85.3) |
  | + local window | 85.8 |
  | + edges | 86.4 |
  | + kinematic chains | 87.6 |

  The gain comes from *relational* tokens, not from signatures per se.

**GFP** (ICCV'25). Additions:
- **Epoch curve** for SkeletonMAE / MAMP / GFP at five checkpoints up to 400 epochs [V: Fig. 4]:

  | Method | Checkpoint values |
  |---|---|
  | SkeletonMAE | 70.2 → 72.6 → 74.2 → 74.6 → 74.8 |
  | MAMP | 82.2 → 84.1 → 84.2 → 84.6 → 84.9 |
  | GFP | 83.6 → 84.9 → 85.2 → 85.5 → 85.9 |

  No crossings: the target-type ranking is set by the first checkpoint.
- Retrieval: 70.9 / 87.1 [V: Table 3].
- No VICReg-off ablation is reported.

**AMR** (CVPR'26). Additions [V: amr Tables 1, 5, 9]:
- NAT (AAAI'25, normalised-coordinate MAE): 85.1; NAT with contrastive: 86.9.
- Decoder ablation: 125-patch reconstruction with decoder masking 84.8 vs downsampling 78.9; + decoupled cross-attention decoder 86.0; + focal reconstruction 87.4.
- Loss on all patches vs masked only: 87.4 vs 87.1.

**MotionBERT** (ICCV'23). The L3 ablation was re-verified [V: motionbert Table 5]:

| Pretraining setting | xsub (fine-tune) |
|---|---|
| None | 87.7 |
| Pretrained | 89.4 |
| + noise | 90.7 |
| + mask | 91.9 |
| + in-the-wild 2-D | 93.0 |

- The input mask is **10% frame-level + 5% joint-level**; mixed Gaussian/uniform noise is "kept smooth" [V].
- This is *fine-tune* accuracy, not linear.

### 2.3 Pointers (well covered in L3/L2; no change)

- **MAMP:** L3 §4.1 is verified. The decoder-depth table shows 3 and 5 layers tie at 84.9 on NTU-60 [V].
- **SkeletonMAE (Wu; Yan):** L3 / L2.
- **MaskSem, ASMa:** L3.
- **PoseBERT, Sapiens, Human-JEPA:** L3 §4.3.
- **SignBERT+, BEST, MASA:** L2 §C and L4.
- **MaskCLR** (CVPR'24): only the fine-tune row in L3; not re-read.

---

## 3. Synthesis

### (a) Coordinate vs motion vs latent targets

1. **Raw joints are worst; any normalisation rescues them.** Evidence: MAMP +10.1 for a motion target over a raw-joint target; AMR shows the reversal once coordinates are normalised (joint 87.4 > motion 86.7); NAT reaches 85.1 [V].
2. **Latent targets beat coordinates in matched studies only when the target has the right form:**
   - Contextual: S-JEPA output vs input +7.2.
   - Distributional, i.e. CE on prototypes or softmax with centering: SLiM +3.4 over coordinates.
   - Feature MSE is roughly equal to coordinates (84.53 vs 84.30) and collapses without a predictor [V].
3. **Unmatched comparisons are confounded.** The leaderboard order is Path-JEPA 87.6 ≈ SLiM 87.9 ≈ AMR 87.4 > Skeleton2vec 85.7 ≈ GFP 85.9 ≈ S-JEPA 85.3 ≈ MAMP 84.9. It mixes latent and coordinate targets, epochs (150–1200), and tokenisation. Leaderboard order is not evidence for latent targets.
4. **Latent and global targets clearly win on kNN and few-shot,** not on linear accuracy:
   - MAMP 62.0 kNN vs GFP 70.9 vs SLiM 72.5 vs STARS 79.9.
   - Few-shot: MAMP 56.2 (lowest in the IJCV benchmark) [V].
   - **This matters most for our retrieval metric.**
5. **Hierarchical or multi-scale targets help:** GFP's learned 5/10/30-frame + global targets beat flattened low-level ones (92.0 vs 86.8 xview) [V\*, L3]. Path-JEPA's relational chain tokens add +1.8 [S].

### (b) Random vs structured / part / motion-aware masking, and mask ratio

1. **Temporal persistence is the dominant structural factor.**
   - Skeleton2vec: tube +3.6 over random; optimum about 15 frames; whole-clip tubes are worse than random (78.2) [V].
   - SLiM uses "constant-volume" tubes: small groups are masked for longer [V].
2. **Anatomical part masking:**
   - SLiM's semantic tubes over 5 groups: +3.5 over joint masking. Part of this is the local-view augmentation role (+2.0) [V].
   - SkeletonMAE-Yan: masking 3 parts is best (L3).
   - There is no shuffled-anatomy control in any skeleton paper (L2 R9c remains a gap).
3. **Motion-aware sampling is a small, fragile gain.**
   - +1.2 in MAMP; +0.5 inside tubes in Skeleton2vec.
   - Over-biasing costs 4 points (β=0.3), and making it deterministic costs 18 points [V].
   - MacDiff: motion-aware is 0.9 below random. ASMa (L3) warns about motion bias as well.
4. **The target decides whether structure helps.** Masked-token targets favour structure; MacDiff's global-conditioned whole-sequence target favours random [V].
5. **Ratio:** 90% fixed or U(0.5, 0.9) sampled; avoid 95% (−2.7 to −6.4 in three papers) [V].
6. **Masks as augmentations in contrastive branches need care.**
   - Naive random masking in SkeletonCLR-style training costs −16.2 (HiCLR).
   - Structured masking inside a triplet or hierarchy helps (PSTL +1.7 over random; HiCLR 77.6).
   - It also buys occlusion robustness (PSTL) [V].

### (c) Per-joint vs part tokens

- **No skeleton SSL paper directly ablates per-joint vs per-part tokenisation.** This is a gap.
- Indirect evidence:
  1. Temporal patching: MAMP's l=4 is best and l=6 loses 1.6 [V]. SLiM's 8-frame patches are fine with latent targets but cost coordinate MAEs 6 points [V].
  2. Relational edge and chain tokens on top of joint tokens: +1.8 [S, Path-JEPA]. With relational tokens, mask propagation is mandatory to avoid leakage (L3).
  3. Joint pruning: dropping torso joints is free and dropping head joints hurts (HA-CM) [V]. For sign, face and torso subsets are candidates for pruning, while hands are not.
  4. Sign-specific part tokens exist (SignBERT+ per-hand tokens; BEST hand/hand/body triplets) but without per-joint baselines (L2/L4).
- **Cost [K]:** COCO-WholeBody has 133 keypoints. 133 × 30 temporal tokens = 3,990 tokens, versus 750 for NTU. Per-joint tokens at MAMP granularity cost about 28 GFLOPs-equivalent per 750 tokens, scaling roughly quadratically. Part or chain tokens, or SLiM-style compact temporal tokens (which need a latent target per R11), are needed on one GPU.

### (d) EMA teacher vs frozen teacher vs regulariser

- **EMA:**
  - Necessary for S-JEPA (1.6% without it).
  - Momentum sensitivity of 5 points (λ=0.99 gives 80.0; 0.99999 gives 80.9; 0.9999 gives 85.3).
  - Interacts with schedule length (Skeleton2vec).
  - Evaluate the EMA encoder (+0.5) [V].
- **Regularisers:**
  - GFP: VICReg variance + covariance with no EMA, a co-trained target network on *motion input*, 400 epochs, 85.9 [V].
  - USDRL: decorrelation instead of InfoNCE gives +9.4 linear / +9.9 retrieval on the STTR backbone (74.8 → 84.2; 63.8 → 73.7) [V].
- **Prototype or centering heads with EMA:** SLiM uses Sinkhorn centering and KoLeo in code (L3) and gets +2.6 over feature MSE with a predictor [V].
- **Frozen teacher:** **no skeleton evidence.** The closest is STARS, whose stage 2 starts from a frozen MAMP, but that is tuning, not distillation.
- **SIGReg (LeJEPA):** **no skeleton evidence.** Both remain open design arms. L1 covers their image-domain evidence.

### (e) Contrastive vs masked for frozen linear, kNN, few-shot and fine-tune

| Protocol | Winner | Evidence [V] |
|---|---|---|
| Linear (joint-only) | Roughly a tie at the top | SCD-Net 86.6 and BYOL-skel 86.8 (J+B, uses multi-view) vs Skeleton2vec 85.7, S-JEPA 85.3, SLiM 87.9 |
| kNN / retrieval | Contrastive or global objectives | CMD 70.6, PCM3 73.7, USDRL 75.0 vs MAMP 62.0; hybrids STARS 79.9, SLiM 72.5 |
| Few-shot on unseen classes | Contrastive | ActCLR 68.0, AimCLR 66.7 vs MAMP 56.2 |
| Fine-tune | Masked transformers | MAMP 93.1 vs 3s-HYSP (GCN) 89.1; confounded by backbone |
| Low-label fine-tune (1%) | Masked or latent, but not uniformly | Skeleton2vec 75.7, SLiM 72.1, MAMP 66.0 vs 3s-SkeAttnCLR 59.6; the EMA-invariance method BYOL-skel reaches 75.8 (class-balanced label sampling) |
| Detection (dense, frozen) | Masked | MAMP 78.8 vs CMD 73.7 vs ActCLR 41.9 |
| Same ViT, objective-only swap | Masked > contrastive; both together best | SLiM: CL 73.6 vs MFP 85.3 vs both 87.7 |

**Implication for us:** with dictionary retrieval as the primary metric, a pure MAE or MAMP-style objective is the weakest choice. Use latent masked prediction plus a global term (GLCL, DINO-CLS, decorrelation, or short NNCLR tuning).

### (f) Epochs, data scale, and rank stability

- **Schedules used:**

  | Method | Epochs |
  |---|---|
  | MAMP / GFP | 400 |
  | MacDiff | 500 |
  | Skeleton2vec | 600 |
  | S-JEPA | 800 or 1200 (inconsistent) |
  | BYOL-skel | 1600 |
  | SLiM | 150 |
  | Path-JEPA | 300 [S] |

- **Convergence:**
  - Skeleton2vec: 100 epochs 79.2 → 200 epochs 83.5 → 600 epochs 85.7.
  - SLiM: 50 → 86.12, 100 → 87.66, 150 → 87.9.
  - GFP: target-type order is fixed from the first checkpoint [V].
- **Rank reversals found:**
  1. **EMA momentum vs schedule.** Low τ0 leads early and falls behind later (Skeleton2vec, text) [V].
  2. **Target type vs normalisation.** Motion > joint (MAMP) becomes joint > motion (AMR) [V].
  3. **Mask structure vs target.** Tube > random (Skeleton2vec) becomes random > tube (MacDiff) [V].
  4. **Protocol.** Among the benchmarked methods, MAMP is top on linear and detection but last on few-shot and near-last on retrieval (IJCV benchmark) [V].
  5. **Batch size vs method.** BYOL-type HYSP prefers small batches; SLiM and STARS prefer large ones [V].
- **Data scale** (PCM3++): halving pretraining data costs 7.2 linear and 10.3 retrieval [V]. ASL Citizen's training split is NTU-60-sized [K], so we are in the regime where ablations at 100–200 epochs gave the same *direction* as full runs (SLiM, Skeleton2vec, GFP). No paper shows a reversal of *method* rankings between about 100 epochs and full schedules on NTU-sized data.
- **Pragmatic rule:**
  - Screen at about 100 epochs.
  - Confirm the top 2–3 at the full schedule with 2 seeds.
  - Treat EMA momentum and batch size as schedule-coupled and re-tune them at the full length.

### (g) Known failure modes

1. **Interpolation shortcut** (per-frame or per-joint random masks): Skeleton2vec α=1 gives 79.4 vs 83.0 with tubes; MAMP at 70% gives 76.6; joint→joint MAE gives 74.8 [V].
2. **Over-hard masks:** whole-clip tubes 78.2; 95% ratio 77.1 (Skeleton2vec); SemMAE- and EPM-style failures (L2).
3. **Deterministic or over-biased masks:** −18.2 (Gumbel off); −4.0 at β=0.3 [V].
4. **Collapse:**
   - No EMA: 1.6%. Feature-MSE MFP: effective rank 1.12.
   - Strong augmentation inside naive contrastive training: AdaIN 13.2% (HiCLR); conservative-only BYOL 51.1 [V].
5. **Reconstruction features that are not metric:** MAMP kNN 62.0 vs linear 84.9; naive MAE kNN 44.1 [V].
6. **Brittleness to corrupt coordinates:**
   - Zero-fill 49.6 and parent-fill 66.5 vs mask token 87.2 (SLiM).
   - MAMP under heavy noise 2.4% (IGM); persistent occlusion 4.2–41.9 (IosPSTL) [V].
   - Degraded-data pretraining costs −2.26 on clean data (SLiM) [V].
7. **Temporal-resampling sensitivity:** MAMP varies 3.7 points from 15 to 60 fps; S-JEPA 2.9 [V, author page; possibly copied baselines]. Relevant for signer speed variation.
8. **Relational-token leakage:** an unmasked bone, edge or chain token reveals the "masked" joint (Path-JEPA argument [V abstract]; L2 §4).
9. **Velocity blindness** of latent targets is plausible (TI-JEPA [S]). Probe velocity and acceleration from frozen features.

---

## 4. Evaluation pitfalls in the skeleton SSL literature

1. **Protocols disagree** (§3e). A method can be first on linear and last on few-shot (MAMP). Report linear, kNN *and* a low-shot probe.
2. **Reproductions disagree by up to 10 points.** MAMP kNN xview is 70.0 in the IJCV benchmark (copied by GFP, SLiM and IGM) vs 80.3 in STARS's reproduction [V]. Always re-run baselines in your own pipeline.
3. **Test-time augmentation differs.** Skeleton2vec averages 10 sub-sequences at test time [V]. MAMP and S-JEPA use a single clip at a 0.9 crop ratio.
4. **Probe hyperparameters vary widely:** 300 epochs at LR 0.1 (S-JEPA), 100 epochs at LR 0.1 (Skeleton2vec), LR 3.0 (CrosSCLR), LR 30 (BYOL-skel) [V]. Fix one probe recipe and sweep its LR.
5. **Ablations run on shorter schedules than headlines:** SLiM 100 vs 150 epochs; Skeleton2vec 200 vs 600; S-JEPA's 800 vs 1200 ambiguity [V].
6. **No pretraining-seed variance anywhere.** SLiM says so explicitly. Semi-supervised "mean of 5 runs" only varies the label subset; the only SDs I found are BYOL-skel's 0.3–1.5 [V].
7. **Baselines are omitted:**
   - BYOL-skel (86.8) and SCD-Net (86.6) are missing from the S-JEPA and SLiM linear tables.
   - STARS kNN (79.9) and USDRL retrieval (75.0) are missing from SLiM's "best retrieval" table [V].
   - Skeleton2vec (2024) is missing from S-JEPA.
8. **Confounds are mixed together:** 3-stream ensembles vs joint-only, GCN vs GRU vs transformer, multi-view positives (MVS), and different input lengths (MAMP† more frames; UmURL's hard-coded length) [V].
9. **Corruption protocols are not comparable.** IGM uses σ² = 0.1 m² absolute noise; SLiM uses 0.05 × bone length; the benchmark zeroes parts at a 0.3–0.7 ratio. Define noise relative to sign-specific scales such as hand size.
10. **Copied robustness baselines:** Path-JEPA's fps-table baselines match their published 30-fps headlines exactly [V].
11. **Which encoder is evaluated matters:** the EMA (target) encoder beats the online one by 0.5 (S-JEPA) [V]. Pre-register which is used.
12. **Text-quality red flags** (IGM's LLM leftover) warrant extra scepticism toward ablation narratives.

---

## 5. Design choice → measured effect → source → transfer risk to sign language

Abbreviations: Δ = gain of the first option over the alternative. "Risk" refers to 2-D COCO-WholeBody keypoints from ASL Citizen: small and noisy hands (21 joints each), fine finger articulation, 68 face points, missing or jittery hands, single camera, signer-independent evaluation.

| # | Design choice | Measured effect | Source | Transfer risk to sign |
|---|---|---|---|---|
| 1 | Motion target vs raw-joint target | +10.1 (84.9 vs 74.8) | MAMP Table 7 [V] | **High.** 2-D velocity amplifies detector jitter on fingers. Use wrist-relative, smoothed, scale-normalised targets; AMR shows normalised joints ≥ motion. |
| 2 | Normalised joints vs motion (target) | +0.7 (87.4 vs 86.7) | AMR Table 8 [V] | Medium. Per-part normalisation (shoulder width for body, hand size for hands) is needed for 2-D. |
| 3 | Teacher-output vs teacher-input targets | +7.2 | S-JEPA Table 6 [V] | Low. Architecture-generic. |
| 4 | Prototype-CE latent vs coordinate MSE (matched) | +3.4; feature MSE without predictor collapses (rank 1.12) | SLiM Table 10 [V] | Medium. Fine handshape differences may be absorbed by coarse prototypes; monitor rank and handshape probes. |
| 5 | CE + centering vs MSE (latent) | +1.6 / +3.4 | S-JEPA Table 5 [V] | Low. |
| 6 | EMA vs none | 85.3 vs 1.6 (collapse) | S-JEPA Table 5 [V] | Low. Momentum must be re-tuned: Δ up to 5.3 across λ. |
| 7 | Decorrelation regulariser vs InfoNCE | +9.4 linear / +9.9 retrieval (STTR) | USDRL Table 10 [V] | Medium. Signer identity may dominate variance; whiten after signer normalisation. |
| 8 | Tube (≈15 frames) vs per-frame random joint mask | +3.6 / +2.1 | Skeleton2vec Table 5, Fig. 4 [V] | **Medium–high.** Fingerspelling and handshape changes last about 100–300 ms [K], so 15-frame tubes may erase whole phonological events. Sweep 4–16 frames. |
| 9 | Whole-clip tube vs optimal tube | −4.8 (78.2 vs 83.0) | Skeleton2vec Fig. 4 [V] | High if the whole hand is masked for the entire sign. The other hand or the arm then becomes the only cue (mirror shortcut, L2). |
| 10 | Semantic part tubes vs joint masking | +3.5 (part of it from the augmentation role) | SLiM Table 5a [V] | Medium. Groups must be redefined (fingers, hands, face regions); legs are absent. |
| 11 | Motion-aware vs random sampling | +1.2 (MAMP); +0.5 inside tubes; −0.9 (MacDiff) | MAMP, Skeleton2vec, MacDiff [V] | **High.** Motion is dominated by arm and wrist translation. Use wrist-relative finger motion, and mix in uniform seeds. |
| 12 | Motion-bias strength β 0.1 vs 0.3 | +4.0 | Skeleton2vec Table 6a [V] | High. Over-masking moving hands removes the dominant hand almost always. |
| 13 | Stochastic (Gumbel) vs deterministic mask | +18.2 | HA-CM Table VI [V] | Low. Always keep randomness. |
| 14 | Mask ratio 90% vs 70% / 95% | +8.3 / +2.7–6.4 | MAMP, Skeleton2vec, MacDiff [V] | Medium. With 133 keypoints and 42 hand joints, the ratio should be set per part. Face points are highly redundant (high ratio OK); fingers are not. |
| 15 | Sampled U(0.5, 0.9) vs fixed 90% | +0.9 | SLiM Table 5a [V] | Low. |
| 16 | Random mask vs tube, with a *global-conditioned generative* target | +3.3 for random | MacDiff Table 7 [V] | Medium. Relevant if a pooled-CLS or diffusion objective is used. |
| 17 | Missing joints → mask token vs zero-fill / parent-fill (8/25 missing) | +37.6 / +20.7 | SLiM Table 11 [V] | **Very high relevance.** Hand detections drop out often. Use confidence → mask token at train *and* test. Note the risk that the detector's own failures correlate with handshape. |
| 18 | Masking-as-augmentation (contrastive) → occlusion robustness | 73.8 vs 54.3 at 10 masked joints | PSTL Table 4 [V] | High relevance for partially occluded hands. |
| 19 | Naive mask / AdaIN augmentation in plain contrastive training | −16.2 / collapse (13.2) | HiCLR Table 3 [V] | Medium. Put strong perturbations only on the student or in a hierarchy. |
| 20 | Aggressive vs conservative augmentation (BYOL) | +28.5 (79.6 vs 51.1) | BYOL-skel Table 2 [V] | **High.** 3-D shear and rotation do not exist in 2-D. Use 2-D affine, small shear, speed resampling, and low-pass filtering (+2.4). L/R drop (+1.2) conflicts with two-handed lexical signs. |
| 21 | Bone-length-preserving vs naive scaling | +1.3; rotation removal −6.1 | SLiM Table 5b [V] | Medium–high. Bone-preserving scaling simulates signer anthropometry (hand size), which suits signer independence. Mirroring changes handedness, so use a true L/R swap or dominant-hand canonicalisation. |
| 22 | Augment static region vs full area (actionlet) | +3.7 linear / +4.8 kNN | ActCLR Table 7 [V] | Medium. Static regions in sign (torso, a resting non-dominant hand) are well defined. |
| 23 | Early fusion J+M+B vs J only | +2.5 | UmURL Table 7 [V] | Medium. Bone vectors of fingers are short and noisy in 2-D. |
| 24 | MAE + 20-epoch contrastive tuning vs MAE (kNN) | +16.8 kNN (linear about +2.2) | STARS Tables 2, 11 [V] | **Very high relevance** for dictionary retrieval. Cheap stage-2 recipe. |
| 25 | Masked feature prediction + global multi-rate contrast vs MFP alone | +2.4 | SLiM Table 4 [V] | High relevance. Multi-rate local views address signing-speed variation. |
| 26 | Relational edge + chain tokens vs joint tokens | +1.8 | Path-JEPA (papernotes) [S] | High relevance (finger chains). Needs leak-free mask propagation. |
| 27 | Compact 8-frame temporal tokens with coordinate target | −6.1 (MAMP); −0.35 with SLiM latent | SLiM Table 7 [V] | High. Do not combine coarse patches with coordinate targets; keep ≤4-frame patches for hands. |
| 28 | Prune torso joints 25→18 | +0.2, 0.6× time; 16 joints −3.4 | HA-CM Table V [V] | Low–medium. Prune face contour and lower body; keep hands, mouth and eyes. |
| 29 | Corrupted-2-D input pretraining (noise + mask) | +2.5 fine-tune | MotionBERT Table 5 [V] | High relevance (noisy 2-D); fine-tune metric only. |
| 30 | Halve pretraining data | −7.2 linear / −10.3 retrieval | IJCV benchmark Table 4 [V] | Medium. ASL Citizen is NTU-60-sized [K]; adding other sign corpora likely helps unseen-sign retrieval most. |
| 31 | Evaluate EMA vs online encoder | +0.5 | S-JEPA Fig. 3b [V] | Low. |
| 32 | Batch size (SLiM 768 vs 384; HYSP 32 vs 512) | +0.95; small batch +4.3 for HYSP | SLiM Table 7; HYSP Table 5 [V] | Medium. A single GPU forces small batches; prefer methods whose collapse guard is not batch-statistics-heavy, or use gradient accumulation with care. |

---

## 6. Corrections to L2 / L3

1. **L3 §0.3 "Structured masking is now consistently better… on skeletons" is too strong.**
   - MacDiff (ECCV'24), with a diffusion decoder conditioned on a global representation, finds **random 86.4 > spatial-temporal 85.3 ≈ motion-aware 85.5 > temporal 84.1 > tube 83.1** [V: Table 7].
   - Skeleton2vec shows over-structured masks also lose: whole-clip 78.2 < random 79.4; β=0.3 gives 79.5 [V].
   - **Restate as:** "structured masking helps when the loss is on masked tokens and the mask persists about 0.5 s; it can hurt with whole-sequence or global-conditioned targets."
2. **L2 bottom line 3** ("with semantic latent targets, structured masking matters less") **does not hold for skeletons.** The largest skeleton structured-masking gains occur *with* latent targets: Skeleton2vec +3.6 (data2vec targets) and SLiM +3.5 (iBOT targets). The low-level-target MAMP gains only +1.2 [V].
3. **L2 §4 and R2 "use part-tubes of ≥8–16 frames, or the whole clip":** the whole-clip option is contradicted for per-joint skeleton masking (Skeleton2vec α=30 is worst). The best tube is about 15 frames; 30 frames already loses 1.3 [V]. V-JEPA's full-extent rule applies to video patches, not to joints with fixed identity.
4. **L2 "expected effect size +1–1.5 points" for mask structure:** update to **+1.2 to +3.6** in skeleton ablations. These were measured at 100–200-epoch schedules and rest on single runs [V].
5. **L2 R5 / L3 "add a motion target (+10.1)":** make this conditional. The +10.1 is relative to *un-normalised* joints. With normalised joints, AMR finds joint ≥ motion [V].
6. **L3 §4.2 contrastive table omits three strong frozen-feature baselines:**
   - BYOL-skeleton (CVPR-W'22): **86.8 / 91.2 / 77.1 / 79.2** (J+B, multi-view positives) [V].
   - SCD-Net (AAAI'24): **86.6 / 91.7 / 76.9 / 80.1** joint [V\*].
   - USDRL-TPAMI: retrieval 75.0 [V].

   The narrative "JEPA/latent beats contrastive on linear" depends on these omissions.
7. **L3 omits Skeleton2vec** (arXiv 2401.00921), an earlier latent-target skeleton model with EMA and data2vec-2.0 targets: 85.7 / 90.3 / 79.7 / 81.3 linear, with the cleanest tube-vs-random ablation [V]. S-JEPA is not the first skeleton latent-target method; it is the first *peer-reviewed* skeleton JEPA with a predictor.
8. **L3 S-JEPA card:**
   - (i) Headline 85.3 is also reported at **800 epochs** (Table 6), conflicting with the stated 1200.
   - (ii) λ sensitivity of 80.0–85.3 is not mentioned.
   - (iii) β=1.0 (84.1) vs "no centering" (84.5) are inconsistent.
   - (iv) Centering is applied to targets only.

   All [V]. The L3 conclusion "no 2-D→3-D depth trick" is **confirmed** by the abstract text.
9. **L3 SLiM card:**
   - The +3.5 STM gain should be qualified. STM on the global view alone, without local-view STM, gives 83.3 (below joint masking with local STM, 84.2). The local augmentation role contributes +2.0 [V: Table 5a].
   - Add SLiM's augmentation ablation (rotation −6.1, naive scaling −1.3) and batch sensitivity (+0.95 from 384 → 768) [V].
10. **L3 SLiM "best retrieval" (72.5 / 89.9):** STARS reaches 79.9 / 88.6 kNN (K=1) and USDRL 75.0 / 89.3. SLiM is best on xview only [V].
11. **L3 MAMP kNN** (via GFP / SLiM): 62.0 / 70.0 is the IJCV benchmark's number for "MAMP† (more input frames)". STARS's reproduction gives 63.1 / **80.3** [V]. Treat MAMP kNN xview as uncertain by about ±10.
12. **L3 Path-JEPA frame-rate table:** the baseline 30-fps values equal their published headlines exactly. They may not have been re-run, so downgrade the robustness claim from [V] to "author claim" [V observation].
13. **L3 §4.1 "SkeletonMAE-Yan is the only pre-2024 direct evidence that structured masking helps":** PSTL (AAAI'23) also shows structured over random masking: +1.7 (77.3 vs 75.6) inside Barlow Twins [V].
14. **L3 MAMP decoder note:** depth 3 and 5 tie at 84.9 on NTU-60. The NTU-120 optimum is 5 [V: mamp Table 10]. Minor.
15. **L2 skeleton entry for MotionBERT:** the 15% mask is 10% frame-level + 5% joint-level, and the noise is smoothed [V]. This is consistent with L2's "2:1"; the clarification is that the gains (+2.5 total from noise and mask) are fine-tune numbers.

---

## 7. Source files (all in `scratchpad/lit/raw/`)

**New today (`new/`):**
- 2401.00921 (Skeleton2vec), 2202.02232 (BYOL-skel), 2409.17951 (HA-CM), 2406.02978 (IJCV benchmark), 2508.12586 (USDRL-TPAMI), 2303.06242 (HYSP), 2212.02082 (HiCo), 2305.00666 (SkeAttnCLR), 2309.12029 (IosPSTL), 2305.19480 (LA2DS), 2604.17914 (TranCLR), 2602.22791, 2512.21064, 2505.23012, 2208.03497 (CPM).
- Each has a `.pdf` and a `.txt`.

**Re-grepped from the previous session (`prev/`):**
- `prev/sjepa_eccv.txt`, `prev/slim_v3.txt`, `prev/pathjepa_index.html`, `prev/pathjepa_papernotes.txt`.
- `prev/L3txt/`: mamp, macdiff, igm, gfp, amr, stars, pstl, aimclr, actclr, crossclr, cmd, umurl, hiclr, motionbert, masksem, heterskel.

**arXiv API queries:** `q/*.xml`.
