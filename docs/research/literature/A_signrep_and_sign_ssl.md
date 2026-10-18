# A. SignRep, deep read + sign-language SSL update (to 2026-10-14)

Scope: input for a label-free **skeleton** SSL design that should beat or complement SignRep (Wong, Camgoz, Bowden, ICCV 2025, arXiv 2503.08529). Extends and corrects `docs/proposal12/literature/L4_sign_ssl.md` (2026-10-11).

**Tags.**
- **[V]** grepped in this session from raw text, with table/section given. Raw files are in `scratchpad/lit/raw/` (SignRep: `signrep_v1.txt` from arxiv.org/html/2503.08529v1; `signrep_iccv_main.txt`, `signrep_iccv_supp.txt` from the CVF open-access PDFs; new papers: `p_<arXivid>.txt`, `signlang*.txt`). SignSeek, STC, SHuBERT and SignDino were re-grepped from the repo copies in `docs/proposal12/research_update_20261011/sources/*.txt`.
- **[V-L4]** verified by L4 against its own raw text and carried over without a new grep.
- **[V\*]** number taken from a *citing* paper's table (secondary).
- **[S]** search snippet or metadata only. **[K]** background knowledge, not verified here. **[I]** project-internal measured record (`docs/proposal4/.../CLAIM_LEDGER.md`, `gap_map.md`).

**Version note [V].** arXiv lists only v1 (11 Mar 2025); `/html/2503.08529v2` and `v3` return the 7.7 kB "not found" stub. Every decimal number in the ICCV camera-ready main+supplement is also in arXiv v1, and v1 has no extra numbers. A word-level diff of the two texts shows only formatting changes plus one Table 1 caption tweak ("…multi-crop inference, RGB and optical flow"). **arXiv v1 ≡ ICCV 2025 for every number below.**

---

## 1. SignRep: complete card

### 1.1 Problem framing and claimed contributions (§1) [V]
- **Framing:** labelled sign data is scarce ("typically under 2000 unique signs"). General pretrained video models lack sign-specific features. SOTA ISLR relies on multi-modal or ensemble models. Sentence-aligned SLT pretraining does not scale to unlabelled data.
- **Goal:** a single-modality, single-model RGB encoder learned without labels, with "strong individual sign embeddings" as a step before inter-sign models.
- **Five claimed contributions:**
  1. An MAE framework with sign priors, adversarial loss and feature regularisation.
  2. A class-similarity analysis giving an auxiliary class-distribution KL loss for fine-tuning.
  3. SOTA ISLR with a single model.
  4. Dictionary retrieval "without any downstream training".
  5. A frozen feature extractor that cuts SLT training memory.

### 1.2 Architecture, input, data, compute
- **Backbone [V, §4, §7.2, A3]:**
  - Unmodified video **Hiera-B**, initialised from **Hiera MAE pretrained on Kinetics**. Output D = 768, drop-path 0.1.
  - Pixel decoder replaced by a "lightweight sign decoder": mean of the encoder tokens z^emb → LayerNorm → FC gives **z^avg ∈ R^768**, the representation.
  - z^avg is "upsampled temporally": Conv1d(k=1) → GELU → ConvTranspose1d(k=T), hidden 512, output **D′ = 384 per frame**, giving z^up ∈ R^{T×384}.
  - One linear head g_P per prior, per frame (Eq. 1). The activity head is an MLP on z^avg.
  - Code [V-L4 audit of `third_party/SignRep`]: `fc` = LN + Linear(768,768, no bias); `active_head` = 768→128→GELU→2. In the vendored code the sigmoid is commented out.
- **Input [V, §7.2, A2, A3]:**
  - 16 consecutive frames, justified by "a co-articulated sign typically lasting for around 13 frames".
  - Upper-torso crop, resized to 256×256, random-resized-crop to 224×224.
  - Patching from the vendored Hiera config [V-L4]: kernel (3,7,7), stride (2,4,4), giving an 8×56×56 token grid. Mask unit (1,8,8) gives 392 units, **each 2 frames × 32×32 px**.
- **Pretraining data [V, §7.2, A2]:**
  - **YouTube-SL-25.** The paper states no hours or clip count for the subset it used. Citing papers list "YT-SL-25 (3,000 h)" (SignSeek Table 2 [V\*]); L4 gives YouTube-SL-25 as 3,207 h [V-L4].
  - Filtering: (i) pose estimates must show a signer present, then the upper-torso crop; (ii) **WLASL video IDs removed** to prevent leakage. Nothing is said about MS-ASL or other YouTube-derived test sets (SignRep does not evaluate MS-ASL).
  - Two 16-frame segments are sampled from the same video in each batch for the discriminator.
- **Pose teacher [V, §3, A1]:**
  - A sign-specific 3D estimator (Ivashechkin, Mendez, Bowden, ICASSPW 2023) extracts *joint angles*.
  - **3D keypoints are reconstructed from these angles using median bone lengths computed over all of YT-SL**, "ensuring that all individuals are represented with the same body shape, thereby avoiding the leakage of person-specific features when converting angles into 3D keypoints".
  - So keypoint and distance targets are identity-normalised forward-kinematics outputs, not raw detections. L4 only said "bone lengths normalised"; this is the precise mechanism.
- **Compute [V, A3]:**
  - **500,000 iterations, batch 20, masking ratio 80%, single NVIDIA RTX 3090.**
  - Adam with 50K-iteration warm-up to 1e-4, then cosine decay; layer-wise LR decay 0.85.
  - Augmentations: Planckian jitter, random resized crop, Gaussian blur, grayscale. **No horizontal flip is listed** (flip would swap handedness; this is an inference from its absence).
  - Wall-clock time is not stated. The model has 51.5 M parameters (SignSeek Table 3 [V\*]).
- **Checkpoint selection [V, A3]:** "evaluated with retrieval on WLASL validation set every 25,000 iterations, the model achieving the best performance … is selected for subsequent retrieval, recognition and translation tasks." This is model selection on a downstream benchmark's validation set.

### 1.3 Loss terms [V, §3, §5, A1, A4, Table 9]

| Prior / term | Definition | Dim | Weight w | Target scale ψ |
| --- | --- | --- | --- | --- |
| Hand keypoints P^{h,k} (LH, RH) | 21 3D hand keypoints, **wrist as origin** ("eliminating positional variations in body space") | 21×3 each | 10 | 2.0 |
| Full-body keypoints P^{b,k} | All 61 3D keypoints (21+21 hands, 19 body) in body space | 61×3 | 10 | 1.0 |
| Hand joint angles P^{h,a} | 41 angles per hand, as sin/cos | 41×2 each | 10 | 1.0 |
| Body angles P^{b,a} | 22 body joint angles, as sin/cos | 22×2 | 10 | 1.0 |
| Fingertip "distance" P^{h,d} | From 5 fingertips to 11 destinations (knuckles, wrist, other tips) | **5×11×3** | 20 | 4.0 |
| Hand-interaction "distance" P^{b,d} | Sources: wrist + 5 fingertips of each hand (12). Destinations: 22 keypoints covering "hands, face and body components" | **12×22×3** | 20 | 1.0 |
| Signer activity P^{act} (LH, RH) | Heuristic: a hand is inactive if it is (1) below the mean y of keypoints {0,3,6,7} ("middle of the stomach") and (2) the summed temporal std of its 21 keypoints is < 0.26 | 2 | 0.2 | – |
| Variance (VICReg-style hinge on z^avg, σ ≥ 1) | Eq. 3–4 | – | 1.0 | – |
| Covariance (off-diagonal² on z^avg) | Eq. 5–6 | – | 0.004 | – |
| Adversarial style loss | Eq. 7–9 | – | 2.0 | – |

Notes:
- **Reconstruction:**
  - Per-frame **smooth-L1** regression of every prior, including the activity prior "for consistency" (Eq. 2).
  - **Keypoints with confidence < 50% are dropped and missing keypoints are masked from the loss** (§5.1).
  - The trailing "×3" on the distance priors means these are per-axis 3D **displacement vectors**, not scalar distances. The paper calls them "distance matrices"; this reading is an inference from the stated dimensions.
- **Style-agnostic adversary (§5.2, A4):**
  - z^style is the Gram matrix of the z^emb tokens averaged over columns (∈ R^768), scaled ×100, then passed through a 2-layer MLP (768) and LayerNorm.
  - It is concatenated with z^avg and fed to a 4-layer spectral-norm MLP discriminator 𝔇, trained with BCE: matching = same video (segments A1, A2); non-matching = another video in the batch.
  - 𝔇 uses Adam with betas (0.5, 0.9) and weight decay 1e-3. An EMA (momentum 0.1) tracks the expected discriminator outputs for matched and unmatched pairs.
  - The encoder's loss is a squared hinge pushing D(matched) down and D(unmatched) up relative to those expectations. It is **applied only when E_M[𝔇] > E_U[𝔇]**.
  - "Style" is defined as "background and appearance". Kinematic signing style is not targeted.
- **No pixel reconstruction and no token-level target.** Every prior is decoded from the **single pooled z^avg** of the visible tokens.

### 1.4 Masking [V, §4, A3; V-L4 code]
- Standard Hiera MAE random masking at **80%**, uniform over mask units of 2 frames × 32×32 px. Nothing is anatomical or hand-centred.
- The masked tokens are never reconstructed. The pooled visible tokens must regress whole-clip, all-frame priors.
- In effect masking works as heavy information dropout. Its value shows only in retrieval (Table 5): "While we observe lower reconstruction loss without masking, this does not translate to better retrieval scores."
- At inference `mask_ratio = 0`.

### 1.5 Every ablation table, reproduced [V]

**Table 5:** "Comparison of the impact of sign priors, masking and regularization on the impact of retrieval. Retrieval results are obtained on WLASL using a stride of 8." The metric is DCG on WLASL (which split is not stated; checkpoints were selected on WLASL val).

| angle | kpt | dist | mask | var+cov | adv | DCG | Δ vs previous row |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ✓ | | | ✓ | | | 45.1 | |
| | ✓ | | ✓ | | | 32.9 | |
| | | ✓ | ✓ | | | 47.2 | |
| ✓ | ✓ | ✓ | ✓ | | | 48.5 | +1.3 over the best single prior |
| ✓ | ✓ | ✓ | | | | 46.3 | −2.2 without masking |
| ✓ | ✓ | ✓ | ✓ | ✓ | | 49.9 | +1.4 from var+cov |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **50.7** | +0.8 from adv |

**Table 6:** "Comparisons of the impact of pretraining and weighting for the class distribution loss on WLASL" (fine-tuned, WLASL2000).

| Method | κ | P-I top-1 | P-I top-5 | P-C top-1 | P-C top-5 |
| --- | --- | --- | --- | --- | --- |
| MAE (Kinetic) | – | 51.5 | 83.4 | 48.4 | 82.0 |
| MAE (YT-SL) | – | 57.2 | 87.4 | 54.6 | 86.3 |
| SignRep | 0.0 | 59.9 | 90.2 | 57.4 | 89.2 |
| SignRep | 0.1 | 60.4 | 90.1 | 57.8 | 89.1 |
| SignRep | 0.2 | **61.0** | 90.3 | 58.9 | 89.4 |
| SignRep | 0.5 | 59.8 | 89.4 | 57.7 | 88.8 |

**Table 4:** "Comparison of generalization results on sign retrieval with no downstream training applied to SignRep." This is the frozen protocol. Test queries are matched to the train-set dictionary by cosine, taking the label of the most similar train video. Stride 2 (§7.2). Raw-pose rows are "raw 3D keypoints and joint angles averaged over time" (§7.4).

| Features | ASLC DCG | ASLC R@1 | ASLC R@5 | WLASL DCG | WLASL R@1 | WLASL R@5 | NMFs DCG | NMFs R@1 | NMFs R@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HieraMAE-Kinetic | 11.64 | 0.25 | 0.84 | 13.21 | 2.08 | 3.40 | 23.29 | 3.96 | 12.18 |
| HieraMAE-YTSL | 12.12 | 0.39 | 1.34 | 14.06 | 2.57 | 4.41 | 28.03 | 7.57 | 18.38 |
| All Joint Angles | 13.82 | 0.57 | 2.17 | 17.92 | 2.54 | 6.50 | 32.51 | 7.93 | 25.50 |
| All Keypoints | 14.29 | 0.98 | 2.74 | 19.37 | 3.16 | 8.23 | 36.64 | 12.26 | 30.99 |
| Hand Keypoints | 25.91 | 7.96 | 19.58 | 28.11 | 7.57 | 20.92 | 41.72 | 15.91 | 42.62 |
| Hand Joint Angle | 26.93 | 8.81 | 21.41 | 30.61 | 9.42 | 24.36 | 44.17 | 18.13 | 46.34 |
| SignRep (avg) | 61.40 | 37.47 | 68.77 | 53.57 | 26.16 | 60.98 | 79.51 | 58.48 | 91.85 |
| SignRep (weighted) | **71.21** | **49.95** | **80.09** | **57.93** | **29.92** | **67.41** | **83.05** | **63.04** | **95.63** |

Activity weighting (Eq. 11–12; γ = max(P_LH, P_RH) from the activity head) adds R@1 **+12.48** on ASL Citizen, +3.76 on WLASL and +4.56 on NMFs-CSL (computed from Table 4).

**Ablations the paper does not run** (absent from the paper):
- Kinetics-MAE initialisation vs scratch.
- Pretraining data scale.
- Mask-ratio sweep; frames ≠ 16.
- Pairwise prior subsets, or hand vs body priors.
- Training without the activity head.
- Any ablation on ASL Citizen or NMFs-CSL.
- Seeds or variance.
- A **skeleton encoder trained on the same priors**.
- The schedule length of the Table 5 runs.

### 1.6 Downstream protocols and main numbers [V]
- **Fine-tuned recognition (§7.2, A3, A6):**
  - Input extended to 64 frames by **inflating the patch embedding**: kernel (3,7,7)/stride (2,4,4) becomes kernel (7,7,7)/stride (8,4,4), zero-initialised, with old temporal taps {0,1,2} copied to {1,3,5}. The token count stays the same.
  - Linear classifier on pooled features. CE with label smoothing 0.1 plus class-distribution KL (κ = 0.2). Same augmentations as pretraining, no patch masking.
  - Batch 8, 100 epochs, AdamW (0.9, 0.95), weight decay 0.5, LR 1e-4, layer decay 0.85, gradient clip 1.0.
  - The class-distribution matrix φ comes from **train-set** cosine similarities (temperature chosen per class so that max φ_c < 0.5), so no test leakage.
- **Table 1 (WLASL2000):**
  - SignRep **61.05 / 90.27 / 58.89 / 89.44** (P-I top-1/top-5, P-C top-1/top-5).
  - Context rows: ST-GCN 34.40; SignBERT 39.40; BEST 46.25; SignBERT+ 48.85; NLA-SLR 61.05/58.05; NLA-SLR (3 crops) 61.26; StepNet (R+F) 61.17; StepNet RGB 56.89; I3D (BSL1K) 46.82.
- **Table 2 (NMFs-CSL):** SignRep **84.1 / 98.8** (top-1/top-5); NLA-SLR 3-crop 83.7; StepNet R+F 83.6; StepNet RGB 77.2.
- **Table 3 (ASL Citizen, fine-tuned recognition, *not* retrieval):** SignRep **DCG 90.84 / MRR 88.05 / R@1 81.37 / R@5 96.11**; ST-GCN\* 76.37/69.97/59.52/82.68; I3D\* 79.13/73.32/63.10/86.09 (\* = results from the ASL Citizen paper).
- **Frozen dictionary retrieval (Table 4 above):** stride 2. The label of the top-1 train video is assigned. Readout is avg or activity-weighted. The paper chose retrieval over a linear probe "as dictionary retrieval is a common and practical task".
- **Frozen features for SLT (§8):**
  - Sign2GPT with pre-extracted SignRep features (stride 2) replacing DINOv2-LoRA, with **no pseudo-gloss pretraining**:
    - Phoenix14T B-4 **20.38** vs 19.42 (ROUGE 45.17 vs 45.23).
    - CSL-Daily B-4 **16.33** vs 12.96 (ROUGE 42.67 vs 41.12).
  - Tarrés et al. on How2Sign: BLEU **8.66** vs I3D-supervised 8.03 (rBLEU 2.74 vs 2.21).
  - Cost claim: 5.5 GFLOP per frame removed; GPU memory 60 GB → 11 GB (batch 8).
- **Third-party measurements of the released checkpoint:**
  - Zero-shot BSL dictionary retrieval (726 native-signer queries vs BSL SignBank; SignSeek Table 3): **SignRep R@1 4.30 / R@5 12.62 / mAP 8.72, below MASA (pose SSL) at 6.80** and SignSeek at 12.76 [V\*].
  - SignSeek marks SignRep as "BSL Seen? ✗". YT-SL-25 likely contains BSL [K], so that flag may be wrong.
  - Project reproduction on ASL Citizen test: 49.02 [I, C011].

### 1.7 Limitations: stated vs apparent
**Stated (A8) [V]:**
- YT-SL-25 bias in signer diversity, language distribution and skin tone.
- **"Focuses solely on manual sign features"**: no facial expression or mouthing priors.
- Pretraining still depends on keypoint quality (mitigated by the SL-specific estimator, the 50% confidence filter and missing-keypoint masking).
- **The 16-frame window**: longer windows would increase token cost.

**Apparent, from the method plus project measurements:**
1. **No sign-level temporal order.**
   - The clip embedding is a weighted mean of independent 16-frame windows [V Eq. 12; I C005].
   - Project measurements: a time-reversed clip's embedding has cosine 0.959 with the original, vs 0.561 for the same gloss from another signer [I, gap_map GAP1].
   - Order-aware training-free readouts of the *same* frozen windows add **+5.90 R@1** (bins2) and **+9.40** (chamfer@20) on ASL Citizen test [I, C012].
   - A label-free set-distilled single vector reaches 57.28 vs 49.02 [I, C019].
   - Within a window, order is supervised: per-frame priors are decoded from z^avg.
2. **No part structure.** One pooled 768-d vector must encode both hands, body and all 16 frames. There are no per-articulator tokens, so per-part probes are impossible.
3. **No handedness canonicalisation.** LH and RH priors are separate. Left-dominant signers produce mirrored targets, and no flip augmentation is used.
4. **Signer invariance only via appearance.**
   - Bone-length normalisation of targets and the appearance-style adversary do not remove kinematic identity.
   - Project measurement: frozen SignRep nearest neighbours are the same signer 48.7% of the time vs same gloss 43.8% [I, H010].
   - Pose kinematics identify signers at 80.18% vs 3.23% chance [V-L4, 2410.13675].
5. **Movement is the weakest-encoded parameter.** Lift of rank-1 confusions: handshape 8.8×, location 2.9×, movement 1.6×. Static-frame windows keep 83% of R@1 [I, GAP2].
6. **Non-manuals only implicit.** Pixels include the face, but no target asks for it; the hand-interaction prior includes some face keypoints as destinations.
7. **Evaluation hygiene.**
   - Checkpoint selected on WLASL val retrieval.
   - Single run, no seeds.
   - ASL Citizen is signer-disjoint but **lexically overlapping**: every test gloss has train-gallery exemplars, so retrieval is not open-vocabulary.
8. **Inference cost.** Hiera-B on 224² RGB, sliding windows at stride 2, 51.5 M parameters, vs a 15.7 M SignSeek pose encoder [V\*].

---

## 2. What actually drives SignRep's performance

Components ranked by ablation evidence. All Table 5 numbers are WLASL DCG at stride 8, single runs, with checkpoint selection on WLASL val; treat differences under ~1.5 DCG as noise-level.

| Rank | Component | Evidence | Effect size |
| --- | --- | --- | --- |
| 1 | **Sign-specific regression targets instead of pixel MAE** (frozen) | Table 4: HieraMAE-YTSL vs SignRep-avg on ASL Citizen R@1 0.39 → 37.47; WLASL DCG 14.06 → 53.57 | ≈ +37 R@1; pixel-MAE features are *worse than mean-pooled raw hand angles* (8.81). Confounded with regularisers and the head, but no regulariser plausibly explains +37. |
| 2 | **Readout: activity-weighted pooling** | Table 4: avg → weighted, ASL Citizen R@1 37.47 → 49.95 | +12.5 R@1 on ASL Citizen, +3.8 WLASL, +4.6 NMFs. A post-processing gain that depends on the *learned activity head*. The project adds a further +5.9 to +9.4 with order-aware readouts [I C012] and +8.3 with a label-free distilled vector [I C019]. |
| 3 | **Relational/invariant targets over absolute keypoints** | Table 5: dist 47.2 ≈ angle 45.1 ≫ kpt 32.9 | +12–14 DCG between target types; combining all adds only +1.3. **Wrist-relative distances and angles carry the signal.** |
| 4 | **In-domain pretraining data** (fine-tuned) | Table 6: MAE-Kinetics 51.5 → MAE-YTSL 57.2 | +5.7 top-1 from YT-SL alone, more than priors+regularisers add after fine-tuning (57.2 → 59.9, +2.7) |
| 5 | Masking (80%) | Table 5: 46.3 → 48.5 | +2.2 DCG; reconstruction loss is *lower* without masking |
| 6 | Var+cov regularisation | 48.5 → 49.9 | +1.4 DCG |
| 7 | Adversarial style loss | 49.9 → 50.7 | +0.8 DCG (within noise) |
| 8 | Class-distribution KL (fine-tune only) | Table 6: κ 0 → 0.2, 59.9 → 61.0 | +1.1 top-1; κ = 0.5 hurts (59.8) |

**Confounds the paper does not control:**
- **Kinetics-MAE Hiera initialisation.** Never ablated. HieraMAE-Kinetics alone is useless frozen (R@1 0.25) but strong fine-tuned (51.5 on WLASL, above all pose SSL in Table 1).
- **Data scale.** About 3,000 h and >3,000 signers of YT-SL-25. No scale curve. Pose SSL baselines used ~146K ISLR clips (MASA, SignSeek Table 2 [V\*]).
- **RGB input.** Contains face/mouth, skin and hand-appearance detail that 3D skeletons lose. That partly explains the pose-vs-RGB location/handshape trade-off (§3, Yin et al.).
- **Teacher quality.** A sign-specific 3D estimator with confidence filtering. A skeleton model fed 2D DWPose/MediaPipe sees noisier, lower-dimensional input than SignRep's *targets*.
- **Checkpoint chosen on WLASL val retrieval**, which matters for the WLASL Table 1/4/5 numbers.
- **Readout.** About a third of the headline 49.95 comes from the activity-weighted readout, not the embedding. Comparisons must run every encoder under the same readout.

**Implication for a skeleton model.** SignRep's gain over pixel MAE comes almost entirely from *what it is asked to predict*: wrist-relative hand angles, displacements, and hand-to-body/face relations. Those are cheap, deterministic functions of the pose. A skeleton encoder sees them *as input*. The open question is whether a label-free skeleton objective can make a *pooled* embedding as discriminative as SignRep's z^avg without the RGB pathway's capacity, Kinetics initialisation and the implicit denoising by a 3D teacher.

The only label-free pose SSL measured on the same frozen protocol, MASA, gets 27.96. MASA was pretrained on 146K ISLR clips, not 3,000 h. So the data and target confounds are open, not settled against skeletons. Cross-domain, MASA already beats SignRep on zero-shot BSL retrieval (6.80 vs 4.30 [V\*]).

---

## 3. New or missing sign SSL and representation papers (relative to L4)

Search log (2026-10-14):
- Semantic Scholar forward citations of SignRep (20 citing papers), SHuBERT (32), Uni-Sign (94), MASA (33) and SignCLIP (27). SignSeek and SignDino returned 0 citations.
- arXiv API queries (sorted by date, results from 2025-03 onward):
  - `abs:"sign language" AND self-supervised` / `pretraining` / `masked` / `phonolog*`
  - `abs:sign AND dictionary AND retrieval`
  - `abs:"ASL Citizen"`, `abs:PopSign|Sem-Lex|ASL-LEX`, `abs:WLASL`, `abs:"isolated sign"`
  - `abs:sign AND (JEPA|joint-embedding)`
  - sign language AND (skeleton|keypoint|pose) AND (representation|contrastive)
- ACL Anthology sign-lang@LREC 2026 volume listing (52 papers).
- WebSearch, standard mode.

**No sign-language JEPA was found in any of these sources.** The JEPA query returned only non-sign papers. This confirms L4's headline as of 2026-10-14, within this search scope.

### 3.1 Cards

**Sigma**: Semantically Informative Pre-training for Skeleton-based SLU (arXiv 2509.21223 v3; the HTML header carries an Elsevier "Journal: Pattern Recognition" template, so acceptance is unverified [S]) [V]
- **Modality:** pose only. RTMPose 2D, **69 keypoints (21 per hand, 9 body, 18 face)**, part-specific ST-GCNs for {lh, rh, body, face}, concatenated (L×4D).
- **Objective:** text-supervised. Sign-aware early fusion with an mT5 text encoder; hierarchical (local cluster-wise + global) contrastive alignment, text matching and language modelling.
- **Data:** YouTube-ASL (530,161 samples) and CSL-News (722,715), both sentence-level (Table 1). There is no masking objective.
- **Results:** WLASL2000 fine-tuned **64.54 / 62.28** (Table 4a), vs Uni-Sign as cited 63.52/61.32. No frozen evaluation.
- **Most useful ablation (Table 10a, pretraining scale):** 0% → 23.00, 25% → 50.68, 50% → 58.83, 75% → 62.46, 100% → 64.54 P-I.
  - Table 10e: no pretraining 0.02, decoder-only 1.80, encoder-only 28.40, full 64.54.
  - **Lesson:** with this architecture, almost all ISLR performance comes from large-scale (text-)supervised pretraining. The 23.00 from-scratch number is far below a supervised ST-GCN (34.40), so the curve also reflects an LM-heavy architecture that needs pretraining.
  - Table 2 storage argument: skeleton 437 KB and 9.58 ms load per sample vs RGB 4,715 KB and 455 ms.

**VQ-ASL**: "Phonological Representation Learning for Isolated Signs Improves Out-of-Vocabulary Generalization" (Kezar, Sehyr, et al., arXiv 2509.04745, cs.CL) [V]
- **Modality:** pose (Sem-Lex poses). Transformer VQ-VAE (5 layers, 256-d, 30 latent vectors, codebook 200).
- **Biases:**
  - **PD (parameter disentanglement):** separate streams and codebooks for wrist-centred RH/LH, wrist trajectories (movR/movL), nose-centred face, and body. Hand and movement codebooks are shared L/R.
  - **PSS:** with some probability, forces ASL-LEX-labelled codes. This is weak supervision.
- **Split:** Sem-Lex **"unseen gloss" split** (train/val/test vocabularies disjoint).
- **Frozen evaluation:** MLP probes on frozen Z_q. OOV ISR probe MRR / R@10 (Table 3):
  - Baseline .381 / 45.1
  - **PD .372 / 44.2**
  - PSS .435 / 50.4
  - Full .452 / 52.8
- **OOV phonological-feature probe (PFR) MRR:** baseline .488, PD .475, PSS .565, full .583.
- **Reconstruction MSE (OOV):** .058 → .046 (PD) → .041 (full).
- **Lesson:** label-free *structural* part-disentanglement improves OOV reconstruction (−21% MSE) but **does not improve, and slightly hurts, discriminability** (.381 → .372). Phonological supervision is what helps recognition. This is a direct warning for a "part-tokenized" label-free design: structure alone is not enough.
- Caveat: the ISR probe's training split for OOV signs is not clearly specified.

**Phonological Perception of Sign Language Models** (Yin, Carter, Lu, Kocab, arXiv 2606.28667, 27 Jun 2026; code and data at github.com/kayoyin/sign-phonology) [V]
- **Protocol:** label-free *minimal-pair* sensitivity on penultimate-layer features.
  - For each pair, intra-sign cosine (same sign, different signers) is compared with inter-sign cosine (the minimally different sign), using 10 signer pairs and a t-test.
  - "% wins" = share of pairs where the trained model's t is significant (p < 0.05) and higher than its random-init counterpart.
  - Sets:
    - **ASL Citizen test: 259 Deaf-curated pairs** (137 handshape, 31 location, 73 movement, 18 orientation/NMM), 4,427 videos from 11 signers.
    - **Sem-Lex: 178 pairs** for out-of-domain testing.
    - HCS nonce stimuli: 36 handshapes × 27 contexts.
  - The ASL-trained models were **retrained with all minimal-pair signs excluded**.
- **Table 1 (ASL Citizen) % wins vs random baseline:**
  - I3D-ASL 69.50 total: handshape 69.34, **location 80.65**, movement 73.97.
  - **STGCN-ASL 81.08 total: handshape 86.13**, location 67.74, movement 80.82.
  - I3D-Kinetics 1.93.
- **Table 2 (Sem-Lex, out-of-domain):** I3D 45.07, STGCN 49.30.
- **Table 3 (human alignment on HCS):** STGCN-ASL r = 0.49 with signers and 0.55 with handshape distance; I3D-ASL r = 0.31 and 0.20.
- **Lesson:**
  - **Pose models capture handshape and movement contrasts better; RGB models capture location better.** The authors attribute this to position-normalised graphs losing frame-relative location.
  - Kinetics pretraining gives no phonological sensitivity at all.
  - Sensitivity roughly halves out of domain.
  - **Only supervised models were tested; SSL is listed as future work.** This protocol is directly usable for a frozen SSL model.

**SignMatch**: Matching Dictionary Signs to Continuous Sign Language Video (Wong, Jang, Momeni, Varol, Zisserman; arXiv 2609.01886; VGG) [V]
- RGB model initialised **from SignRep**; pose model from the Jang et al. 2025 MediaPipe+lip backbone (AGCN + Conformer, ISLR/SLT-trained).
- Prototype-structured sign space learned from **BSL Corpus annotations + BSL SignBank dictionaries (phase 1), then pseudo-labels on YouTube-ASL, YouTube-BSL, BSLCorpus and BOBSL (phase 2)**. This is label-supervised but with **no ASL Citizen training**.
- **ASL Citizen dictionary retrieval** (Table 5 "Dictionary-to-dictionary retrieval performance on ASL-Citizen"):

  | Features | DCG / R@1 / R@5 |
  | --- | --- |
  | I3D-Kinetics\* | 12.34 / 0.41 / 1.36 |
  | I3D-WLASL\* | 31.53 / 9.85 / 25.20 |
  | SignRep (SSL) | 71.21 / 49.95 / 80.09 |
  | I3D trained on ASL Citizen | 76.52 / 58.03 / 84.50 |
  | **Ours (Pose)** | **75.88 / 57.32 / 83.67** |
  | **Ours (RGB)** | **83.17 / 67.12 / 91.78** |

  \* = reported by the ASL Citizen paper.
- BSL-only training, then ASL Citizen: 80.35 / 62.75 / 89.47 (Table 9). Phase 1 alone: 74.96 / 54.92 (Table 4).
- **Lesson:** *dictionary-style supervision in another language* plus scale lifts SignRep 49.95 → 67.12. A pose encoder trained this way (57.32) roughly matches gloss-supervised SignSeek (57.66). These are the **label-supervised** ceilings that a label-free skeleton model should be compared against, not mixed with.

**OneSign** (arXiv 2609.33090) [V]
- Pose-based (four parts, Dynamic Part Graph Fusion) unified LLM (Llama-3.2-1B MA-MoE) trained on ISLR, CSLR and SLT labels.
- Table 8, ASL Citizen top-1:
  - OneSign♡ (single dataset) **65.41**
  - OneSign♠ (joint) **68.41**
  - **Uni-Sign reproduced by the authors: 60.92**
- WLASL2000: 60.45/61.02 (♡), 64.15/65.34 (♠).
- **Lesson:** supervised pose ISLR on ASL Citizen sits around 60–75 (see the table in §3.2). Uni-Sign's YouTube-ASL pretraining does not transfer especially well to ASL Citizen (60.92).

**Bootstrapping Sign Language Annotations with Sign Language Models** (arXiv 2604.07606; CVPR 2026 Findings per Semantic Scholar [S]) [V]
- Supervised pose ISR (6.6 M parameters) on MediaPipe holistic (body normalised by shoulders) plus **dominant and non-dominant hand**. It has two branches (two-hand and one-hand) with a max over logits.
- ASL Citizen R@1: **0.69 without handedness → 0.74 with handedness** (Table 3); SHuBERT 0.65, SignCLIP 0.60, baseline 0.63.
- **Lesson:** dominant/non-dominant canonicalisation is worth about +5 R@1 in a supervised pose model.
- Caveat: it says "Data splits follow the same 70/15/15 stratification by signer as the fingerspelling model", so this **may not be the official ASL Citizen split**.

**Grounding Sign Language Representation Learning in Phonology** (Vandendriessche, De Coster, Dambre; sign-lang@LREC 2026, pp. 468–476) [V]
- Dual-branch (hand / body keypoints) network with **phonological subspaces**: handshape sub-embeddings follow the Dependency Model; location uses ASL-LEX MinorLocation.
- Trained on ASL Citizen + Sem-Lex with multi-label ASL-LEX focal losses (every morpheme, not only the first) and an auxiliary gloss loss.
- **Cross-lingual non-parametric probe:** multi-hot kNN phoneme prediction on the VGT dictionary (11,248 signs, 9,826 phonologically annotated), 7-fold.
  - Handshape recall: gloss-baseline falls to **0.221 at K=100** (peak 0.588 at K=1); hierarchical model's minimum is 0.594.
  - Location: baseline peak 0.679 (K=2) vs 0.668 (K=4); at K=100, 0.534 vs 0.590.
- **Lesson:** gloss-trained embeddings are phonologically coherent only in very small neighbourhoods. The neighbourhood-scale kNN probe is a cheap, label-light frozen protocol for part embeddings. (The method itself is phoneme-supervised, not SSL.)

**SSL-SLR** (Madjoukeng et al., arXiv 2509.05188 v2; Semantic Scholar lists TMLR [S]) plus the companion sign-lang@LREC 2026 paper "Leveraging Unannotated Sign Language Data via a Robust Data Augmentation Method…" [V]
- **Modality:** pose. MediaPipe skeletons with a ViT-style transformer, up to 64 frames.
- **Objective:** negative-free three-branch (original + two augmentations) MSE with predictor and stop-gradient. Its augmentation **temporally permutes the non-discriminative start and end frames** (repositioning/co-articulation).
- **Linear evaluation (Table 1, "ASL (our)" = ASL Citizen top-500 signs): 20.46%** vs SimCLR 14.13, BYOL 16.43.
- **Protocol correction (resolves L4's "unclear"):**
  - SSL-SLR's ASL Citizen **47.06 is a *fine-tuned* score**: "SSL-SLR was pretrained and fine-tuned under the same conditions" (§6).
  - It is placed next to SignRep's *frozen retrieval* 37.47/49.95. The related-work text even says SignRep "achieved 49.95% accuracy … during fine-tuning", which is wrong; SignRep fine-tuned is 81.37.
  - It also cites SignCLIP as 46.00.
- **Lesson:** label-free pose contrastive models remain at ~20% linear on a 500-sign ASL Citizen subset.

**SignVerse-2M** (arXiv 2605.01720 v3; signerx.github.io/SignVerse-2M) [V]
- DWPose (body, hands, face, with confidences) at 24 fps over a manifest drawn from public multilingual video. Its Figure 3 is the YouTube-SL-25 per-language hour distribution.
- About **2 M clips, 55+ sign languages**. Released as per-person `poses.npz`.
- Designed for generation, but it is a **ready-made pose version of roughly SignRep's pretraining corpus**. That allows a data-matched skeleton-vs-SignRep comparison without re-extraction. The overlap with SignRep's exact subset and the WLASL-ID removal are unverified.

**Logos** (arXiv 2505.10481; Russian SL ISLR dataset) [V]
- RGB, **label-supervised** cross-lingual encoder.
- **Frozen encoder + trained classifier: WLASL2000 62.44** (AUTSL 97.25), Table 2. Fine-tuned 65.57; co-training 66.82.
- Frozen few-shot WLASL: 10-shot 61.12, 3-shot 54.10, 1-shot 37.07 (Table 3).
- **Label-mapping without a head: 23.63** (Table 5).
- Pretraining-set size matters: AUTSL-pretrained → WLASL 28.46; MM-WLAuslan → 45.21 (Table 6).
- **Lesson:** a frozen *supervised* foreign-language encoder with a linear head reaches about 62 on WLASL2000, roughly double the best SSL linear probe (STC 32.35). This is the supervised frozen-linear reference.

**Geo-Sign** (arXiv 2506.00129; NeurIPS 2025 per Semantic Scholar [S]) [V]
- Pose-based (on Uni-Sign), hyperbolic contrastive regularisation for SLT.
- Also WLASL2000 ISLR **63.64 / 61.89** ("Table 3: ISLR results on WLASL2000"). Upgrades L4's [V\*] to [V].

**SignReGraph**: "rethinking skeleton-based sign language recognition with masked self-supervision" (Ket Dong Vu, Hoai Nam Vu; *Multimedia Systems*, 2026-10-10, DOI 10.1007/s00530-026-02686-z) [S]
- Closed access, abstract elided. The Semantic Scholar TLDR says it is a skeleton GCN with "Graph Enhancement and Masked Frame Reconstruction" modules. Nothing is verified.
- Note: the first author's name matches the owner of the local ASL Citizen data path (`/home/dongvk`). It may be a lab-internal paper, so ask the user for the PDF.

**SLiM**: Less is More: Compact-Token Masked Feature Prediction for Skeleton Representation Learning (Do, Chen, Youk, Kim, KAIST; arXiv 2603.10648 v3; NeurIPS 2026 catalogue lists it as "…Feature *Learning*…" [V via `docs/proposal12/research_update_20261011/LITERATURE_AND_CODE.md` R06])
- *General* skeleton SSL (NTU-type 25 joints), not sign-specific.
- EMA-teacher masked feature prediction plus contrastive learning, compact temporal patch tokens, **semantic (anatomical) tube masking**, 50–90% mask budget.
- **This is the "SLim (NeurIPS 2026)" that L4 could not identify** (correction §5).

**Other new items, lower relevance (one line each):**
- **Lost in Expression** (Sazonov, Gurbuz, Malaia; sign-lang@LREC 2026): Uni-Sign on Isharah-1000 CSLR/SLT with feature attribution; non-manuals are under-used and generalise poorly. Qualitative diagnostic, no SSL [V abstract].
- **Gloss-Free Representation Learning for Cross-Dataset Sign Spotting** (arXiv 2608.11332): frozen DINOv2 + trainable encoder with pseudo-gloss (transcript-derived) pretraining on Turkish news. Top-5 localisation mIoU 0.235 → 0.465. RGB, weakly text-supervised [V].
- **Improving Handshape Representations… GNN** (arXiv 2509.18309): anatomically informed graph plus contrastive learning for handshape; 46% vs 25% baseline over 37 handshape classes [V abstract].
- **The Importance of Facial Features in Vision-based SLR** (arXiv 2507.20884): **the mouth is the most important facial region** for isolated SLR (DGS) [V abstract].
- **Siformer** (2503.20436): supervised skeleton with kinematic hand-pose rectification; WLASL100 86.50 [V abstract].
- **GSP-MC** (2505.02304): skeleton–LLM-description multi-positive contrastive; supervised [V abstract].
- **SignNet-1M** (2606.24361): 3DGS/diffusion-augmented robustness dataset [V abstract].
- **Seeing Who Is Signing and With Which Hand** (sign-lang 2026): **hand-height heuristics classify handedness reliably**, useful for label-free dominant-hand canonicalisation [V abstract].
- **Video-based reverse dictionary** (sign-lang 2026): supervised metric-learning pose retrieval; WLASL val mAP 0.237 [V abstract].
- **Irish SL annotation** (sign-lang 2026): DINOv2 86.12% > fusion 84.28% > MediaPipe 76.74% for gloss-level annotation support [V abstract].
- **Not found** (in the arXiv API, Semantic Scholar citation lists or WebSearch):
  - any 2025–2026 SHuBERT follow-up by the same authors beyond SignMusketeers;
  - any "SignLLM"-style *recognition* pretraining. "SignLLM" is a production model [K].
  - any new paper using PopSign for frozen evaluation. PopSign ASL v1.0 (NeurIPS 2023 D&B; 250 signs, phone recordings) is still unused as a frozen benchmark [K].

### 3.2 Corrected benchmark table: frozen protocols
Protocol F-R is training-free dictionary retrieval: test queries against the train-split gallery. R@1 means the nearest-neighbour or class-max cosine gloss is correct. SignRep and SignSeek use the same rule.

**ASL Citizen F-R (test queries, train gallery, 2,731 glosses, signer-disjoint, lexically overlapping)**

| Supervision | Modality | Method | R@1 | DCG | Tag / source |
| --- | --- | --- | --- | --- | --- |
| none (features) | 3D pose, mean-pooled | All joint angles / all keypoints | 0.57 / 0.98 | 13.82 / 14.29 | V, SignRep T4 |
| none (features) | 3D pose, mean-pooled | Hand keypoints / hand joint angles | 7.96 / **8.81** | 25.91 / 26.93 | V, SignRep T4 |
| **label-free SSL** | **2D pose** | **MASA** (146K ISLR clips, no labels) | **27.96** | 49.31 | V\*, SignSeek T2 |
| label-free SSL | RGB | HieraMAE-Kinetics / HieraMAE-YTSL | 0.25 / 0.39 | 11.64 / 12.12 | V, SignRep T4 |
| label-free SSL (+3D pose teacher) | RGB | **SignRep avg / weighted** | 37.47 / **49.95** | 61.40 / 71.21 | V, SignRep T4 |
| label-free SSL (+teacher), project readouts | RGB | SignRep repro / bins2 / chamfer / H008 distilled | 49.02 / 54.93 / 58.42 / 57.28 | 70.42 (repro) | I, C011, C012, C019 |
| Kinetics labels | RGB | I3D-Kinetics features | 0.41 | 12.34 | V\*, SignMatch T5 (from ASL Citizen paper) |
| text (Spreadthesign) | pose | SignCLIP | 4.00 | 21.80 | V, SignSeek T2 |
| other-dataset glosses | RGB | I3D-WLASL features | 9.85 | 31.53 | V\*, SignMatch T5 |
| glosses (Sem-Lex, AUTSL, SLR500, MSASL subset, MeinDGS) | pose | **SignSeek** | **57.66** | 77.21 | V, SignSeek T2 |
| BSL+ASL dictionary + pseudo-labels (no ASL Citizen) | pose / RGB (SignRep init) | SignMatch | 57.32 / **67.12** | 75.88 / 83.17 | V, SignMatch T5 |
| in-domain ASL Citizen glosses | RGB | I3D trained on ASL Citizen | 58.03 | 76.52 | V\*, SignMatch T5 |

**Other frozen protocols**

| Protocol | Supervision | Modality | Method | Score | Tag |
| --- | --- | --- | --- | --- | --- |
| WLASL2000 F-R R@1 | none | 3D pose | Hand joint angles | 9.42 | V, SignRep T4 |
| WLASL2000 F-R R@1 | label-free SSL | RGB | SignRep avg / weighted | 26.16 / 29.92 | V |
| WLASL2000 F-R R@1 | label-free SSL | pose | MASA † (WLASL unlabelled in pretraining: transductive) | 14.38 | V\* |
| WLASL2000 F-R R@1 | gloss-supervised | pose | SignSeek | 34.75 | V |
| WLASL2000 F-R R@1 | label-free (project) | RGB | H008 on SignRep | 32.55 | I, C020 |
| NMFs-CSL F-R R@1 | label-free SSL / gloss-supervised / text | RGB / pose / pose | SignRep 63.04; SignSeek 65.20; MASA † 54.69; SignCLIP 26.75 | – | V |
| BSL SignBank zero-shot F-R R@1 | label-free / label-free / gloss | RGB / pose / pose | **SignRep 4.30 < MASA 6.80** < SignSeek 12.76 | – | V\*, SignSeek T3 |
| WLASL2000 linear, P-I | label-free SSL | pose | SignBERT 18.24; BEST 21.19; SignBERT+ 24.50; STC 32.35 | – | V, STC "linear 100%" rows |
| MSASL1000 linear, P-I | label-free SSL | pose | 31.54 / 30.89 / 34.44 / 48.30 (same order) | – | V |
| WLASL2000 frozen + trained classifier | RSL gloss labels | RGB | Logos | 62.44 | V, Logos T2 |
| ASL Citizen-500 linear | label-free SSL | pose | SSL-SLR | 20.46 | V, T1 |
| ASL Citizen linear R@1 | text | pose | SignCLIP frozen+linear | 0.23 | V-L4 |
| ASL Citizen linear R@1 | label-free SSL | DINOv3 crops | SignDino frozen+linear (unreviewed, flagged) | 0.63 | V (signdino.txt) |
| ASL Citizen val, cosine linear on frozen SignRep | label-free + val labels for the head | RGB | project | 66.1 | I, C017 |
| Sem-Lex unseen-gloss MLP probe MRR (OOV) | label-free / phonology-semi-supervised | pose | VQ-VAE .381; PD .372; PSS .435; full .452 | – | V, 2509.04745 T3 |

**Not frozen; listed only so they are not mixed up with the above.** All are ASL Citizen R@1 / top-1:

| Method | Modality | Training | Score |
| --- | --- | --- | --- |
| SignRep | RGB | fine-tuned | 81.37 |
| SignSeek | pose | fine-tuned | 74.7 [V, SignSeek T6] |
| PoseFormer | pose | supervised | 75.1 [V-L4] |
| Bootstrapping | pose | supervised, split caveat | 74 |
| OneSign | pose | supervised | 65.41 / 68.41 |
| SHuBERT | crops + body | LoRA | 65 [V\*, Bootstrapping T3] |
| SSL-SLR | pose | fine-tuned | 47.06 |
| Uni-Sign | pose | reproduced | 60.92 |

**Take-away.**
- On frozen dictionary retrieval, the label-free **skeleton** best is still **27.96** (MASA, small data).
- Label-free RGB SignRep is **49.95** (≈ 57–58 with better readouts [I]).
- Supervised pose sits at **57.3–57.7** without in-domain labels.
- Nobody has published a label-free skeleton model pretrained at YT-SL-25 scale.

---

## 4. Open gaps for a label-free skeleton model (evidence → falsifiable test)

Each test assumes frozen evaluation with the same readouts (mean, activity- or motion-weighted, bins2, chamfer) for every encoder, development on ASL Citizen **val** (test quarantined [I]), ≥ 3 seeds, and signer-clustered bootstrap CIs.

1. **Scale gap.** Label-free skeleton SSL has never been pretrained on the SignRep corpus.
   - *Evidence:* MASA 27.96 used 146K ISLR clips. SignRep used ~3,000 h. In-domain data alone gave +5.7 fine-tuned (Table 6). Text-supervised Sigma shows a monotonic 23 → 64.5 scale curve. SignVerse-2M provides DWPose for YT-SL-25.
   - *Test:* pretrain the same skeleton objective on 10 / 30 / 100% of YT-SL-25 poses (WLASL/MSASL IDs removed).
   - *Falsified if:* ASL Citizen-val F-R R@1 does not rise monotonically with scale, or 100% stays below MASA + 10 points.
2. **Target gap.** Relational/invariant targets vs coordinates vs latents.
   - *Evidence:* SignRep's distance (47.2) and angle (45.1) targets ≫ keypoint targets (32.9); pose MAE reconstruction methods give 18–25 linear on WLASL; OpenHands DPC beat masked reconstruction [V-L4]; SignSeek's latent MAP adds +7.5 over GAC [V].
   - *Test:* same skeleton encoder, data and budget, varying only the target: (a) masked coordinates; (b) SignRep-style priors (wrist-relative hand angles + fingertip and hand-body displacements) computed from the input pose; (c) EMA latent targets (JEPA); (d) b + c.
   - *Falsified if:* (c) ≤ (b) on F-R and minimal pairs. That would mean the latent objective adds nothing over cheap hand-crafted relational targets.
3. **Readout and order gap.** SignRep's representation is a bag of windows.
   - *Evidence:* reversal cosine 0.959; order-aware readouts add +5.9 to +9.4 R@1 [I]; every surveyed method uses order-insensitive readouts [I C007].
   - *Test:* the skeleton model's own sign-level embedding (pooled after a temporal encoder over the whole clip) vs mean-of-windows. Report the reversal cosine and the bins2 − mean delta.
   - *Falsified if:* the native embedding's reversal cosine stays ≥ 0.9 **and** training-free order readouts still add ≥ 3 R@1. Then the model has not internalised order.
4. **Part-structure gap, with Kezar's warning.**
   - *Evidence:* SignRep decodes everything from one vector. Label-free part disentanglement **did not** help OOV recognition (.381 → .372), while phonology supervision did. Gloss embeddings are phonologically coherent only at small K (Vandendriessche).
   - *Test:* per-articulator tokens (dominant hand, non-dominant hand, hand-body relation, face) with frozen 16 ASL-LEX linear heads on a **lexically held-out** split (Sem-Lex unseen-gloss or an ASL Citizen gloss-disjoint split), plus the multi-hot kNN-vs-K probe.
   - *Falsified if:* part tokens ≤ the pooled vector on handshape (dominant-hand token) and location (relation token). If so, drop the part-structure claim.
5. **Location gap of skeletons.**
   - *Evidence:* STGCN loses to I3D on location minimal pairs (67.74 vs 80.65 % wins) because position-normalised graphs lose frame-relative location. SignRep keeps hand-to-face and body displacements as targets.
   - *Test:* the Yin et al. 259-pair ASL Citizen and 178-pair Sem-Lex minimal-pair protocol on frozen skeleton SSL vs SignRep, ablating the hand-body relation input or token.
   - *Prediction:* removing relation features lowers location wins but not handshape wins.
   - *Falsified if:* location wins are already ≥ SignRep's without relation features, or do not change when they are removed.
6. **Signer-invariance gap.**
   - *Evidence:* frozen SignRep nearest neighbours are the same signer 48.7% vs same gloss 43.8% [I]; pose identifies signers at 80% [V-L4]; SignRep's bone-length normalisation applies only to *targets*.
   - *Test:* bone-length-normalised *inputs* (median-skeleton retargeting via angles, as in SignRep A1) vs raw inputs. Measure frozen signer-ID linear accuracy (lower is better), same-signer NN rate, and F-R R@1.
   - *Falsified if:* signer-ID accuracy does not drop by ≥ 10 points, or R@1 drops by > 1 point.
7. **Handedness gap.**
   - *Evidence:* +5 R@1 from dominant-hand canonicalisation in supervised pose ISR (Bootstrapping, split caveat); SignSeek's shared mirrored hand encoder; reliable label-free handedness from hand height (sign-lang 2026); SignRep has none of this.
   - *Test:* label-free dominant/non-dominant canonicalisation (per-clip activity or height heuristic, mirroring) vs L/R inputs.
   - *Falsified if:* the gain is < 1 R@1 on ASL Citizen val or is concentrated in left-handed signers only.
8. **Non-manual gap.**
   - *Evidence:* SignRep is "solely manual" by design. The mouth is the most informative facial region (2507.20884). Uni-Sign's face sub-pose adds +1.89 from scratch on MSASL1000 [V-L4]. Non-manuals generalise poorly ("Lost in Expression").
   - *Test:* face/mouth keypoint stream on vs off, measured on NMFs-CSL F-R (non-manual-heavy) and ASL Citizen.
   - *Falsified if:* NMFs-CSL F-R does not improve by ≥ 2 R@1.
9. **Cross-domain robustness, where skeletons may already win.**
   - *Evidence:* zero-shot BSL SignBank retrieval has MASA (pose, label-free) 6.80 > SignRep 4.30. Phonological sensitivity halves out of domain (Sem-Lex).
   - *Test:* report zero-shot BSL SignBank (726 queries), Sem-Lex minimal pairs and an extractor swap (DWPose ↔ MediaPipe), alongside in-domain numbers.
   - *Falsified if:* the skeleton model trails SignRep on these as well. Then the "complement" story fails.
10. **Keypoint-noise gap.**
    - *Evidence:* hand AP is only 58–66 on COCO-WholeBody, and MediaPipe drops hands in 20–24% of frames [V-L4]. SignRep sidesteps this at inference.
    - *Test:* σ-jitter and whole-hand dropout sweeps at test time, comparing JEPA-latent vs coordinate targets.
    - *Prediction:* latent targets degrade less.
    - *Falsified if:* the degradation slopes are equal.
11. **Evaluation-protocol gap** (not a method gap but needed for a defensible claim).
    - *Evidence:*
      - No paper reports frozen retrieval with a lexically held-out gallery.
      - SignRep selected its checkpoint on WLASL val.
      - SSL-SLR mixes fine-tuned and frozen numbers.
      - The Bootstrapping split may be non-official.
      - SignSeek's Sem-Lex pretraining shares the ASL-LEX vocabulary with ASL Citizen.
    - *Test:* a fixed protocol card covering:
      - official splits;
      - train gallery with class-max cosine;
      - avg and weighted readouts both reported;
      - minimal pairs;
      - frozen ASL-LEX probes on a gloss-disjoint split;
      - signer-ID probe;
      - zero-shot BSL.
      - Every comparator is re-run under it.

---

## 5. Corrections and additions to L4

1. **"SLim (NeurIPS 2026) could not be identified" → identified.** It is **SLiM**, Do, Chen, Youk & Kim (KAIST), arXiv 2603.10648, "Less is More: Compact-Token Masked Feature Prediction for Skeleton Representation Learning". The NeurIPS 2026 catalogue lists the title as "…Feature Learning…". It is general skeleton SSL with anatomical tube masking, not sign-specific. It is already read and code-audited in `docs/proposal12/research_update_20261011/LITERATURE_AND_CODE.md` [R06].
2. **SSL-SLR protocol resolved.** Its ASL Citizen 47.06 is **fine-tuned**, compared against SignRep's frozen retrieval, and the paper misstates SignRep's 49.95 as a fine-tuning result. Its frozen number is the **linear 20.46 on a 500-sign ASL Citizen subset**. Venue: TMLR per Semantic Scholar [S]. The arXiv v2 title is "SSL-SLR"; the Semantic Scholar title is "SL-SLR".
3. **SignRep keypoint targets are reconstructed from angles with YT-SL median bone lengths** (A1), not just "bone lengths normalised". Distance priors are 3-component displacement arrays (5×11×3, 12×22×3). The activity prior is regressed with smooth-L1, weight 0.2. The adversary is applied only when E_M > E_U, and the discriminator uses EMA momentum 0.1, style scaling ×100 and Adam betas (0.5, 0.9).
4. **SignRep version:** only arXiv v1 exists. The ICCV camera-ready (CVF PDF and supplement) is numerically identical, so L4's v1 citations hold for ICCV.
5. **Geo-Sign WLASL2000 63.64/61.89:** upgrade from [V\*] to [V]. It is in Geo-Sign's own Table 3 (arXiv 2506.00129). Venue NeurIPS 2025 [S].
6. **SignSeek MAC is also label-supervised.** It aligns the dominant-only query with *same-gloss* references. So the no-GAC variant (MAC+MAP: 55.12 R@1 on ASL Citizen, Table 7) is not label-free. **MAP alone was never reported.** SignSeek's retrieval rule is class-max cosine over gallery instances (§4 "Retrieval protocol"), consistent with SignRep's top-1 rule.
7. **Additional SignSeek facts L4 lacked:**
   - Zero-shot BSL SignBank (Table 3): SignRep 4.30 < MASA 6.80 < SignSeek 12.76 R@1.
   - Parameter counts: SignRep 51.5 M, SignSeek 15.7 M.
   - Fine-tuned ASL Citizen 74.7 and NMFs-CSL 73.8 (Table 6).
8. **Missing from L4, now carded:**
   - Sigma (pose, text-supervised, WLASL 64.54).
   - Kezar et al. 2025 VQ-ASL (label-free part disentanglement does not help OOV recognition).
   - Yin et al. 2026 minimal-pair protocol (pose wins handshape and movement, RGB wins location; released data).
   - SignMatch (pose 57.32 / RGB 67.12 on ASL Citizen retrieval, supervised).
   - OneSign (Uni-Sign reproduced at 60.92 on ASL Citizen).
   - Bootstrapping (+5 from handedness).
   - Vandendriessche et al. 2026 (kNN-vs-K phonology probe).
   - Logos (frozen supervised linear 62.44 on WLASL2000).
   - SignVerse-2M (DWPose YT-SL-25 corpus).
   - SignReGraph (metadata only).
9. **L4 §3b row "SignCLIP retrieval R@1 4.00" is correct** (SignSeek Table 2). Note that SSL-SLR cites a different SignCLIP ASL Citizen figure, 46.00, of unclear protocol. Do not use it.
10. **L4 §1C "SignCLIP fine-tuned + linear 0.60"** is corroborated by the Bootstrapping Table 3 (0.60) [V\*]. **SHuBERT 0.65** is corroborated there too.
11. **"No sign-language JEPA" still holds** as of 2026-10-14, across the arXiv API (`abs:sign AND (JEPA|joint-embedding)`), Semantic Scholar citations of SignRep, SHuBERT, Uni-Sign, MASA and SignCLIP, the sign-lang@LREC 2026 listing, and WebSearch. It remains a scoped claim, not a proof of absence.
