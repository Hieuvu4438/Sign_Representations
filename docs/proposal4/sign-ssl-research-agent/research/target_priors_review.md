# Target priors for JEPA-style sign-language SSL — focused review (H)

Date: 2026-10-07. Search log: `search_log_H.md` (queries H1–H42). Read depth: at most ABSTRACT_READ (abstract fetched from export.arxiv.org API); no full text read, no code audited. No numeric result from any paper is reproduced here (by design: WebFetch/snippet numbers are unreliable in this project).

Labels follow `.claude/rules/00-evidence.md`: FACT (from abstract / verified metadata), INFERENCE (agent reasoning), HYPOTHESIS (testable prediction), UNKNOWN. Anything in the per-source table that is not in the fetched abstract is marked "(snippet)" = DISCOVERED-level detail.

Question: beyond SignRep's pose priors (per-frame 3-D joint angles, 3-D keypoints, pairwise distances, hand-activity head; manual only — per task brief/project context, not re-verified here), which prediction targets could carry more sign-relevant signal in a JEPA-style objective, testable as continued pretraining on ASL Citizen with one GPU and a few GPU-hours per arm?

---

## 1. Per-source table

Columns: target type · what is predicted from what · teacher / supervision provenance · sign or human-action benchmarks named in the abstract (names only, no numbers).

### 1a. Sign-language SSL / pretraining

| ID | Work (authors, year, venue, id) | Read | Target type | Predicted from what | Teacher / supervision provenance | Sign / action evaluation named |
| --- | --- | --- | --- | --- | --- | --- |
| S01 | SignRep: Enhancing Self-Supervised Sign Representations — Wong, Camgoz, Bowden; 2025; ICCV 2025 (snippet: CVF open access); arXiv 2503.08529 | ABSTRACT_READ | masked autoencoding + skeleton-derived "sign priors" (keypoint, angle, distance, signer activity — categories from snippet; exact list per project context) + feature regularisation + adversarial style-agnostic loss | RGB (Hiera) masked input → priors via lightweight sign decoder; no keypoints at inference (FACT, abstract) | pose teacher (3-D pose estimator; trained on labelled pose data — provenance details UNKNOWN here) | WLASL, ASL-Citizen, NMFs-CSL (finetune); dictionary retrieval (frozen); Phoenix2014T, CSL-Daily, How2Sign (translation) |
| S02 | SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction — Gueuwou, Du, Shakhnarovich, Livescu, Liu; 2024/2025; ACL 2025; 2411.16765 | ABSTRACT_READ | discrete cluster IDs of hand, face and body-pose streams (HuBERT-style) | masked frame tokens of a multi-stream feature sequence → per-frame cluster assignment per stream (abstract); streams built from DINOv2 features of hand/face crops + body pose, k-means (snippet) | DINOv2 (SSL image model) + pose estimator; ≈1,000 h ASL video unlabelled | SLT, ISLR, fingerspelling detection (abstract); How2Sign / OpenASL / FLEURS-ASL (snippet) |
| S03 | SignDino: Self-Supervised Sign Language Representation Learning via Temporal-Axis Self-Distillation — Hu, He, Huang, Li, Li, Fang; 2026; arXiv 2609.06296 | ABSTRACT_READ | temporal DINO prototypes + iBOT frame-level masked-token prediction + KoLeo + Gram anchoring | short temporal crops of a tracked articulator stream (left hand, right hand, face) → EMA-teacher outputs on long crops; per-frame features from **frozen DINOv3 ViT-B/16** on crops | DINOv3 (SSL) + YOLOv8n/ByteTrack detector (detector trained on labelled boxes; details UNKNOWN) | translation, ISLR, fingerspelling detection (abstract); "phonological probing" (snippet) |
| S04 | SignMusketeers: An Efficient Multi-Stream Approach for Sign Language Translation at Scale — Gueuwou, Du, Shakhnarovich, Livescu; 2024/2025; ACL Findings 2025; 2406.06907 | ABSTRACT_READ | frame-level SSL (DINOv2-S recipe on face and hand crops — snippet) | single crop frames → DINO-style self-distillation; not a video target | self-supervised on crops; body pose from off-the-shelf estimator | How2Sign SLT |
| S05 | SignMAE: Segmentation-Driven Self-Supervised Learning for Sign Language Recognition — Xie, Cai, Stefanov; 2026; ICPR 2026; 2605.02094 | ABSTRACT_READ | pixel reconstruction with segmentation-based masking; hand-region-only reconstruction (snippet) | masked RGB video (and keypoint-heatmap stream, snippet) → hand-region tokens | segmentation / pose models (provenance UNKNOWN) | WLASL, NMFs-CSL, Slovo |
| S06 | Towards Privacy-Aware Sign Language Translation at Scale (SSVP-SLT) — Rust, Shi, Wang, Camgöz, Maillard; 2024; ACL 2024; 2402.09611 | ABSTRACT_READ | pixel MAE (SignHiera) on face-blurred video (snippet for architecture) | masked RGB → pixels | none for MAE stage; optional CLIP-style video-text stage | How2Sign SLT (finetuned + zero-shot) |
| S07 | Self-Supervised Video Transformers for Isolated Sign Language Recognition — Sandoval-Castaneda, Li, Brentari, Livescu, Shakhnarovich; 2023; WACV 2024 submission; 2309.02450 | ABSTRACT_READ | compares four video SSL methods; MaskFeat (HOG target) best in their study | — | — | WLASL2000 ISLR; linear probing of phonological features |
| S08 | SignBERT (ICCV 2021, 2110.05382) / SignBERT+ (TPAMI, 2305.04868) — Hu, Zhao, Zhou, (Wang), Li | ABSTRACT_READ | masked hand-pose token reconstruction with hand-model-aware (MANO, snippet) decoder; joint/frame/clip masking | 2-D hand pose sequence (off-the-shelf detector) → masked pose tokens | pose detector | NMFs-CSL, SLR500, MSASL, WLASL (SignBERT); ISLR, CSLR, SLT (SignBERT+) |
| S09 | BEST: BERT Pre-Training for Sign Language Recognition with Coupling Tokenization — Zhao, Hu, Zhou, Shi, Li; 2023; AAAI 2023; 2302.05075 | ABSTRACT_READ | discrete pseudo-labels of pose triplet units (hand, hand, body) via coupling tokenization | masked pose triplet units → token classification | pose detector | four SLR benchmarks (names not in abstract) |
| S10 | MASA: Motion-aware Masked Autoencoder with Semantic Alignment for SLR — Zhao, Hu, Zhou, Mao, Wang, Li; 2024; TCSVT 2024; 2405.20666 | ABSTRACT_READ | **motion residuals** of masked frames + momentum (EMA) semantic alignment of augmented views | pose sequence (motion-aware masking) → masked-frame motion residuals | pose detector | four public SLR benchmarks (names not in abstract). Related, same group: Spatial-Temporal Consistency SSL (TIP, 2406.10501, ABSTRACT_READ) — first-order motion modality in contrastive pose SSL |
| S11 | Uni-Sign: Toward Unified Sign Language Understanding at Scale — Li, Zhou, Zhao, Wu, Hu, Li; 2025; ICLR 2025; 2501.15187 | ABSTRACT_READ | generative (text) pretraining, pose+RGB fusion | video/pose → text | **text-supervised** (CSL-News video–text pairs) — not SSL | ISLR, CSLR, SLT |
| S12 | M³-SLR: Self-Supervised Pretraining with MaxFlow MaskFeat for Multi-View Sign Language Representation — authors UNKNOWN; 2025; IEEE (doc 11132300) | DISCOVERED | HOG (MaskFeat) target, **optical-flow-guided cube masking** | masked RGB → HOG | flow estimator (which one UNKNOWN) | Multi-VSL200, MM-WLAuslan (snippet) |
| S13 | Improving Sign Language Understanding with a Multi-Stream Masked Autoencoder Trained on ASL Videos (MS-MAE) — Mo, Vo, Nakayama; 2025; IJCNLP-AACL 2025 | DISCOVERED | skeleton-based multi-stream MAE | pose → masked pose | pose detector | ISLR + SLT, several sign languages (snippet) |
| S14 | Phonological Representation Learning for Isolated Signs Improves OOV Generalization — Kezar, Sehyr, Thomason; 2025; 2509.04745 | ABSTRACT_READ | VQ autoencoder with parameter disentanglement + phonological semi-supervision | isolated sign (modality per full text, UNKNOWN) → reconstruction; phonology labels as regulariser | **phonology annotations (labelled)** | isolated sign identification; one-shot reconstruction of unseen signs. Related: 2609.18772 zero-shot handshape (5 phonological features; ABSTRACT_READ); Kezar et al. EACL 2023 phonology auxiliary (DISCOVERED) |

### 1b. Generic masked video / image targets

| ID | Work | Read | Target type | Predicted from what | Teacher / provenance | Action evaluation named |
| --- | --- | --- | --- | --- | --- | --- |
| S15 | VideoMAE — Tong, Song, Wang, Wang; 2022; NeurIPS 2022; 2203.12602 | ABSTRACT_READ | pixels, tube masking, very high ratio | masked tubes → pixels | none | K400, SSv2, UCF101, HMDB51 |
| S16 | MaskFeat — Wei, Fan, Xie, Wu, Yuille, Feichtenhofer; 2021; 2112.09133 | ABSTRACT_READ | **HOG** of masked regions (five feature types studied; local contrast normalisation essential) | masked video → HOG per masked patch | none | K400/600/700, AVA, SSv2 |
| S17 | MotionMAE — Yang, Huang, Wen, Wu, Yao, Jiang et al.; 2022; 2210.04154 | ABSTRACT_READ | pixels + **temporal frame difference** of nearby frames | masked patches → appearance and motion per masked patch | none | SSv2, UCF101, video object segmentation |
| S18 | MAM2 (It Takes Two) — Song, Yang, Wu, He, Li, Wang; 2022; 2210.05234 | ABSTRACT_READ | VQGAN codes (appearance) + **RGB difference** (motion), two decoders; RGB-difference found "simple yet effective" among motion targets | masked video → both targets | VQGAN tokenizer | K400, SSv2, UCF101, HMDB51 |
| S19 | MME: Masked Motion Encoding — Sun, Chen, Chen, Li, Li, Tan et al.; 2022; CVPR 2023; 2210.06096 | ABSTRACT_READ | **dense motion trajectories** (position + shape change) in masked regions | sparse masked input → dense trajectories | trajectory extractor (details UNKNOWN) | action datasets (names not in abstract) |
| S20 | MGMAE — Huang, Zhao, Zhang, Qiao, Wang; 2023; ICCV 2023; 2308.10794 | ABSTRACT_READ | pixels; **optical flow used for masking** (temporally consistent masking volume), not as target | — | online flow estimator (RAFT, snippet) | SSv2, K400 |
| S21 | TrackMAE — Vandeghen, Thoker, Van Droogenbroeck, Ghanem; 2026; CVPR 2026; 2603.27268 | ABSTRACT_READ | pixel or feature target + **point-track displacements** (trajectory decoder) + motion-aware masking | masked video → track displacements | off-the-shelf point tracker (CoTracker3, snippet) | six datasets (names not in abstract) |
| S22 | MVD: Masked Video Distillation — Wang, Chen, Wu, Chen, Dai, Liu et al.; 2022; CVPR 2023; 2212.04500 | ABSTRACT_READ | **features of MIM-pretrained image and video teachers** (spatial-temporal co-teaching) | masked video → teacher features of masked patches | teachers themselves SSL (MAE / VideoMAE) | K400, SSv2, AVA (abstract) |
| S23 | UMT: Unmasked Teacher — Li, Wang, Li, Wang, He, Wang et al.; 2023; ICCV 2023; 2303.16058 | ABSTRACT_READ | align **unmasked** tokens with image foundation model (CLIP, snippet) features | visible tokens → teacher tokens | CLIP (image–text supervised) | scene/temporal/video-language tasks |
| S24 | SMILE — Thoker, Jiang, Zhao, Ghanem; 2025; CVPR 2025; 2504.00527 | ABSTRACT_READ | **CLIP features** as target + synthetic motion overlays | masked video → CLIP features | CLIP (image–text) | 7 datasets (names not in abstract) |
| S25 | BEiT v2 (2208.06366) — Peng, Dong, Bao, Ye, Wei; MILAN (2208.06049) — Hou, Sun, Chen, Xie, Kung; EVA (2211.07636) — Fang, Wang, Xie, Sun, Wu, Wang et al.; 2022 | ABSTRACT_READ | semantic tokens from VQ-KD tokenizer (BEiT v2); caption-supervised features (MILAN); image–text-aligned features (EVA) | masked image → teacher token/feature | language-supervised teachers (CLIP-type) | image tasks (EVA abstract also lists video action recognition) |
| S26 | data2vec — Baevski, Hsu, Xu, Babu, Gu, Auli; 2022; ICML 2022; 2202.03555 | ABSTRACT_READ | **contextualised EMA-teacher latents** of the full input | masked view → teacher latents | self (EMA) | speech / vision / language |
| S27 | I-JEPA (2301.08243, ICCV 2023); V-JEPA (2404.08471); V-JEPA 2 (2506.09985) — Assran / Bardes et al. | ABSTRACT_READ | EMA-teacher latents of target blocks (I-JEPA); masked spatio-temporal feature prediction (V-JEPA, V-JEPA 2) | context → target-block latents | self (EMA); V-JEPA: no pretrained image encoders, text or reconstruction (abstract) | K400, SSv2, ImageNet (V-JEPA); SSv2, Epic-Kitchens anticipation (V-JEPA 2) |
| S28 | V-JEPA 2.1 — Mur-Labadia, Muckley, Bar, Assran, Sinha, Rabbat et al.; 2026; 2603.14482 | ABSTRACT_READ | dense predictive loss on **visible and masked** tokens + deep self-supervision at intermediate layers | context → latents at all tokens / layers | self | Ego4D, EPIC-KITCHENS, SSv2, NYUv2 depth probe, robotics |
| S29 | SALT: Rethinking JEPA with Frozen Teachers — Li, Huang, Li, Malach, Susskind, Thilak et al.; 2025; ICLR 2026 (snippet); 2509.24317 | ABSTRACT_READ | **frozen teacher latents** (teacher = pixel-MAE under V-JEPA masking) | masked regions → frozen teacher latents; "student quality robust to teacher quality" (abstract) | self (two-stage) | frozen-backbone benchmarks (names not in abstract) |
| S30 | MotionJEPA — Karmann, Li, Internò, Andreis, Klindt, Balestriero et al.; 2026; 2609.23881 | ABSTRACT_READ | regulariser predicting **temporal difference-image embeddings** (DISReg) — no pixel loss | inverse-dynamics-style module → difference embeddings | self | latent probing; planning under distractors (not action recognition) |
| S31 | Factorized Latent Dynamics for Video JEPA — Premi; 2026; 2605.17165 | ABSTRACT_READ | 18 auxiliary JEPA objectives (kinematic penalties on temporal differences, motion-guided masking, SIGReg, physics priors); appearance/dynamics factorised subspaces | — | self | Diving-48, SSv2, ImageNet-100; abstract reports capability trade-offs across auxiliaries |
| S32 | Human-JEPA — Wei, Sun, Zhao; 2026; 2608.21160 | ABSTRACT_READ | anchored forecasting: dense targets pinned to a **frozen copy of the initialisation** (V-JEPA 2.1 init, snippet), past→future masks | past frames → future / dense latents | self + frozen init | pose, person re-identification, anticipation (frozen probes) |
| S33 | Skeleton latent/feature prediction: S-JEPA (ECCV 2024; no arXiv id found) — DISCOVERED; GFP 2509.03609 (ICCV 2025) and SLiM 2603.10648 — ABSTRACT_READ | mixed | latent of masked joints (S-JEPA, snippet, with centering); learned high-level feature targets instead of coordinates (GFP, SLiM) | skeleton → latent targets | self | NTU60/120, PKU-MMD |
| S34 | MultiMAE — Bachmann, Mizrahi, Atanov, Zamir; 2022; ECCV 2022 (snippet); 2204.01678 | ABSTRACT_READ | RGB + **depth + semantic segmentation** as inputs/targets, all from pseudo-labels | any visible subset → all modalities | pseudo-label networks (supervised) | image classification, segmentation, depth |
| S35 | Sapiens (2408.12569, ECCV 2024) / Sapiens2 (2604.21681, ICLR 2026) — Khirodkar et al. | ABSTRACT_READ | (target generator, not SSL-target paper) 2-D pose, **body-part segmentation**, depth, normals; Sapiens part vocabulary includes lips/teeth/tongue (snippet, v1) | — | supervised task heads on human-image SSL backbone | human-centric benchmarks |
| S36 | HuC-VideoMAE — Pizarro, Valle, Buenaposada, Bergasa, Baumela; 2026; 2610.08433 | ABSTRACT_READ | pixels with **keypoint/person-box-driven masking** on synthetic human video | — | synthetic ground truth keypoints | NTU RGB+D, Toyota-Smarthome |
| S37 | SignMouth (abstract calls the method "SignClip") — Wu, Yuan, Li, Wang, Fu; 2025; 2509.10266 | ABSTRACT_READ | supervised SLT with lip-movement features + hierarchical contrastive alignment (not SSL) | — | text-supervised | PHOENIX14T, How2Sign |
| S38 | Seeing in 2D, Thinking in 3D: 3D Hand Mesh-Guided Feature Learning — Papadimitriou et al. (author list UNKNOWN beyond first); ICCVW 2025 (CV4A11y) | DISCOVERED | distil HaMeR (MANO hand mesh) knowledge into RGB fingerspelling recogniser (snippet) | RGB → HaMeR-derived targets | HaMeR (supervised 3-D hand) | fingerspelling (snippet) |

---

## 2. What the search shows (FACT vs INFERENCE)

FACT (abstract level):
- Generic video SSL has moved from pixels to (a) hand-crafted appearance targets (HOG, S16), (b) motion targets — frame differences (S17, S18), dense trajectories (S19), point tracks (S21) — and (c) frozen/EMA teacher latents (S22–S29). Motion targets are reported to help motion-centric benchmarks (S17, S21 abstracts).
- In JEPA specifically, S30 states that standard JEPA training has a bias toward slow features and adds a difference-embedding regulariser; S31 reports that auxiliary objectives on Video-JEPA trade one capability against another.
- S29 states a frozen (pixel-MAE) teacher suffices for masked latent prediction and that student quality is robust to teacher quality.
- Sign-specific SSL targets found: skeleton priors on RGB (S01); clusters of DINOv2 hand/face/pose features (S02); temporal self-distillation over frozen DINOv3 crop features (S03); pixel MAE (S06, S05 with hand-region targets); HOG (S07, S12); pose-token / motion-residual targets on pose input (S08–S10, S13).
- Only one sign work in scope uses an explicit **motion** target (S10 MASA) and it operates on pose input, not RGB.
- No sign-language JEPA / V-JEPA pretraining was found (H13, H41).

INFERENCE:
- SignRep's priors are per-frame static descriptors (angles, keypoints, distances, activity); none listed is a temporal derivative. Movement therefore enters SignRep only implicitly through masking. This is the clearest gap relative to the phonological parameters (handshape, location, movement, orientation, non-manual).
- Several high-performing generic targets (CLIP/DINO features) carry appearance and identity information. The project's observation that same-signer gallery clips act as distractors (research/MEMORY.md; WLASL removal check) suggests signer appearance already competes with sign content in retrieval, so targets that are appearance-free (flow, keypoint-derived, part maps) or discretised (clusters) are preferable for cross-signer retrieval. This is a hypothesis about the evaluation protocol, not a result.

---

## 3. Ranked shortlist for a JEPA-style continued-pretraining pilot (ASL Citizen, 1 GPU)

Shared setup (assumed, matches EX13 style): SignRep warm-start, JEPA masked-latent loss as the base arm (B0-like), each candidate added as an **auxiliary head on predictor outputs at masked tokens** (or on ROI-pooled tokens), same steps/seed/mask/data order; evaluation on validation only (leave-one-signer-out retrieval). Required control: base JEPA arm with matched steps, and for any learned-teacher target, a random/shuffled-target control. Cost lines are INFERENCE (order-of-magnitude, to be measured by a timing probe); nothing below was run.

Ranking criterion (INFERENCE): expected new sign-relevant information beyond SignRep priors × likelihood of being signer-invariant × low cost/risk within a few GPU-hours.

### R1 — Hand-ROI motion targets: optical flow / frame difference / keypoint velocity (movement)

- Rationale: movement is the phonological parameter least covered by per-frame priors (INFERENCE). Targets are largely appearance-free → plausibly signer-robust (HYPOTHESIS).
- Target computation: per masked token, (a) hand-ROI optical flow (OpenCV Farneback or DIS on hand crops from COCO-WholeBody hand boxes) pooled to patch grid, magnitude/direction histograms; (b) cheaper variant: 2-D velocity/acceleration of the 42 hand + wrist keypoints at lags matching the patch time-stride (TrackMAE-style "tracks", with keypoints as tracks); (c) MotionMAE-style RGB difference in hand ROIs. Normalise by shoulder width; mask joints below the calibrated confidence thresholds.
- Cost: (b) negligible (keypoints exist); (a) CPU flow on small crops, one-time pass; training overhead = one small head.
- Risk: keypoint jitter turns velocity into noise (needs temporal smoothing and confidence masking); frame differences include lighting flicker; overlap with ASC-JEPA B4 "change" term (latent temporal differences, EX13) — B4 predicts change in latent space, R1 predicts explicit kinematics; report both.
- Closest prior art: MotionMAE S17, MAM2 S18, MME S19, TrackMAE S21 (generic RGB); MASA S10 (sign, pose input); MotionJEPA S30 (latent difference regulariser); S31 kinematic penalties.
- Novelty (scoped): RGB sign encoder with explicit hand-motion (flow / track / keypoint-velocity) targets in a JEPA objective — not found in searches H1–H42 on 2026-10-07. Mechanism is not new in generic video (S17–S21).

### R2 — Hand-crop frozen-teacher features, discretised (handshape, orientation)

- Rationale: 21 hand keypoints under-describe handshape/orientation (finger contact, palm facing, occlusion); a frozen image teacher on hand crops captures appearance-level shape (INFERENCE). SHuBERT and SignDino both use hand-crop SSL features as targets/inputs (FACT, S02/S03).
- Target computation: crop both hands from COCO-WholeBody hand boxes (fixed scale, chirality kept, no flip), DINOv2 (ViT-S/B) CLS or mean-patch per frame; k-means (SHuBERT-style, codebook fit on train only) → cluster ID per hand per frame; predict from predictor output ROI-pooled at the hand location. Continuous variant: cosine regression to whitened features.
- Cost: one-time DINOv2 pass over 2 crops/frame on subsampled frames (moderate, likely the largest preprocessing item after R4); clustering cheap.
- Risk: DINOv2 crop features encode skin tone, lighting, sleeves → signer leakage (HYPOTHESIS; check by signer-ID linear probe on targets before training). Discretisation should reduce it (INFERENCE). Detector failure on fast motion/blur.
- Closest prior art: SHuBERT S02 (cluster targets, but feature-sequence encoder, not RGB pixels); SignDino S03; SignMusketeers S04; MVD/UMT S22/S23 (generic teacher features).
- Novelty (scoped): RGB video encoder predicting hand-crop DINOv2 cluster targets at hand-ROI tokens inside a JEPA objective — not found in H1–H42 on 2026-10-07. Close to SHuBERT's target definition; contribution would be the setting, not the target type.

### R3 — Hand-ROI HOG (MaskFeat-style; handshape edges, orientation)

- Rationale: HOG was the strongest SSL target for ISLR in S07 (FACT, abstract); local contrast normalisation removes much illumination/skin-tone variation (INFERENCE). Very cheap control for R2 ("does any appearance target on hands help, or only a semantic teacher?").
- Target computation: OpenCV/skimage HOG on hand crops (or on full-frame patches restricted to hand ROI tokens), per masked token.
- Cost: negligible (CPU, can be on-the-fly).
- Risk: low-level target may pull a warm-started semantic encoder toward texture (S22 reports teacher-feature targets beat low-level ones in generic video — INFERENCE from abstract framing); low novelty.
- Closest prior art: MaskFeat S16, S07, M³-SLR S12.
- Novelty (scoped): HOG as a JEPA auxiliary restricted to hand ROIs — not found in H1–H42; low novelty, value is as a control.

### R4 — Body-part segmentation maps (location / contact, partially non-manual)

- Rationale: sign location is defined relative to body landmarks (face, chin, chest, non-dominant hand). A part map gives *which body part lies under/next to the hand*, including occluding contact that 2-D keypoint distances miss (INFERENCE). Appearance-free target. Sapiens part vocabulary includes lips/teeth/tongue (snippet, v1) → weak mouthing signal.
- Target computation: Sapiens2 0.4B part segmentation on subsampled frames at reduced resolution; per patch, class histogram (soft label); loss = cross-entropy at masked tokens, optionally only near hand ROIs. Verify Sapiens2 class vocabulary locally (UNKNOWN).
- Cost: highest preprocessing cost in the list (0.4B model per frame) — subsample frames (e.g. every 4th) and cache; training overhead small.
- Risk: segmentation errors on fast/blurred hands; target may mostly encode static body layout (easy, uninformative) unless restricted to hand neighbourhoods.
- Closest prior art: MultiMAE S34 (segmentation pseudo-labels as MAE target, images); SignMAE S05 (segmentation for *masking*); HuC-VideoMAE S36 (keypoint masking).
- Novelty (scoped): body-part-segmentation targets for sign SSL — not found in H1–H42 on 2026-10-07.

### R5 — Pose-derived phonological pseudo-tokens (handshape / location / movement clusters)

- Rationale: discrete, parameter-factorised targets (handshape cluster from normalised hand joint angles; location bin of wrist relative to face/shoulders; movement direction class) — robust to keypoint noise, explicitly aligned with phonological parameters (INFERENCE). Kezar et al. S14 report that phonological inductive biases help generalisation (with labelled phonology; here labels would be unsupervised clusters).
- Target computation: from COCO-WholeBody 2-D keypoints only; k-means / fixed bins fit on train; three classification heads.
- Cost: negligible.
- Risk: largely redundant with SignRep's continuous angle/distance priors (the warm-start already encodes them); 2-D hand keypoints from COCO-WholeBody may be noisy, giving unstable handshape clusters.
- Closest prior art: BEST S09 (pose triplet tokenisation), SHuBERT S02 (body-pose clusters), S14 (phonology-structured VQ).
- Novelty (scoped): factorised location/movement pseudo-tokens as RGB SSL targets — not found in H1–H42; handshape clustering itself is close to BEST/SHuBERT.

### R6 — Non-manual geometry targets (face keypoints: mouth aperture/shape, brow height, head pose)

- Rationale: SignRep explicitly excludes non-manual priors; non-manuals carry lexical and grammatical information (S37 framing; project rule 10). Using geometry rather than face appearance avoids identity leakage (INFERENCE).
- Target computation: from the 68 COCO-WholeBody face keypoints: normalised mouth width/height, lip-corner distances, inner-brow-to-eye distances, coarse head yaw/pitch proxy; per frame, predicted at face-ROI tokens.
- Cost: negligible.
- Risk: for isolated-sign retrieval on ASL Citizen, non-manuals may be a small share of discriminative signal (HYPOTHESIS → expected small effect, possibly null on the primary metric); face keypoint noise at the resolution used. Better tested with a phonology/non-manual probe as secondary metric.
- Closest prior art: SHuBERT/SignDino face streams (appearance features, S02/S03); SignMusketeers face encoder S04; SignMouth S37 (supervised).
- Novelty (scoped): face-geometry non-manual regression targets in RGB sign SSL — not found in H1–H42 on 2026-10-07.

### R7 — Frozen generic video teacher latents (V-JEPA 2.1 ViT-B; SALT/MVD-style) — control arm

- Rationale: separates "any good frozen teacher helps" from "sign-specific target helps". S29 reports frozen teachers suffice and student quality is robust to teacher quality (FACT, abstract); S22 reports image vs video teachers transfer differently.
- Target computation: frozen V-JEPA 2.1 ViT-B features (selected layer) on the same clips; align token grids (Hiera vs ViT patch geometry) by interpolation; cosine/L1 at masked tokens.
- Cost: teacher forward per step (online) or cached features (memory-heavy); moderate.
- Risk: generic features may pull the encoder toward scene/appearance and away from SignRep's sign-specific space (could degrade retrieval); token-grid mismatch.
- Closest prior art: SALT S29, MVD S22, UMT S23, Human-JEPA S32 (frozen-init anchoring).
- Novelty: none as mechanism; not found applied to sign in H1–H42. Include as control, not as a contribution.

### Not shortlisted (and why)

- MANO / HaMeR hand-mesh targets (S08, S38): HaMeR not in the assumed local asset list; per-frame mesh recovery cost is high. Strong candidate if HaMeR becomes available (handshape + 3-D orientation).
- CLIP/text-aligned feature targets (S23–S25): add language supervision; CLIP on sign frames is unlikely to encode handshape (INFERENCE) and would change the supervision setting.
- Depth targets: physically relevant (hand in front of body, contact) but no local depth model assumed; Sapiens2 abstract lists pointmap/normal tasks — availability of those heads locally UNKNOWN.

### Suggested pilot ordering (INFERENCE)

1. Cheapest discriminating trio first: base JEPA (matched steps) vs +R1(b keypoint velocity) vs +R3 (hand HOG) vs +R6 (face geometry) — all negligible preprocessing.
2. If R1 or R3 > base: add R1(a) flow and R2 (DINOv2 hand clusters, after a signer-ID leakage probe on the targets).
3. R4 only if location-related confusions dominate the error analysis; R7 as the "generic teacher" control whenever a learned-teacher arm (R2) is reported.
Each pilot is screening-only (one seed, validation); confirmatory claims need the multi-seed protocol rules.

---

## 4. Blind spots and limitations of this review

- Read depth: abstracts only; method details marked "(snippet)" are DISCOVERED-level and may be wrong. SignRep's exact prior list is taken from the task brief/project context, not re-read here.
- Not found ≠ absent: search was WebSearch top results + arXiv abstracts, 2026-10-07; no Semantic Scholar/Google Scholar forward-citation snowballing of SHuBERT, SignRep or TrackMAE was done. Forward citations of SignRep and SHuBERT are the most likely place for missed sign-specific target papers.
- Name collision: "SignDino" (Hu et al., arXiv 2609.06296, S03) vs "SignDINO" (Gan et al., CVPR 2026, "Learning Effective Sign Features without Text for Gloss-free SLT", cited in the proposal8 spec) — distinct works; the CVPR one was not read here.
- S-JEPA, M³-SLR and MS-MAE have no arXiv abstract fetched (DISCOVERED only).
- Not searched: audio-visual / mouthing SSL from lip-reading (e.g. AV-HuBERT-type targets), egocentric hand-object SSL, event-camera sign SSL, and SignSeek-style articulator latent prediction (2609.03695, already in project logs; gloss-supervised).
- Cost estimates are not measured; a timing probe (flow, DINOv2 crops, Sapiens2) on a 200-clip subset should precede any arm.
- Supervision provenance: every shortlisted target depends on the COCO-WholeBody pose estimator at least for ROI placement (supervised on labelled keypoints); R2, R4 and R7 add a further pretrained teacher (DINOv2, Sapiens2 segmentation head, V-JEPA 2.1). Arms must declare this.
