# L4: Self-supervised and pretrained representation learning for sign language (pose-centric), October 2026

**Scope.** This review covers pose and keypoint SSL for sign language, the supervised pose ISLR state of the art, pretraining corpora and keypoint noise, phonology, and signer invariance. It ends with the gaps a **skeleton JEPA with anatomical masking** would target, and with evaluation protocol advice.

**Verification legend.**
- **[V]** The number was grepped from the paper's own arXiv HTML or PDF text. Raw files are in `scratchpad/litreview/raw/sl_*`, and text conversions in `scratchpad/litreview/l4/txt/*.txt`.
- **[V\*]** The number was grepped from a *citing* paper's table (secondary source).
- **[S]** The fact comes from search snippets only.
- **[K]** Background knowledge that I did not verify here.
- **[I]** An internal project record (`docs/proposal4/...`).

No number here comes from a WebFetch summary.

---

## 0. Headline findings

1. **No sign-language JEPA exists on arXiv as of 2026-10-11.**
   - An arXiv API query for `abs:"sign language" AND abs:JEPA` returned nothing. So did `abs:"predictive architecture"`.
   - The closest prior work comes in three forms:
     - OpenHands' Dense Predictive Coding on poses (2021).
     - SignSeek's latent masked-articulator prediction (Sep 2026). It is *supervised* contrastive plus a JEPA-like stop-grad cosine loss.
     - SignDino's temporal DINO/iBOT on frozen DINOv3 crop features (Sep 2026).
2. **"SLim (NeurIPS 2026)" could not be identified.**
   - arXiv API `ti:SLiM AND abs:sign` returned 0 hits.
   - Web searches found only a person named Slim Ouni (a GenSign poster).
   - The only NeurIPS 2026 sign paper found is ReSCUE (arXiv 2610.03022, simultaneous SLT). That is not an SSL paper. **Treat "SLim" as unverified.**
3. **Pose-only masked modelling is weak under frozen evaluation, but competitive after fine-tuning.**
   - Linear probe on WLASL2000 (per-instance top-1): SignBERT 18.24, BEST 21.19, SignBERT+ 24.50, contrastive STC 32.35 [V, STC Table VII].
   - Fine-tuned, the same models reach 39–52.
4. **Frozen dictionary retrieval on ASL Citizen** (training-free cosine top-1 against the train set as dictionary):
   - Raw pooled hand joint angles: 8.81 [V].
   - MASA (pose SSL): 27.96 [V\*].
   - SignRep (RGB plus pose-prior targets): 49.95 [V].
   - SignSeek (pose, gloss-supervised contrastive plus latent masked prediction): 57.66 [V].
   - **A label-free skeleton SSL model has never been shown above about 28 on this protocol.**
5. **What drives pose ISLR:**
   - Hands are indispensable. NLA-SLR keypoints on WLASL2000: body only 21.37 vs body+hands 48.54 [V].
   - Graph reduction to about 27 nodes beats the raw 133 nodes. SL-GCN on AUTSL val: 95.02 vs 63.69 [V].
   - Multi-stream joint/bone/motion ensembling adds about 6 points (SL-GCN WLASL2000: joint 45.61 → 4-stream 51.50 [V]).
   - Separating hand and trunk branches helps (STC: 65.22 vs 60.37 MSASL1000 [V]).
   - Large-scale language-supervised pretraining gives the biggest jump. Uni-Sign pose-only WLASL2000 63.13 [V]; LA-Sign 64.73 [V].
6. **Identity leaks through pose.** A frozen sign-trained pose network with a linear signer head identifies AUTSL signers at 80.18%, against 3.23% chance [V, 2410.13675].

---

## 1. Paper cards

### 1A. Pose and keypoint SSL for sign recognition

**SignBERT** (Hu et al., ICCV 2021, arXiv 2110.05382) [V]

| | |
| --- | --- |
| Input | Both hands' 2D poses from MMPose 133-keypoint whole-body. The 42 hand joints are tokens. |
| Pretraining | BERT-style masked joint, frame and clip modelling (50% of tokens chosen). A **hand-model-aware decoder** (MANO) reconstructs the poses. |
| Pretraining data | Training sets of the 4 downstream datasets only. |
| Hand-only "(H)" results | WLASL2000 39.40/36.74; MSASL1000 49.54/46.39; NMFs-CSL 67.0; SLR500 94.5. |
| "(H+P)" results (late fusion with a full-keypoint model) | WLASL2000 47.46/45.17; MSASL1000 59.80/57.06. |
| "(H+R)" results (late fusion with RGB) | WLASL2000 54.69. |

**SignBERT+** (Hu et al., TPAMI 2023, arXiv 2305.04868) [V]
- Hand tokens with spatial and temporal position encodings.
- Joint, frame and clip masking. Best ratio is 40% (MSASL1000: 62.42 at 40%; 59.80 at 60%).
- Pretraining data: 230,246 videos.
- Pose results: WLASL2000 48.85/46.37; MSASL1000 62.42/60.15; SLR500 95.4.
- Label-free pretraining beats supervised pretraining on the same "Partial" data (MSASL1000 60.13 vs 58.82).

**BEST** (Zhao et al., AAAI 2023 oral, arXiv 2302.05075) [V]
- Frame-wise pose *triplet* units: left hand, right hand and upper body.
- A d-VAE gives coupled tokenization: a shared hand codebook plus a body codebook, with 512-d codewords.
- Objective: masked unit modelling with cross-entropy on the tokens.
- Pretraining data: downstream training sets.
- Results: WLASL2000 46.25/43.52; MSASL1000 58.82/54.87; NMFs-CSL 68.5; SLR500 95.4.
- Tokenizer ablation on MSASL1000: k-means 54.29 < separate VQ 55.87 < coupled 58.82.

**MASA** (Zhao et al., TCSVT 2024, arXiv 2405.20666) [V]
- GCN embeddings, with **separate hand (GCN_H) and body (GCN_B) graphs**.
- Motion-aware masking that reconstructs **motion residuals** (interval k=3), plus momentum contrastive "semantic alignment".
- **Optimal mask ratio is 90%** (WLASL2000: 49.06 at 90% vs 25.64 without pretraining).
- Results: WLASL2000 49.06/46.91; MSASL1000 63.47/60.79; NMFs-CSL 71.7; SLR500 96.3.
- Robust to Gaussian keypoint noise up to σ=10 px (48.61), then collapses (42.15 at σ=20).

**STC: "SSL with Spatial-Temporal Consistency for SLR"** (Zhao et al., arXiv 2406.10501; arXiv comment says "Accepted by TIP") [V]
- Contrastive learning with joint and motion branches, knowledge transfer, and **separate hand and trunk** streams.
- Fine-tuned results: WLASL2000 51.98/49.46; MSASL1000 65.22/62.68; NMFs-CSL 75.6; SLR500 96.9.
- **It is the only pose paper reporting linear and semi-supervised protocols for all prior pose SSL models** (Table VII).
- Linear probe, WLASL2000 P-I: SignBERT 18.24, BEST 21.19, SignBERT+ 24.50, STC 32.35.
- Linear probe, MSASL1000 P-I: 31.54 / 30.89 / 34.44 / 48.30.
- Linear probe, SLR500: 69.89 / 70.84 / 71.65 / 94.3.
- Separate vs unified hand/trunk on MSASL1000: 65.22 vs 60.37.

**OpenHands** (Selvaraj et al., ACL 2022 [S], arXiv 2110.05877) [V]
- Pose-based toolkit (MediaPipe) with 1,129 h of unlabelled Indian SL.
- Pretraining comparison on INCLUDE (ST-GCN):
  - none 91.2
  - masked 91.3
  - contrastive 90.8
  - **dense predictive coding 94.7**
- Cross-lingual transfer to WLASL2000: 21.4 → 27.4.
- **This is the earliest evidence that latent *predictive* objectives beat reconstruction on sign poses.** It is the closest ancestor of a skeleton JEPA.

**SSL-SLR** (Madjoukeng et al., arXiv 2509.05188, v2 Mar 2026) [V]
- Negative-free SimSiam/BYOL-like objective, plus augmentation that preserves the discriminative frames.
- Reports ASL Citizen top-1 47.06 next to SignRep retrieval (37.47/49.95).
- **Mixed protocol:** it is unclear whether SSL-SLR used a linear probe or retrieval. Do not compare.

**SignMAE** (ICPR 2026, arXiv 2605.02094) [V]
- Three separately pretrained encoders. The relevant one for us is a **keypoint-heatmap** encoder with segmentation-guided spatio-temporal masking that **reconstructs only hand-region tokens**.
- Pretraining data: the downstream dataset itself (WLASL for ASL).
- WLASL2000 heatmap encoder by masking strategy:
  - random mask, whole-frame reconstruction: 24.36
  - tube mask: 26.48
  - ST mask, whole frame: 40.31
  - **ST mask, hand-region reconstruction: 47.08**
- Fused video+keypoint: 60.95/58.29.
- **Strong evidence that "where you mask and what you reconstruct" (hands) dominates the result.**

**CCL-SLR** (arXiv 2503.12485) [V]
- RGB+pose cross-modal consistency SSL.
- Results: WLASL2000 62.20/58.80; MSASL1000 77.71/76.27; NMFs-CSL 84.4.

**SignSeek** (arXiv 2609.03695, Sep 2026) [V]. This is the most important new competitor.
- **Input and architecture:** 2D keypoints split into 4 articulators (body, face, LH, RH). Each has its own graph encoder. The LH is mirrored and **shares the hand encoder with the RH**. A Conformer fuses the streams.
- **Losses:**
  - **GAC:** supervised (gloss) contrastive across signers.
  - **MAC:** masked contrastive with one articulator only.
  - **MAP:** **latent masked prediction of the most salient articulator**. A predictor regresses stop-grad clean encoder features with a cosine loss. This is JEPA-like, but it is trained jointly with labels.
- **Masking:** Articulator Saliency-Guided Masking picks the articulator with the largest standardized embedding norm.
- **Pretraining data:** 266,915 *labelled* samples (Sem-Lex, AUTSL, SLR500, MSASL subset, MeinDGS).
- **Frozen retrieval:** ASL Citizen R@1 57.66 (DCG 77.21); WLASL2000 34.75; NMFs-CSL 65.20.
- **Same table (frozen retrieval):** SignRep 49.95/29.92/63.04; MASA 27.96; SignCLIP 4.00.
- **Ablations:**
  - Losses on ASL Citizen R@1: GAC alone 48.40; GAC+MAP 55.93; all three 57.66.
  - Saliency masking vs fixed articulator on WLASL: ASGM 34.75; always-dominant-hand 31.65; uniform over articulators 29.66.
- **Fine-tuned WLASL2000:** 56.65/53.64.
- **Caveat:** Sem-Lex shares ASL-LEX vocabulary with ASL Citizen. The ASL Citizen numbers are therefore *not* lexically held out, and GAC uses gloss labels.

### 1B. Multi-stream, hybrid and RGB SSL

**SHuBERT** (Gueuwou et al., ACL 2025, arXiv 2411.16765). The exact protocol follows [V].
- **Streams per frame** (MediaPipe-located, about 95% hand detection on OpenASL, gaps interpolated):
  1. **Face:** blurred and partially masked face crop, encoded by DINOv2 continued-pretrained on 5M YouTube-ASL face crops. 384-d.
  2. **Left hand** and 3. **right hand:** DINOv2 continued-pretrained on 5M hand crops. 384-d each.
  4. **Body:** 7 upper-body landmarks (nose, shoulders, elbows, wrists), normalized to signing space. **14-d.**
  - Each stream is layer-normed and projected to 256-d, then concatenated into 1024-d per frame.
  - **The hands are appearance crops, not hand keypoints.**
- **Masking:** "random masking" independently masks small frame spans **in each channel separately**, with span = **3 frames (about 200 ms)** at about 14.89 fps.
  - Channel masking and time masking were ablated: How2Sign BLEURT 29.9 / 31.2 / 31.4 for channel / time / random.
  - **The masking probability is not stated in the paper text.**
- **Targets:** offline k-means with **k=256 per stream**, fitted on 10% of the data **on the input features**. There is no mention of HuBERT-style re-clustering iterations.
  - Four linear heads per output frame. Cross-entropy only on masked positions.
- **Model and data:**
  - 12 layers, 768-d, 86M parameters.
  - 400K steps on 8×A6000, about 7 days.
  - 984 h: YouTube-ASL minus OpenASL-overlap clips, refilled from YouTube-SL-25 ASL.
  - The paper notes YouTube-ASL contains **72.4% of the OpenASL test set and 38.2% of the MSASL test set**, so it does not report MSASL.
- **ISLR** (rank-1 LoRA, 0.17M trainable; mean-pool, BN, linear):
  - ASL Citizen R@1 0.65 / R@5 0.87 / R@10 0.91.
  - Sem-Lex 0.54.
  - WLASL2000 60.90/58.01.
- **Frozen vs fine-tuned SLT on How2Sign:** BLEU 4.7 vs 7.5 (last layer). Frozen with a layer-weighted sum: 7.1.
- **Phonology probe (App. B/C):**
  - 16 ASL-LEX 2.0 features: major/minor/second-minor location, contact, thumb contact, sign type, repeated movement, path movement, wrist twist, selected fingers, thumb position, flexion, spread, spread change, non-dominant handshape, handshape.
  - "Training setups identical to the full fine-tuning method … 16 classification heads simultaneously", with weight decay removed. **This is not a frozen probe.**
  - Uses the standard ISLR splits. ASL Citizen is signer-disjoint but **lexically overlapping**: every test sign is seen in training, so feature prediction can piggyback on sign identity.
  - Metric: per-feature R@1.
  - Averages: **Sem-Lex 79.38, ASL Citizen 85.02.** Handshape is the hardest (62.93/70.80).
  - The Sem-Lex result uses the full public test set, so it is not comparable to Kezar et al.

**SignDino** (arXiv 2609.06296, 5 Sep 2026, unreviewed) [V]
- **Method:** YOLOv8n plus ByteTrack track LH, RH and face crops. A **frozen DINOv3 ViT-B/16** embeds each crop. Per-stream 6-layer temporal Transformers (384-d) are student and EMA teacher.
- **Losses:** temporal DINO, iBOT frame masking (25–50%), KoLeo, Gram anchoring. Each stream is trained independently. No body pose.
- **Results:**
  - ASL Citizen R@1: frozen+linear 0.63; LoRA 0.67; full fine-tune 0.704.
  - WLASL2000 full fine-tune 68.5.
  - How2Sign BLEU 17.9.
- **Phonology protocol:**
  - The stream-fused representation is temporally averaged and fed to **16 parallel linear heads**.
  - "This probe neither trains the representation nor informs model selection", which implies frozen.
  - Yet it claims a "matched protocol" with SHuBERT's *full fine-tuning* probe. **That claim does not hold.**
  - Averages: Sem-Lex 87.64 vs 79.38; ASL Citizen 91.85 vs 85.02.
- **Reliability flags:**
  - Table 21 attributes "This is a really great point for self defense" to SHuBERT. SHuBERT's own Table 11 shows that line is Rust et al.'s output.
  - Its own translations nearly reproduce the references verbatim.
  - **Use its numbers only as claims.**

**SignMusketeers** (ACL Findings 2025, arXiv 2406.06907) [V]
- Per-frame SSL (DINO) on face and hand crops plus body pose.
- How2Sign SLT comparable to SSVP-SLT at under 3% of the compute.

**SSVP-SLT** (Rust et al., ACL 2024, arXiv 2402.09611) [V]
- Hiera MAE on anonymized (face-blurred) YouTube-ASL plus How2Sign, then SLT.
- How2Sign BLEU 15.5.
- **Pretraining clip length matters:** 128 vs 16 frames gives 14.3 vs 10.4 BLEU (YT+H2S).
- Face blurring costs about 0.6–1.3 BLEU on DailyMoth.

**SVT-ISLR** (Sandoval-Castaneda et al., arXiv 2309.02450) [V]
- Compares MaskFeat, VideoMAE, SVT and BEVT.
- 79.02 is on a **gloss-merged "gloss-based WLASL2000"**. On the original WLASL2000, MaskFeat reaches 52.83. Do not mix the two.
- Also runs layer-wise linear probes of phonological features.

**SignRep** (Wong, Camgoz, Bowden, ICCV 2025, arXiv 2503.08529; CVF open-access PDF exists) [V]. Internals are in §4.
- Fine-tuned results:
  - WLASL2000 61.05/58.89.
  - NMFs-CSL 84.1.
  - **ASL Citizen fine-tuned (Table 3): DCG 90.84, MRR 88.05, R@1 81.37.**
- **Frozen retrieval (Table 4):**
  - ASL Citizen R@1 **49.95** with activity-weighted pooling, 37.47 with plain averaging.
  - WLASL2000 29.92; NMFs-CSL 63.04.
  - Raw-pose baselines: hand joint angles 8.81, hand keypoints 7.96, all keypoints 0.98.
  - Pixel MAE baselines: HieraMAE on YT-SL 0.39, on Kinetics 0.25.
- Internal reproduction of the ASL Citizen frozen retrieval: 0.4902 [I].

### 1C. Language-supervised and large-scale pretraining (pose)

**Uni-Sign** (ICLR 2025, arXiv 2501.15187) [V]
- RTMPose-x whole-body. 69 keypoints in sub-poses: 21 per hand, 9 body, 18 face.
- **Generative SLT-style pretraining** on CSL-News (1,985 h) and YouTube-ASL.
- Pose-only: WLASL2000 63.13/60.90; MSASL1000 77.88/76.55.
- From-scratch sub-pose ablation (MSASL1000): hands only 36.07; +body 41.56; +face 43.45. **The face matters.**

**MSLU** (Zhou et al., arXiv 2408.08544) [V]
- SL-1.5M pose–text pairs.
- **Includes gloss→sentence templated ISLR data**, which is label leakage for ISLR.
- Objective: sign–text contrastive plus masked pose modelling.
- Results: WLASL2000 56.29/53.29; MSASL1000 74.07/71.81; NMFs-CSL 80.2.

**SignCLIP** (EMNLP 2024, arXiv 2407.01264) [V]
- MediaPipe Holistic pose and text contrastive training on Spreadthesign (456,913 videos).
- ASL Citizen R@1:
  - zero-shot 0.00
  - 10-shot 0.05
  - frozen + linear 0.23
  - fine-tuned + linear 0.60

**LA-Sign** (arXiv 2603.29057) [V]
- Pose-only. YouTube-ASL pretraining stage. Looped Transformer with hyperbolic alignment.
- Results: WLASL2000 64.73/62.41; MSASL1000 79.68.
- MSASL is subject to the YouTube-ASL overlap noted under SHuBERT.

**PhonSSM** (arXiv 2604.08761; ICLR 2026 workshop) [V]
- Phonological subspaces (handshape, location, movement, orientation) with graph attention and an SSM.
- Claims WLASL2000 72.08 and WLASL1000 62.90.
- **Flags:**
  - The splits are non-standard. It reports 17,272 train / 8,634 test for WLASL2000; the standard split is 14,289 / 3,916 / 2,878.
  - Accuracy rises with vocabulary size.
  - It cites DSTA-SLR as 53.70, but DSTA-SLR reports 53.68.
- **Not comparable.**

**Scaling SLT** (Zhang et al., arXiv 2407.11855) [V]: How2Sign BLEU 21.1 with about 6,600 h of partly private data [V\*, SHuBERT T1].

### 1D. Requested items not found or out of scope
- **Sign2Vec / pose wav2vec:** no such paper was found (the search for "sign2vec / pose2vec / SignVQ" was empty). OpenHands DPC is the closest work.
- **SignGraph** (CVPR 2024) is an **RGB continuous SLR** frame-graph model, not pose-based [S].
- **SPOTER** (WACVW 2022) [V]: pose Transformer with **hand landmarks normalized in a separate local frame**. WLASL100 63.18; WLASL300 43.78.
- **Human-JEPA** (arXiv 2608.21160) [V]: a human-centric video JEPA with a past→future split. Its abstract reports a 17-point re-ID collapse under block masking. It is not sign-specific.

---

## 2. Supervised pose ISLR state of the art and what drives it

- **SL-GCN / SAM-SLR-v2** (arXiv 2103.08833 / 2110.06161) [V]
  - HRNet whole-body pose with graph reduction 133→**27 nodes** (7 upper body + 10 per hand), all in **one graph**.
  - Without graph reduction: AUTSL val 63.69 vs 95.02.
  - WLASL2000 by stream: joint 45.61, bone 43.27, joint-motion 27.23, bone-motion 31.26; **4-stream 51.50/48.87**.
  - AUTSL 96.47 and SLR500 98.16 (multi-stream).
- **DSTA-SLR** (LREC-COLING 2024, arXiv 2403.12519) [V]
  - Input-dependent plus prior graphs and multi-scale temporal modelling, with 4 streams.
  - **WLASL2000 53.68/51.17** (joint only 51.44); **MSASL1000 65.74/62.31**.
  - This is the strongest pure-supervised pose GCN with no pretraining.
- **NLA-SLR** (CVPR 2023, arXiv 2303.12080) [V]
  - The keypoint stream is HRNet *heatmaps* of 63 keypoints (11 body, 10 mouth, 42 hands) fed to S3D.
  - Keypoint-64 alone: 49.10/46.18.
  - Video+keypoint: 61.05/58.05.
- **Pose-TGCN** (WLASL, WACV 2020) [V\*]: WLASL2000 23.65.
- **ST-GCN** (as reproduced): WLASL2000 34.40/32.53; MSASL1000 36.03/32.32 [V\*].
- **StepNet** (arXiv 2212.12857) is **RGB** part-aware (left/right/top/bottom regions). WLASL2000 56.89 RGB; 61.17 RGB+flow [V\*].
- **PoseFormer for one-shot ISLR** (arXiv 2502.20171) [V]
  - MediaPipe pose and hands (no face), supervised on ASL Citizen.
  - **ASL Citizen R@1 0.751, MRR 0.833**, versus I3D 0.631 and ST-GCN 0.595.
  - This is the best published pose-only ASL Citizen result. It is far above GCNs, which points to the input pipeline and frame embedding.
- **Kezar et al.** (EACL 2023, arXiv 2302.05759) [V]: SL-GCN on WLASL2000 goes from 29.4 to **38.1** (+8.7) with auxiliary ASL-LEX phoneme heads.

**Do strong models process the hands separately?**

| Model | Hand handling |
| --- | --- |
| SL-GCN, DSTA-SLR | Single graph; hands are subgraphs. |
| MASA, STC, Uni-Sign, SignSeek | Separate hand and body encoders. STC measured a gain of about 5 points. |
| SPOTER, SignSeek | Hands in a local frame or with a shared hand encoder. |
| SignBERT/+ | Hands only. |

The consistent winners isolate hand geometry: hand-local normalization or wrist-relative coordinates, a shared L/R hand encoder, and mirroring.

---

## 3. Benchmark tables

### 3a. WLASL2000 (original split), per-instance / per-class top-1

FT means fine-tuned.

| Method | Input | Pretraining | Protocol | P-I / P-C | Tag |
| --- | --- | --- | --- | --- | --- |
| Pose-TGCN | pose | none | sup. | 23.65 / – | V\* |
| OpenHands SL-GCN | MediaPipe pose | none | sup. | 30.6 | V |
| ST-GCN | pose | none | sup. | 34.40 / 32.53 | V\* |
| Kezar SL-GCN (+phon aux) | pose | none | sup. | 29.4 → 38.1 | V |
| SignBERT (H) | hands | SSL, DS sets | FT | 39.40 / 36.74 | V |
| BEST | hands+body tokens | SSL, DS sets | FT | 46.25 / 43.52 | V |
| SignMAE heatmap enc. | kp heatmaps | SSL, WLASL | FT | 47.08 / 45.24 | V |
| SignBERT+ | hands | SSL, 230k vids | FT | 48.85 / 46.37 | V |
| MASA | hand+body GCN | SSL, 4 sets | FT | 49.06 / 46.91 | V |
| NLA-SLR keypoint-64 | 63-kp heatmaps | none | sup. | 49.10 / 46.18 | V |
| SL-GCN 4-stream | 27-node pose | none | sup. | 51.50 / 48.87 | V |
| STC | hand/trunk, joint+motion | SSL (contrastive) | FT | 51.98 / 49.46 | V |
| DSTA-SLR | 27-node pose, 4-stream | none | sup. | 53.68 / 51.17 | V |
| MSLU | pose | pose–text, 1.5M (incl. ISLR labels) | FT | 56.29 / 53.29 | V |
| SignSeek | 4-articulator pose | sup.-contrastive + MAP, 266k | FT | 56.65 / 53.64 | V |
| Geo-Sign | pose | – | FT | 63.64 / 61.89 | V\* |
| Uni-Sign (pose) | 69 kp | generative SLT, YT-ASL | FT | 63.13 / 60.90 | V |
| LA-Sign | pose | YT-ASL stage | FT | 64.73 / 62.41 | V |
| PhonSSM | 75 landmarks | none | non-standard split | 72.08 (n/c) | V, flagged |
| *Reference:* SHuBERT | crops + 14-d body | SSL, 984 h | LoRA | 60.90 / 58.01 | V |
| *Reference:* SignRep | RGB | SSL + pose priors | FT | 61.05 / 58.89 | V |
| *Reference:* SignDino | DINOv3 crops | SSL | FT | 68.5 / – | V, flagged |

### 3b. Frozen-representation protocols (most relevant to SSL claims)

| Method | Data | Protocol | Score | Tag |
| --- | --- | --- | --- | --- |
| SignBERT / BEST / SignBERT+ / STC | WLASL2000 | linear, P-I | 18.24 / 21.19 / 24.50 / 32.35 | V |
| Same four | MSASL1000 | linear, P-I | 31.54 / 30.89 / 34.44 / 48.30 | V |
| Pooled hand joint angles | ASL Citizen | training-free retrieval R@1 | 8.81 | V |
| Pooled hand keypoints | ASL Citizen | training-free retrieval R@1 | 7.96 | V |
| HieraMAE on YT-SL (pixel MAE) | ASL Citizen | training-free retrieval R@1 | 0.39 | V |
| SignCLIP | ASL Citizen | retrieval R@1 / linear R@1 | 4.00 / 0.23 | V |
| MASA (pose SSL) | ASL Citizen | retrieval R@1 | 27.96 | V\* |
| SignRep (avg / weighted) | ASL Citizen | retrieval R@1 | 37.47 / 49.95 | V |
| SignSeek | ASL Citizen | retrieval R@1 | 57.66 | V (gloss-supervised) |
| SignDino | ASL Citizen | frozen + linear R@1 | 0.63 | V, flagged |
| SignRep / SignSeek / MASA | WLASL2000 | retrieval R@1 | 29.92 / 34.75 / 14.38\* | V (\* MASA saw WLASL in pretraining) |

### 3c. ASL Citizen (signer-disjoint splits; 2,731 signs; 52 signers) [V]

| Method | Input | Protocol | Score |
| --- | --- | --- | --- |
| ST-GCN | MediaPipe, 27 kp | sup. | R@1 59.52, DCG 76.37 |
| I3D | RGB | sup. | R@1 63.10 |
| SignCLIP | pose | FT + linear | 0.60 |
| SHuBERT | crops + body | LoRA | 0.65 |
| SignDino | DINOv3 crops | full FT | 0.704 (flag) |
| PoseFormer | MediaPipe pose + hands | sup. | **0.751** |
| SignRep | RGB | FT | **81.37** |

### 3d. Other benchmarks (P-I top-1) [V]

| Dataset | Results |
| --- | --- |
| MSASL1000 | ST-GCN 36.03; SignBERT(H) 49.54; BEST 58.82; SignBERT+ 62.42; MASA 63.47; STC 65.22; DSTA-SLR 65.74; MSLU 74.07; Uni-Sign pose 77.88; LA-Sign 79.68. The last two may be affected by the YouTube-ASL test overlap. |
| NMFs-CSL (pose) | ST-GCN 59.9; SignBERT(H) 67.0; BEST 68.5; MASA 71.7; STC 75.6; SignMAE keypoint 66.6; MSLU 80.2. RGB: SignRep 84.1; CCL-SLR 84.4. |
| SLR500 | ST-GCN 90.0; SignBERT 94.5; BEST/SignBERT+ 95.4; MASA 96.3; STC 96.9; DSTA-SLR 98.1; SL-GCN 98.16. Saturated. |
| AUTSL | SL-GCN multi-stream 96.47. Saturated. |

---

## 4. SignRep internals (local code `third_party/SignRep` plus paper)

**What the local repo contains.** The repo is *inference only*.
- `example_usage.py`
- `models/final_models/FINAL_hiera_latent_model_head_v25_active.py` (`Model`)
- `models/head_models/dict_head_v6_FINAL_thoughtful.py` (`ProjectionModel`)
- `models/hiera/{hiera_nodec_mae.py, hiera_clstoken.py, hiera_utils.py}`
- `augmentation/video/base_video_aug.py`

**No losses, discriminator or training loop are released.** The loss details below come from the paper [V].

**Backbone**
- `mae_hiera_base_16x224`: Hiera-B (embed 96, stages (2,3,16,3)), initialized from **K400 MAE**.
- Input: 16×224×224. Patch kernel (3,7,7), stride (2,4,4), which gives a token grid of 8×56×56.
- `mask_unit_size=(1,8,8)`, so there are 8×7×7 = 392 mask units. **Each unit is 1 temporal token (2 frames) × 32×32 px.**
- `MaskedAutoencoderHiera` has **no pixel decoder** (`norm`/`head` = Identity). `get_random_mask` drops units **uniformly at random**, with the 80% ratio from the paper.
- `num_cls_tokens=0`, so the latent is the **mean of the stage-4 tokens (768-d)**.
- At eval time, `mask_ratio=0`.

**Heads**
- `fc` = LayerNorm(768) → Linear(768,768, no bias). Its output is `features` (z^avg, the representation used for retrieval).
- `active_head` = Linear(768,128) → GELU → Linear(128,2): left/right hand activity logits.
- `ProjectionModel`:
  - Conv1d(768→512, k=1) → GELU → **ConvTranspose1d(512→384, kernel=16)**. This unrolls the *single pooled vector* into 16 per-frame vectors.
  - Then **linear** per-prior heads: body angles 22×2 (sin/cos), L/R hand angles 41×2, body keypoints 61×3, L/R hand keypoints 21×3 (wrist-relative 3D), body–hand distances 12×22×3, L/R fingertip distances 5×11×3.
- **The whole 16-frame pose trajectory must be decodable from one global vector.** There is no token-level or part-level prediction.

**Targets and losses** (paper §3, §5, Appendix Table 9)
- 3D pose from a sign-specific 3D estimator (Ivashechkin et al.).
- **Bone lengths are normalized to dataset medians** to avoid identity leakage.
- Smooth-L1 per frame. Keypoints with confidence below 0.5 are masked out.
- Weights:
  - angles and keypoints: 10
  - distances: 20 (hand-distance scale 4)
  - activity: 0.2
  - VICReg-style variance: 1.0 (hinge at σ=1)
  - covariance: 0.004
  - **adversarial style loss:** 2.0. A Gram-matrix "style" of the tokens is paired with z^avg, and a spectral-norm MLP discriminator tries to tell same-video from different-video pairs.
- Activity prior: a hand is inactive if it is below the stomach line *and* the summed temporal std of its keypoints is below 0.26.

**Training**
- YouTube-SL-25, with WLASL video IDs removed. Upper-torso crop, resized to 256.
- 500K iterations, batch 20, single RTX 3090, LR 1e-4, layer decay 0.85.
- The **checkpoint was selected by WLASL-val retrieval**, which is mild model-selection leakage for WLASL numbers.

**Ablations** (WLASL DCG, stride 8)
- angle 45.1; keypoint 32.9; distance 47.2.
- All priors: 48.5. Without masking: 46.3.
- +var/cov: 49.9. +adv: 50.7.
- **Masking helps even though reconstruction error is lower without it.**

**Fine-tuning**
- 64 frames via inflated patch embedding (kernel (7,7,7), stride (8,4,4)).
- Class-similarity KL with κ=0.2 adds about 1.1 points.

**Implications for a skeleton JEPA**
- SignRep's encoder sees pixels, but its *targets* are exactly the hand-centric geometric features (wrist-relative hand keypoints, finger angles, fingertip distances) that a skeleton model sees as *input*.
- Its masking is uniform over 32 px blocks, not anatomical.
- Its readout is a single global vector.
- A skeleton JEPA can supply what SignRep lacks:
  - part-tokenized latents
  - anatomically structured masking
  - **latent** targets instead of hand-crafted priors

---

## 5. Pretraining corpora and keypoint noise

**Corpora**

| Corpus | Size | Notes | Tag |
| --- | --- | --- | --- |
| YouTube-ASL | ~1,000 h, >2,500 signers | MediaPipe 85-point baseline: 21 per hand, 6 body, 37 face | V |
| YouTube-SL-25 | 3,207 h, >3,000 signers, >25 languages | | V |
| OpenASL | 288 h, >200 signers | | V |
| How2Sign | >80 h | 3 h Panoptic 3D | V |
| BOBSL | ~1,467 h | 39 interpreter signers; signer-independent splits | V |
| CSL-News | 1,985 h | | V |
| SL-1.5M | | ISLR datasets as templated text | V |
| Spreadthesign (SignCLIP) | 456,913 videos | | V |

**Contamination risks:** YouTube-ASL contains 72.4% of the OpenASL test set and 38.2% of the MSASL test set [V]. WLASL comes from YouTube; SignRep removed overlapping IDs.

**Hand keypoints are the weakest link**
- **COCO-WholeBody hand AP is far below body and face AP** [V]:
  - RTMW-x 384: hand 66.4 vs body 76.3 / face 88.4.
  - RTMPose-l 384: hand 57.9.
  - DWPose-l: hand 62.1.
- **MediaPipe drops whole hands** [V, 2604.24609]. On Phoenix it misses the left hand on 20.22% and the right on 23.61% of signing frames.
- Hand jitter is much higher than body or face jitter. MediaPipe J_acc on Signsuisse: hands 14.5 vs all non-leg points 1.44 vs face 0.27.
- In SLT, SDPose and Sapiens reach BLEU about 11.5 vs MediaPipe 10.3. Missing hands correlate with lower BLEU.
- **Failures cluster on hand–hand and hand–face contact** [V, Moryossef CVPRW 2021].
  - Hand-present frames: 85.13% for correct predictions vs 79.78% for wrong ones (p<0.0001).
  - On AUTSL, pose-only models reached 80–85% val while top RGB entries reached 98.4%.
- **The extractor choice changes benchmark numbers by several points** [V, 2609.18772].
  - Sem-Lex SL-GCN gloss: released poses 57.1 vs re-extracted MediaPipe 72.2.
  - Phonology average: 78.7 vs 87.1.
- **HaMeR** (arXiv 2312.05251; CVPR 2024 [K]) [V]: ViT-H MANO regressor trained on 2.7M examples. It gives 3D hands robust to occlusion, but costs a ViT-H per hand crop.
- Local data: ASL Citizen keypoints here are COCO-WholeBody-133 2D with unnormalized scores [I, memory].

---

## 6. Sign linguistics and phonology as structure or targets

**Parameters and resources**
- The parameters are handshape, location, movement, orientation and non-manual markers (NMMs).
- ASL-LEX 2.0 has 2,723 entries and 58 handshapes [V\*, 2609.18772 Table 1].
- Sem-Lex Table 1 lists the 16 features used by SHuBERT, SignDino and Sem-Lex [V].
- **Dominance:** ASL-LEX codes the dominant handshape and the non-dominant handshape separately. Sign type is one-handed, symmetrical/alternating, or asymmetric with same or different handshape.
  - A skeleton model should **canonicalize to dominant/non-dominant** rather than left/right. SignSeek mirrors the LH into a shared hand encoder [V], which is a partial step.

**Phonology as auxiliary supervision**
- **Kezar 2023** [V]: auxiliary phoneme heads add +8.7 on WLASL2000 (SL-GCN). Dominant handshape plus minor location alone gives 50.2 → 55.7 on the labelled subset.
- **Sem-Lex** (ASSETS 2023) [V]: 84,568 videos, 3,149 signs, 41 Deaf signers.
  - SL-GCN ISLR 66.6 → 68.6 with phonology recognition.
  - Supervised SL-GCN phonology average 0.858 (fine-tune) / 0.850 (multitask).
  - **This supervised pose GCN beats SHuBERT's SSL probe** (79.38), though on a different test subset.
  - 2609.18772 reproduces the comparison on one test set: SL-GCN 87.1 vs SHuBERT 80.2 [V].
- **WLASL-LEX** (ACL 2022) [V]: graph networks on skeletons recognize six phonological properties **even for signs unseen in training**. This is the lexical-split protocol that SHuBERT and SignDino lack.
- **Zero-shot cross-lingual handshapes** (WSLP@EMNLP 2026, arXiv 2609.18772) [V]
  - Five handshape features decoded into Catalan SL handshapes.
  - A static **MLP on 63-d dominant-hand MediaPipe 3D landmarks** beats SL-GCN and SHuBERT zero-shot: 53.8–54.5% handshape accuracy vs 8–29%.
  - It also beats them on LSC phonology accuracy: 79.7–80.0 vs SL-GCN 47.9–59.1 and SHuBERT 55.1–66.1.
  - SHuBERT hands-only equals all channels.
  - **High-capacity temporal models overfit to dataset framing.** This is a warning for any SSL model evaluated only in-domain.
- **ASL-MTP minimal translation pairs** (GenSign@CVPR 2026, arXiv 2604.27232) [V]: an SLT model relies mostly on manual cues and misses non-manual ones.
- **PhonSSM** [V] reports that errors concentrate on minimal pairs (38.7% of errors share 3 of 4 components). Its numbers are flagged.

---

## 7. Signer invariance

**Evidence that features encode identity**
- **Pose identifies people** [V, 2410.13675, citing Battisti et al. 2024 LREC-COLING WS]:
  - Signer ID from a frozen ISLR pose network on AUTSL: 80.18% (chance 3.23%).
  - Average-pose anonymization: 65.34%. Appearance transfer: 52.20%.
  - Sign accuracy drops from 80.97 to 65.82–71.46 under those transforms. **Identity and sign content are entangled in kinematics**: bone proportions, prosody, movement style.
- **The signer gap is large.** On PHOENIX-2014-SI, WER is 22.7 for seen signers vs 34.4 for unseen. A gradient-reversal signer-removal module narrows this to 23.0 vs 32.7 [V, arXiv 2212.13023, TOMM].
- **Frozen SSL features are signer-dominated** [I, H010]. For frozen SignRep features on ASL Citizen, unlabelled nearest neighbours are the same signer 48.7% of the time vs the same gloss 43.8%. Among the most confident 10%, 82% are the same signer.
  - Temporal-difference readouts reduce same-signer neighbours and raise retrieval R@1 by +3.1 to +3.8 (exploratory).

**Mechanisms in the literature**
- SignRep: median bone-length normalization of targets, plus an adversarial style discriminator.
- SignSeek: cross-signer supervised contrast.
- Signer-removal GRL.
- Appearance transfer and anonymization.

**No label-free skeleton SSL method explicitly targets signer invariance.**

---

## 8. Gaps and opportunities for a skeleton JEPA with anatomical masking

1. **Latent prediction on poses is under-explored and promising.**
   - Masked *reconstruction* of coordinates or tokens (SignBERT/+, BEST, MASA) gives poor frozen features: linear 18–25 on WLASL2000.
   - DPC-style latent prediction beat reconstruction in OpenHands.
   - SignSeek's MAP term adds +7.5 R@1 on ASL Citizen over GAC alone.
   - **A label-free JEPA with an EMA target encoder and articulator-structured masks is open.** Its natural claim: close the gap between MASA (27.96) and SignSeek (57.66) in frozen ASL Citizen retrieval *without gloss labels*. Use SignRep's 49.95 as the RGB reference.
2. **Anatomical masking with evidence behind it:**
   - Hand-region-only reconstruction gave +6.8 over whole-frame (SignMAE heatmaps).
   - Saliency-chosen articulator masking beat a fixed dominant hand by +3.1 (SignSeek).
   - Very high mask ratios work on pose (MASA 90%).
   - SHuBERT's per-channel random 3-frame spans beat whole-channel masking.
   - **Proposed design space:**
     - Mask entire articulators (dominant hand, non-dominant hand, face, body) over a time span.
     - Mask finger chains within a hand (selected-finger groups).
     - Use cross-hand masks: predict the non-dominant hand from the dominant hand and body (sign type, symmetry).
     - Predict hand-to-face and hand-to-body relations as latent targets (location, contact).
3. **Part-tokenized latents for phonology-aligned readouts.**
   - SignRep decodes all priors from one global vector. SHuBERT predicts stream-wise cluster IDs.
   - A JEPA with per-articulator target latents allows phonology probes on *part* tokens: handshape from the hand token, location from the hand–body relation token.
4. **Noise-aware targets and context.**
   - Hands are the noisiest part: AP about 58–66, with 20–24% of frames missing in MediaPipe.
   - Confidence-gated targets (SignRep uses a 0.5 threshold), and masking that treats missing detections as natural masks.
   - JEPA latent targets should be less sensitive to jitter than coordinate regression. Test this with MASA-style σ noise sweeps.
5. **Canonicalization**
   - Wrist-relative hand coordinates and per-hand local frames (SPOTER, SignRep's hand prior).
   - Dominant/non-dominant mirroring with a shared hand encoder.
   - Bone-length normalization (SignRep) to strip identity before the encoder sees the data.
6. **Signer invariance via JEPA itself.** Same-video different-time positives are confounded by signer. Options:
   - temporal-difference / motion latents
   - bone-normalized inputs
   - an optional GRL signer head using channel or video IDs as pseudo-signers
   - Report signer-ID linear probe accuracy as a *negative* metric.
7. **Context length.** Use 16 frames (SignRep) vs ≥64–128 (SSVP: 128 > 16 frames by +3.9 BLEU). Skeletons are cheap, so pretrain on long windows.
8. **Scale.** YouTube-ASL plus YouTube-SL-25 poses (thousands of hours, thousands of signers) remain unused for *label-free* pose SSL. SHuBERT used crops. Uni-Sign and LA-Sign used text supervision.

---

## 9. Evaluation-protocol recommendations

1. **Primary frozen metric: ASL Citizen training-free dictionary retrieval** (SignRep Table 4 protocol).
   - Cosine top-1 / R@5 / DCG, test queries vs train-set dictionary, signer-disjoint.
   - Report **avg** and **activity-weighted** pooling separately.
   - Note: in this project the ASL Citizen test split is quarantined and already used [I]. Use val or a new holdout for development.
2. **Secondary frozen metrics:**
   - Linear probe (STC protocol) on WLASL2000 and MSASL1000, P-I and P-C.
   - k-NN on Sem-Lex.
   - Always include raw-pose baselines (pooled hand angles 8.81 / hand keypoints 7.96) and a MASA-style pose-MAE baseline.
3. **Phonology probes done right:**
   - **Frozen** encoder plus 16 linear heads on ASL-LEX 2.0 features.
   - Report both the **standard split** (comparable to SHuBERT 79.38/85.02, while stating that SHuBERT fine-tuned) and a **lexically held-out split** (signs unseen in probe training; WLASL-LEX style).
   - Per-part probes: dominant-hand token → handshape, selected fingers, flexion; relation token → location, contact.
4. **Signer-invariance metrics:**
   - Frozen linear signer-ID accuracy (lower is better).
   - Same-signer vs same-gloss nearest-neighbour rates.
   - Seen-vs-unseen signer gap.
5. **Robustness:**
   - Keypoint-noise and dropout sweeps (σ 5–20 px; drop a whole hand for 20% of frames).
   - An extractor swap (MediaPipe vs RTMPose/RTMW).
   - Optionally a cross-lingual zero-shot handshape test (2609.18772).
6. **Fine-tuned comparisons** on original WLASL2000 and MSASL1000:
   - Use original splits only. Never use gloss-merged WLASL or PhonSSM-style splits.
   - Flag YouTube-ASL overlap for MSASL and OpenASL.
   - Separate pose-only from fused results (SignBERT "H+P/H+R" are late fusions).
   - Report seeds.
7. **Leakage hygiene:**
   - Exclude WLASL/MSASL video IDs from YouTube pretraining.
   - Do not select checkpoints on the test benchmark (SignRep selected on WLASL-val).
   - Declare any labelled corpora (SignSeek, MSLU) explicitly as supervised.

---

## 10. Reliability notes
- **SignDino** (unreviewed): misattributed SHuBERT outputs in its qualitative table, and a protocol mismatch in its phonology probe.
- **PhonSSM:** non-standard splits and non-monotonic accuracy.
- **SSL-SLR:** its ASL Citizen comparison to SignRep mixes protocols.
- **SVT-ISLR:** 79.02 is on gloss-merged WLASL.
- **MSLU:** pretraining includes ISLR labels.
- **SignSeek:** gloss-supervised, and Sem-Lex shares vocabulary with ASL Citizen.
- **Unverified items:** venue for OpenHands (ACL 2022, search snippet only); HaMeR (CVPR 2024, background knowledge only); "SLim NeurIPS 2026" not found; Battisti et al. 2024 numbers are only as cited by 2410.13675.
