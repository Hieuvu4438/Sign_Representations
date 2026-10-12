# L2 — Masking strategies & structured / part-based latent prediction
*Lit review for: "mask anatomical parts (hands / face / arms) of a sign-language skeleton and predict the masked part's latent from the rest" (structured anatomical masking + JEPA). Compiled 2026-10-11.*

**Verification legend.** **[V]** means the number or quote was grepped from raw paper text: arXiv HTML pulled with curl into `raw/L2/txt/*.txt`, the CVF PDF for EPM, or the ECCV PDF for S-JEPA in `raw/L3/sjepa_eccv.txt`. **[S]** means it comes only from a search/WebFetch summary. **[mem]** means it is from my background knowledge, not checked in this pass; use with care. For venues, **(meta)** means confirmed from the arXiv comment field or the paper header, and **†** means the venue is from memory.
**Corrections to the brief.** (1) Kong et al., *Understanding MAE via Hierarchical Latent Variable Models*, appeared at **CVPR 2023 (Highlight)**, not NeurIPS 2023 (meta). (2) EPM (*Evolved Part Masking*, Feng & Zhang, CVPR 2023) has no arXiv version; I verified it from the CVF open-access PDF. (3) I could not retrieve S2GAE (WSDM 2023) full text: curl of the NSF PDF failed. Only its bibliographic facts are given, [S]. (4) For MSN, Hiera, MultiMAE and HiCLR I did not run ablation greps in this pass; their descriptions are [mem].

---

## 0. Bottom line (for the proposal)

1. **Whole-part masking used alone usually hurts. Whole-part masking reached by a curriculum, or mixed with random masking, helps.** Three independent image studies agree:
   - **SemMAE.** Masking 75% of learned parts gives 52.9 linear probe (LP), versus 66.8 for random masking. An intra-part → whole-part curriculum gives 68.7.
   - **EPM.** Pure part masking gives 75.71, versus 78.63 for grid masking. An evolving grid→part schedule gives 79.89.
   - **I-JEPA.** A single large block target gives 20.2, versus 54.2 for 4 medium blocks with an informative context. All [V].
2. **Skeleton and sign evidence is positive but small, and it depends on which part is masked.**
   - SkeletonMAE (ICCV'23) gains about 1–1.5 points over random joint masking. Masking the limbs that dominate the action is best; masking head+torso is worst [V].
   - BEST (AAAI'23, sign) is the most relevant result. Hand-only masking beats no pretraining. **Body-only masking is *below* the no-pretraining baseline** (MSASL100 68.82 vs 73.44). Mixing hand, body and hand+body masking is best (80.98) [V].
   - MAMP: motion-aware masking adds +1.2 points, while changing the *target* (joint coordinates → motion) adds +10.1 [V]. **What you predict matters more than where you mask.**
3. **With semantic latent targets, structured masking matters less than with low-level targets.**
   - BEiT: block masking helps pixel targets by 4.3 mIoU but tokenizer targets by only 1.7 [V].
   - HPM: COCO gain +1.58 AP with RGB targets vs +0.36 with CLIP targets [V].
   - MaskAlign: attentive vs random masking, 85.3 vs 85.1 [V].

   A JEPA with EMA latent targets is the regime where mask-structure gains are likely *smaller*. Plan for that.
4. **Leakage is the main failure mode with irregular or structured tokens.** Mask positional and kinematic giveaways:
   - Point-MAE and Point2Vec: positional embeddings of masked patches leak shape. Concealing them adds +0.7 on ScanObjNN [V].
   - Point-M2AE: inconsistent masks across scales let the model "peek", costing −4.5 [V].
   - VideoMAE and V-JEPA: masks must span the whole time axis, otherwise the model copies from adjacent frames [V].
5. **Linear probe vs fine-tuning.** Mask design moves LP by large margins but FT by small ones:
   - MAE: LP spans about 20 points across masking ratios; FT is flat from 40–80% [V].
   - AutoMAE: LP +3.0, FT ±0.1 [V].

   If the proposal's claims are about frozen or retrieval evaluation, which fits the SignRep frozen protocol, structured masking has more room to show an effect.

---

## 1. Evidence table — structured vs random masking: when it helps

| Paper (venue) | Domain / target | Structured variant | Random baseline | Result (metric) | Verdict | Tag |
|---|---|---|---|---|---|---|
| MAE (CVPR'22†) | image / pixels | block 50% · block 75% · grid 75% | random 75% | FT 83.9 / 82.8 / 84.0 vs **84.9**; LP 72.3 / 63.9 / 66.0 vs **73.5** (ViT-L) | random wins; block at a high ratio collapses LP | [V] |
| SimMIM (CVPR'22†) | image / pixels | square; BEiT block-wise | random, patch 32 | best block-wise 82.7, square 82.6 vs **83.0** (Swin-B FT) | random with large mask units wins; AvgDist sweet spot 10–20 | [V] |
| BEiT (ICLR'22†) | image / dVAE tokens and pixels | block-wise 40% | random 40% | tokens: 82.86/44.65 vs 82.77/42.93; pixels: 81.04/41.38 vs 80.50/37.09 (IN / ADE) | block helps, **mostly for pixel targets** | [V] |
| I-JEPA (CVPR'23†) | image / EMA latents | multi-block (4×0.15–0.2, context 0.85–1) | random 0.6 | **54.2** vs 17.6; single block 20.2; quadrant 15.5 (1% IN LP) | structured multi-block is essential for latent prediction | [V] |
| V-JEPA (TMLR'24†) | video / EMA latents | 3D multi-block (short + long, full temporal extent) | random-tube 0.9 | K400/SSv2/IN **72.9/67.4/72.8** vs 51.5/46.4/55.6 | multi-block ≫ random tube for latent targets | [V] |
| VideoMAE (NeurIPS'22†) | video / pixels | tube 90% | random 90%; frame 87.5% | SSv2 **69.6** vs 68.3; frame 61.5 | tube > random ≫ frame | [V] |
| ST-MAE (NeurIPS'22†) | video / pixels | space-only (tube) 90; block 75; time-only 75 | agnostic random 90 | 83.5 / 83.2 / 79.1 vs **84.4** (K400 FT) | random wins; time-only is worst | [V] |
| MGMAE (ICCV'23†) | video / pixels | flow-warped tube | tube | SSv2 69.6→**71.0**, K400 80.0→81.2 | motion-consistent masks help | [V] |
| AdaMAE (CVPR'23†) | video / pixels | learned sampling, 95% | tube 90 / random 90 | **70.04** vs 69.3 / 68.3; frame 61.5 | adaptive is slightly better | [V] |
| SemMAE (NeurIPS'22†) | image / pixels | mask 75% of whole parts; curriculum | random | 52.9 vs 66.8; curriculum 0→1, γ=2: **68.7** | whole-part alone fails; curriculum wins | [V] |
| EPM (CVPR'23, CVF) | image / pixels | part-wise (graph-cut parts) | grid | α=1 (parts) 75.71 vs α=0 78.63; dynamic **79.89** (ViT-S IN top-1) | pure part masking fails; evolving schedule wins | [V] |
| AttMask (ECCV'22†) | image / iBOT distillation | mask most-attended tokens | random | kNN 49.7 vs 47.8; LP 57.9 vs 56.7; mask least-attended 44.0 | hard semantic masking helps; easy masking hurts | [V] |
| ADIOS (ICML'22, meta) | image / joint-embedding (mask as augmentation) | learned adversarial masks; GT object masks | MAE / BEiT masks | IN100-S 59.2 vs 43.7 / 46.4 (no mask 57.0); CLEVR F1-macro GT 0.373, shuffled-GT 0.332 | semantic masks help; random masks *hurt* joint-embedding | [V] |
| AutoMAE | image / pixels | learned adversarial mask | random | LP **66.7–68.8** vs 63.7; FT 83.32 vs 83.26 | LP gain only | [V] |
| HPM (CVPR'23†) | image / pixels | mask predicted-hard patches, easy→hard | random | 82.95 vs 82.49; hard→easy 81.71; mask-easy 82.36 | hard plus curriculum helps a little; reverse order hurts | [V] |
| MaskAlign (CVPR'23†) | image / CLIP-feature alignment | attentive 70% | random 70% | 85.3 vs 85.1 | negligible with semantic targets | [V] |
| Audio-MAE (NeurIPS'22, meta) | spectrogram / pixels | time / freq / time+freq | unstructured | quote: "unstructured masking with a higher ratio is preferred" for pretraining; structured is better for fine-tuning | random for pretraining | [V] (quote; numbers are in a figure) |
| Point-MAE (ECCV'22†) | point patches / coordinates | block 40/60/80 | random 60 | 92.67/92.67/92.50 vs **93.19** (MN40 FT) | random wins | [V] |
| Point-JEPA (WACV'25†) | point patches / EMA latents | contiguous multi-block (via sequencer) | random single 0.6 | **93.7** vs 92.5 (MN40 LP); contiguous single block 92.3 | multi-block wins, single contiguous block does not | [V] |
| Graph-JEPA (TMLR, meta) | graphs / latents | METIS subgraphs | random subgraphs | MUTAG 91.25 vs 91.58; REDDIT-M5 56.73 vs 56.08 | about equal | [V] |
| SkeletonMAE (ICCV'23, meta) | skeleton / GIN features | body-part (6 regions of 17 joints) | random joints 30–90% | best part pair **91.8**; head+torso 90.1; random 89.7–90.3 (FineGym) | part masking helps by about 1.5 points; part choice matters | [V] |
| MAMP (ICCV'23, meta) | skeleton / motion targets | motion-aware sampling | random | 84.9/78.6 vs 83.7/77.3 (NTU60/120 xsub LP) | +1.2 points (target change gives +10.1) | [V] |
| BEST (AAAI'23, meta) | sign pose / tokenized triplets (hands + body) | hand-only · body-only · hand+body | no pretraining | MSASL100 76.75 · **68.82** · 80.98 vs 73.44 | body-only masking is *harmful*; mixing is best | [V] |
| SignBERT+ (TPAMI, meta) | hand pose / coordinates | joint · frame · clip masking | no pretraining | +5.13 · +6.92 · **+8.63** top-1 (MSASL); all three combined is best | coarser temporal masking helps most | [V] |

**Reading of the table.** A structured mask wins when three conditions hold. (a) It removes a *redundant shortcut*: neighbouring patches (BEiT pixels), adjacent frames (VideoMAE tube), or within-part copying. (b) It *keeps an informative context*: I-JEPA's context ≥0.85, V-JEPA's multiple blocks. (c) It is *mixed with or scheduled after* easier masks (SemMAE, EPM, HPM, BEST, SignBERT+). Structured masks lose when the whole semantic unit is removed and the remainder carries little information about it: SemMAE whole parts, EPM α=1, BEST body-only, ST-MAE time-only.

---

## 2. Paper cards

### A. Masking in images, video and audio

**MAE** — He et al., CVPR'22†, arXiv 2111.06377.
- Random 75% beats block and grid: FT 84.9 / LP 73.5 for random vs 82.8 / 63.9 for block-75 [V].
- Quote: "Linear probing accuracy increases steadily with the masking ratio until the sweet point: the accuracy gap is up to ∼20% (54.6% vs. 73.5%). For fine-tuning... a wide range of masking ratios (40–80%) work well" [V].
- Block-75 has *higher training loss* and blurrier reconstructions. Grid is easier, with lower loss but worse representations [V].
- Lesson: task difficulty is non-monotone. The mask controls LP far more than FT.

**SimMIM** — Xie et al., CVPR'22†, 2111.09886.
- Random masking with large 32-px units at ratio 0.5 gives 83.0. Best block-wise is 82.7, best square 82.6 [V].
- Small mask units (4 px) give 81.9–82.1 regardless of ratio [V].
- Introduces **AvgDist**, the mean distance from a masked pixel to its nearest visible pixel. FT accuracy is ridge-shaped in AvgDist, with good runs at about 10–20 [V].
- Skeleton analogue: the mean graph or kinematic hop distance from a masked joint to its nearest visible joint.

**BEiT** — Bao et al., ICLR'22†, 2106.08254.
- Block-wise masking at 40% (blocks of at least 16 patches, aspect 0.3–3.3) [V].
- Ablation: removing block masking costs 0.09 IN / 1.72 ADE with tokenizer targets, but 0.54 / 4.29 with pixel targets. Quote: "blockwise masking is even more helpful for pixel-level auto-encoding, which relieves the suffering of short-distance dependency" [V].

**I-JEPA** — Assran et al., CVPR'23†, 2301.08243.
- Quote: "it is crucial to (a) sample target blocks with sufficiently large scale (semantic), and to (b) use a sufficiently informative (spatially distributed) context block" [V].
- Ablations, all 1% IN LP with ViT-B, [V]:

  | Ablation | Values |
  |---|---|
  | Mask type | multi-block 54.2; random 17.6; single block 20.2; quadrant 15.5 |
  | Target scale | 0.075–0.2: 19.2; 0.15–0.2: **54.2**; 0.2–0.3: 33.6 |
  | Context scale | 0.40: 31.2; 0.65: 47.1; 0.85: 54.2 |
  | Number of targets | 1: 9.0; 2: 22.0; 3: 48.5; 4: 54.2 |
  | Where targets are masked (ViT-H) | encoder *output*, i.e. full-context targets: 67.3; *input*: 56.1 |
  | Target space (ViT-L) | latent 66.9; pixels 40.7 |

- Overlap between context and targets is removed.
- The extreme sensitivity to target scale and number is a warning for part-level targets.

**V-JEPA** — Bardes et al., 2404.08471 (TMLR†).
- Short-range masks: union of 8 blocks at 15% each. Long-range masks: 2 blocks at 70%. About 90% masked overall, extended over the *entire* clip duration [V].
- Attentive probe, K400/SSv2/IN [V]:

  | Masking | K400 | SSv2 | IN |
  |---|---|---|---|
  | multi-block | 72.9 | 67.4 | 72.8 |
  | random-tube 0.9 | 51.5 | 46.4 | 55.6 |
  | causal, first 6 frames | 61.3 | 49.8 | 66.9 |

- Quote: "low spatial or temporal coverage results in a trivial prediction task." Several smaller blocks beat one large block, and two masks per clip (complementary tasks) beat one [V].

**VideoMAE** — Tong et al., NeurIPS'22†, 2203.12602.
- Tube masking fights "information leakage" from temporal correlation [V].
- SSv2/K400 FT: tube 90% 69.6/80.0; tube 75% 68.0/79.8; random 90% 68.3/79.5; frame 87.5% 61.5/76.5 [V].

**ST-MAE** — Feichtenhofer et al., NeurIPS'22†, 2205.09113.
- Contrary finding: spacetime-agnostic random 90% gives 84.4, space-only (tube) 83.5, block 83.2, time-only 75% 79.1 and 87.5% 75.4 [V].
- Quote: "optimal masking ratios for structure-aware sampling are in general lower" [V].
- VideoMAE vs ST-MAE shows that the tube benefit depends on the dataset (SSv2 motion vs K400 scene).

**MGMAE** — Huang et al., ICCV'23†, 2308.10794. Flow-warped masks keep the masked object masked through time. 800 epochs: SSv2 +1.4, K400 +1.2 over VideoMAE. Long schedule: K400 +0.3, SSv2 +1.5 [V].

**MotionMAE** — 2210.04154.
- K400 masking: time-only 74.6, random 78.8, space-only 79.0 [V].
- Adding a temporal-difference target raises SSv2 from 67.8 (frames) to 68.4 (frames+motion) [V].
- The best ratio is data-size dependent: UCF 75% 94.3 vs 90% 92.7; SSv2 66.0 vs 68.4 [V].

**AdaMAE** — Bandara et al., CVPR'23†, 2211.09120. A REINFORCE-trained sampler keeps visible tokens in high-information regions and reaches a 95% ratio. SSv2 70.04 vs tube 69.3 and random 68.3; frame masking 61.5 [V].

**SemMAE** — Li et al., NeurIPS'22†, 2206.10207. The closest image analogue to "mask a body part".
- Six unsupervised parts are learned per image (iBOT ViT-S plus an attention module). Linear probe on ViT-B, 8×8 patches [V]:

  | Masking | LP |
  |---|---|
  | random | 66.8 |
  | 75% of patches inside every part | 66.5 |
  | 75% of whole parts | **52.9** |
  | curriculum α 0→1, γ=2 (intra-part first, whole-part later) | **68.7** |
  | reverse curriculum (1→0) | 63.3 |

- Quote: whole-part masking "makes the task too hard"; the intra-part task "can enable the model to further learn better image representation in the latter task" [V].
- Mask granularity matters: parts need fine 8×8 patches. 16×16 LP: 65.0 vs MAE 63.7 [V].

**AttMask** — Kakogeorgiou et al., ECCV'22†, 2203.12719.
- Uses the teacher's CLS attention. In iBOT on 20% of IN, AttMask-High reaches kNN 49.7 / LP 57.9, versus random 47.8/56.7 and BEiT block 46.7/56.4. AttMask-Low (masking background) gives 44.0/53.4. Aggressive random 75% gives 47.3/55.5 [V].
- AttMask-Hint (leaves 1–5% of salient tokens visible) gives 49.5 [V].
- Lesson: in distillation (latent-target) SSL, mask the *informative* parts but keep a few hints.

**ADIOS** — Shi et al., ICML'22 (PMLR 162, [V] header), 2201.13100.
- An adversarial U-Net occluder plays a min-max game with a joint-embedding encoder (SimCLR/BYOL/SimSiam).
- IN100-S LP: ADIOS 59.2, no mask 57.0, MAE masks 43.7, BEiT masks 46.4 [V]. **Random masks hurt joint-embedding SSL.**
- CLEVR F1-macro: GT object masks 0.373, ADIOS 0.377, FG/BG 0.346, shuffled GT masks (same area, wrong place) 0.332, MAE 0.309 [V].
- Quote: "what is masked is more important than how much is masked" [V].
- The **shuffled-mask control** is the key experimental design to copy.

**AutoMAE** — 2303.06583.
- Manually raising the mask probability inside GT boxes helps slightly; quote: "Excessively raising the masking probability... degrades" [V].
- The learned mask generator gives LP 66.7–68.8 vs MAE 63.7 and SemMAE 65.0, but FT is 83.32 vs 83.26 [V].

**Hard Patches Mining (HPM)** — Wang et al., CVPR'23†, 2304.05919.
- An auxiliary head predicts the reconstruction loss, and the hardest patches are masked.
- Random 82.49; mask-hard 82.95; mask-easy 82.36; easy→hard 82.95; **hard→easy 81.71** [V].
- With an RGB target the COCO gain is +1.58 AP_box; with a CLIP target it is +0.36 [V].

**CL-MAE** — Madan et al., WACV'24†, 2308.16572.
- A learnable masker moves from cooperative (easy) to adversarial (hard). kNN ViT-B 42.1 vs 39.2 [V].
- Documents masker shortcuts. Quote: when minimizing error "it tends to avoid masking altogether... [as adversary] it tends to mask all the patches", which is fixed with a KL ratio constraint [V].

**EPM** — Feng & Zhang, CVPR'23 (CVF PDF).
- A part graph is built from the model's own features and cut into parts. α mixes grid masks and part masks.
- ViT-S, 200 epochs: grid 78.63/36.67; mix 0.5 79.02/38.13; **pure part 75.71/35.82**; dynamic γ=2 79.89/40.42 (IN top-1 / ADE mIoU) [V].
- Plug-in gains: MAE 79.20→79.89, BEiT 79.05→79.61, SimMIM 78.51→79.08 [V].
- Follow-up: *Evolved Hierarchical Masking* (2504.09155) is on arXiv; I did not read its numbers.

**Other image-masking papers.** ColorMAE (ECCV'24†, 2407.13036; filtered-noise data-independent masks) and *Symmetric masking* (2408.12772) were downloaded but not number-checked. Treat them as minor.

**CrossMAE** — Fu et al., 2401.14391 (venue unverified).
- In the MAE decoder, mask→visible cross-attention magnitude is 1.42 vs mask→mask self-attention 0.39 [V].
- Decoding only 25% of tokens independently gives ViT-B 83.5 vs MAE 83.3; 75% gives 83.7 [V].
- Implication: masked targets can be **predicted independently from context queries**. This supports a JEPA predictor that reads out each masked part with cross-attention.

**MaskAlign** — Xue et al., CVPR'23†, 2211.08887.
- Aligns *visible* student tokens to a frozen CLIP teacher's full-image features. FT is flat across mask ratios: 0% 84.9, 60–70% 85.3, 90% 84.4. Attentive vs random at 70%: 85.3 vs 85.1 [V].

**LeJEPA** — Balestriero & LeCun, 2511.08544.
- Uses DINO-style global and local views plus the SIGReg isotropic-Gaussian regularizer instead of EMA or stop-gradient.
- On mask design it states only that the prediction task "is designed based on a priori knowledge of the data... it is relatively intuitive to form x so that its views share the relevant information", while the anti-collapse design is "a game of Whac-A-Mole" [V].
- So LeJEPA hands mask/view design back to domain priors. Anatomical masks are a legitimate prior here.

**Audio-MAE** — Huang et al., NeurIPS'22 (meta), 2207.06405. Pretraining prefers unstructured masking at 0.8; structured (time+frequency) masking is better *during fine-tuning*. Quote: structured masking "drop[s] in performance" at high ratios "presumably because the task becomes too difficult" [V].

### Theory papers

**Kong et al.** — CVPR'23 Highlight (meta), 2306.04898.
- Theorem 1: each mask **m** induces a *minimal* set of latents **c** that captures all dependence between the masked part x_m and the visible part x_mᶜ. MAE identifies **c** [V].
- Quote: "extremely large or small masking ratios inevitably lead to low-level representations" [V].
- If the mask does not cover all observable descendants of a high-level variable, a low-level variable mixes in. If the visible part is too small, the same thing happens in reverse [V].
- IN1K LP, ViT-B with mask patch 16: ratio 0.1 47.45; 0.25 53.58; 0.5 60.07; 0.75 67.41; 0.9 62.97 [V].
- At ratio 0.75, mask patch 8 → 62.57, 16 → 68.96, 32 → 73.31 [V].
- Quote: "Learning high-level representations can be challenging for random masking" without prior knowledge of the latent structure [V].

**How Mask Matters** — Zhang, Wang & Wang, NeurIPS'22†, 2210.08344. Masking induces implicit positive pairs, and the MAE loss lower-bounds an alignment loss on a mask-induced augmentation graph. MAE suffers dimensional collapse, so they add a uniformity term (U-MAE) [V, text].

### B. Graphs and point clouds

**GraphMAE** — Hou et al., KDD'22 (meta), 2205.10803.
- Uniform random node masking is chosen because it "helps prevent a potential bias center, i.e., one's neighbors are neither all masked nor all visible" [V].
- Uses mask ratio ≥50%, re-masking before the GNN decoder, and SCE loss.
- Cora: 84.2; without masking 79.7; without re-mask 82.7; MSE loss 79.1 [V].

**GraphMAE2** — WWW'23 (meta), 2304.04779. Reconstruction "relies on the discriminability of the input features" [V], so it adds multi-view random re-mask decoding plus latent-representation prediction. This is relevant to noisy, low-dimensional pose features.

**MaskGAE** — Li et al., KDD'23 (meta), 2205.10053.
- Shows that a GAE behaves like contrastive learning over overlapping subgraphs, and that redundancy grows with overlap.
- Path-wise masking (random walks) "breaks the short-range connections between nodes, the model must look elsewhere". Edge and path masking peak at a ratio of about 0.7 [V].

**S2GAE** — Tan et al., WSDM'23 [S]: a graph-masking autoencoder for link and node tasks. Full text was not retrieved.

**Graph-JEPA** — Skenderi et al., TMLR (meta), 2309.16014.
- METIS subgraphs plus 1-hop expansion. One context subgraph predicts the targets' coordinates on a 2D unit hyperbola (cosh, sinh) for hierarchy.
- Hyperbolic vs Euclidean: MUTAG 91.25 vs 87.04; REDDIT about equal [V].
- METIS vs random subgraphs: about equal (91.25 vs 91.58) [V].
- Variance comes from "trivial context subgraphs" and from collapse "with positional embeddings of smaller, similar subgraphs" [V].

**Point-MAE** — Pang et al., ECCV'22†, 2203.06604.
- Random 60% gives 93.19, versus block 92.5–92.7 [V].
- **Positional-leakage principle.** Quote: "leakage of location information to mask tokens makes the reconstruction task less challenging" [V]. The fix is to give mask tokens to the decoder only.

**Point-M2AE** — NeurIPS'22 (meta), 2205.14401. Visible regions must be "block-wise within one scale and consistent across scales". Masking each scale independently lets the network "peek" and costs −4.5 (88.4 vs 92.9) [V].

**Point2Vec** — GCPR'23 (meta), 2303.16570. data2vec-style student embeddings for masked patches carry their centres, which "reveals the overall object shape... even under heavy masking". Concealing them improves ScanObjNN 86.8→87.5 and MN40 94.0→94.8 [V].

**Point-JEPA** — Saito et al., WACV'25†, 2404.16432. A greedy sequencer orders patches so that contiguous indices are spatially contiguous. Multi-block contiguous targets give 93.7, versus random single 92.5 and contiguous single 92.3 (MN40 LP) [V].

### C. Skeleton, pose and sign masking

**SkeletonMAE (Yan et al.)** — ICCV'23 (meta), 2307.08476.
- 17 joints are split into 6 parts: head, torso, left/right arm, left/right leg. A GIN encoder reconstructs masked parts' features and edges.
- FineGym [V]:

  | Masking | Mean acc. |
  |---|---|
  | random joints 30/50/70/90% | 89.7 / 90.3 / 89.9 / 90.1 |
  | right arm + right leg | **91.8** |
  | left arm + left leg | 91.1 |
  | head + torso | 90.1 |
  | no pretraining | 86.3 |

- The authors call part masking "action-sensitive".
- Caveat: this is FT inside their SSL pipeline, and the parts are chosen by an action-dominance heuristic.

**SkeletonMAE (Wu et al.)** — ICME'23 Workshop (meta), 2209.02399. A spatial-temporal MAE on skeletons. Not number-checked.

**MAMP** — Mao et al., ICCV'23 (meta), 2308.07092.
- Motion-aware masking samples masked tokens with probability softmax(motion intensity/τ). LP NTU60/120 xsub: 84.9/78.6 vs random 83.7/77.3 [V].
- Target ablation: joint→joint 74.8/72.5; joint→motion 84.9/78.6 [V].
- The input is a 4-frame segment × joint embedding at a 90% ratio [mem].

**S-JEPA** — Abdelfattah & Alahi, ECCV'24 (paper PDF in `raw/L3`).
- Reuses MAMP's motion-aware mask to pick target joints, and predicts EMA latents.
- Quote: the targets are richer than "isolated joint positions" because attention mixes all joints [V].
- Ablation (NTU60/120 LP): full 85.3/79.6; no EMA → collapse 1.6/0.83; no geometric transformations 84.6/78.9; MSE instead of CE 83.7/76.2; no centering 84.5/79.4 [V].
- This is the direct skeleton-JEPA precedent. Its masking is **not** anatomical.

**PSTL** — Zhou et al., AAAI'23 oral (meta), 2302.09018. A triplet of an unmasked anchor, Central Spatial Masking (joints with higher degree centrality are dropped more often), and Motion-Attention Temporal Masking [V, design text]. Structured "hub-joint" masking is used here as an *augmentation for invariance*, not as a prediction target.

**HiCLR** — AAAI'23 (meta), 2211.13466. Hierarchical consistency over growing augmentations [mem].

**ActCLR** — CVPR'23 (meta), 2303.10904. "Actionlets" (motion regions) are treated differently from static regions in augmentation [mem].

**MotionBERT** — ICCV'23 (meta), 2210.06551. 2D→3D lifting pretraining with frame-level and joint-level masks in a fixed 2:1 ratio. Masking-ratio sweep measured in MPJPE [V, text].

**PoseBERT** — TPAMI'22 (meta), 2208.10211. Quote: "increasing the masking ratio at training time always increases the performance at inference time" for random temporal masking of SMPL pose [V].

**MoMask** — CVPR'24†, 2312.00063, and **BAMM** — ECCV'24†, 2403.19435. Generative masked motion-token modelling. MoMask uses a cosine mask-ratio schedule, γ(τ)=cos(πτ/2) [V]. Both mask over *time tokens*, not body parts, so their relevance here is limited.

**UNIMASK-M** — AAAI'24 (meta), 2308.07301. Patchified skeletons with a single masked model for prediction and inbetweening. Not number-checked.

**Recent skeleton masking papers** (downloaded, not reviewed in depth; cite only after reading):
- GFP, ICCV'25 (meta), 2509.03609: general feature prediction.
- MacDiff, ECCV'24 (meta), 2409.10473.
- ASMa, 2602.06251: asymmetric spatio-temporal masking.
- AMR, CVPR'26 (meta), 2606.11450: adaptive masked reconstruction.
- SLiM, 2603.10648: compact-token masked feature prediction.
- 2409.17951: spatial-hierarchy and temporal-attention guided cross masking.

**Sign-specific.**
- **SignBERT** — ICCV'21 (meta).
- **SignBERT+** — TPAMI (meta), 2305.04868.
  - Hand-pose tokens. A chosen token gets joint, frame or clip masking, or is left unchanged. Joint masking uses zeros or "random spatial disturbance" to "mimic the failure cases of pose detectors" [V].
  - Gains over no pretraining (MSASL top-1): +5.13 joint, +6.92 frame, +8.63 clip; all three combined is best [V].
- **BEST** — AAAI'23 oral (meta), 2302.05075.
  - Masks per frame a "triplet unit" of left hand, right hand and body. Three cases: hand-only, body-only, hand+body.
  - MSASL100/200/1000 per-instance top-1 [V]:

    | Masking | MSASL100 | MSASL200 | MSASL1000 |
    |---|---|---|---|
    | none (no pretraining) | 73.44 | 70.79 | 51.96 |
    | hand-only | 76.75 | 71.37 | 54.60 |
    | **body-only** | **68.82** | **67.11** | **47.72** |
    | all three cases mixed | 80.98 | 76.60 | 58.82 |

  - Quote: hand-only beats body-only "due to the dominance of hand during sign language expression" [V].
- **MASA** — TCSVT'24 (meta), 2405.20666. Motion-aware masked autoencoding plus semantic alignment. Beats BEST by 5.81 / 5.99 / 2.81 on WLASL100 / 300 / 2000 [V].
- **SignMAE** — ICPR'26 (meta), 2605.02094. Segmentation-driven masking. Not read in depth.
- **SHuBERT** — ACL'25 (meta), 2411.16765. Multi-stream (hands / face / body) masked cluster prediction. Reference only, per user instruction.

### D. Cross-part and conditional prediction

**CroCo** — Weinzaepfel et al., NeurIPS'22 (meta), 2210.10716. One view is 90% masked and completed using a second, *unmasked* view. Quote: "high masking ratios, e.g. 90%, lead to the best pre-training performance" [V].

**SiamMAE** — Gupta et al., NeurIPS'23†, 2305.14344. The past frame is fully visible and 95% of the future frame is masked, decoded by cross-attention to the past [V].
- Both are templates for **asymmetric conditional prediction**: one stream fully visible, the other heavily masked.
- The analogue here: body+face visible, the dominant hand heavily masked.

**CrossMAE** (see A): independent per-target readout from visible tokens.

**MultiMAE** — ECCV'22†, 2204.01678. Masks across RGB, depth and semantic modalities and predicts each one from the visible tokens of the others [mem]. This is the closest analogue to treating hands, face and body as "modalities".

**MSN** — ECCV'22†, 2204.07141. A masked view is matched to an unmasked view's prototype assignment (joint-embedding, label-efficient) [mem].

**Hiera** — ICML'23 (meta), 2306.00989. A hierarchical ViT trained with MAE. Masking happens on coarse "mask units" so it stays consistent across stages [mem]. This parallels Point-M2AE's consistent-across-scale masking.

**MC-JEPA** — 2307.12698. A joint motion + content JEPA [mem].

**Graph-JEPA and Point-JEPA** (see B): subgraph and contiguous-block targets with positional queries.

**Part tokens / object-centric JEPA.** I found no peer-reviewed "slot-JEPA" with masking ablations in an arXiv search (`all:slot AND all:JEPA` returned only unrelated 2026 preprints). I am not claiming any.

---

## 3. Theory: what masking granularity makes the model learn

1. **Shared-latent view** (Kong et al., Theorem 1 [V]). A mask defines a split (x_m, x_mᶜ). The learned representation is the minimal latent **c** that the two sides share.
   - *Too small* a masked set, e.g. one fingertip, or partial masking of a part: the masked side is explained by a low-level local latent. Example: fingertip position given the other finger joints, which is hand-shape kinematics.
   - *Too large* a masked set, e.g. both hands *and* arms *and* face: the visible side contains too few descendants of the high-level variable (the sign's lexical identity). Again only low-level shared factors survive, such as torso pose and signer body shape.
   - For a sign skeleton, whole-part masks place the split *along anatomical boundaries*. Ideally **c** is then the cross-part shared factor: lexical identity, rhythm, prosody and phonological co-articulation.
   - But **c** also contains *kinematic coupling*. The wrist position is shared by the arm chain and the hand root. Hand orientation correlates with forearm direction. The non-dominant hand mirrors the dominant one in symmetric signs. Unless those carriers are masked too, **c** is dominated by kinematics. This is the theoretical root of the shortcut risks in §4.
2. **Redundancy and difficulty** (MAE / VideoMAE / ST-MAE [V]). The optimal ratio tracks redundancy: language about 15%, images 75%, video 90%. Structured masks have lower optimal ratios because each mask removes a correlated unit.
   - Skeleton tokens are low-dimensional (2–3 coordinates per joint) and temporally very redundant (30 fps). Time-redundancy argues for masking a part across *all* frames of a span (tube). Without that, the target can be interpolated from adjacent frames: VideoMAE frame 61.5 vs tube 69.6; V-JEPA masks the full temporal extent.
3. **Prediction distance** (SimMIM AvgDist [V]). Performance is ridge-shaped in the distance from masked to visible content. Whole-hand masking puts fingertips about 4 hops from the nearest visible joint (the elbow, if the wrist is masked), while knuckles are 1–2 hops away. Measure and control this hop distance rather than the masked-joint count.
4. **Masks as views** (How Mask Matters; ADIOS; MaskGAE [V]). Mask-and-predict is implicit alignment between mask-induced views. Views that overlap too much (shared kinematic chain) add redundancy without task-relevant information (MaskGAE Proposition 2). Semantically aligned masks beat area-matched shuffled masks (ADIOS 0.373 vs 0.332).
5. **Latent targets reduce the need for structured masks** (BEiT, HPM, MaskAlign [V]), with one exception. JEPAs *require* structured multi-block targets (I-JEPA 54.2 vs 17.6; V-JEPA 72.9 vs 51.5). With EMA latents, random scattered targets are trivially predictable from contextualized neighbours. Latent targets reduce the *gain from hard or semantic selection*, not the need for *contiguous* targets.
6. **Hierarchy** (Graph-JEPA hyperbolic coordinates; Point-M2AE / Hiera consistent multi-scale masks). Signs are hierarchical: finger → hand → arm → body, plus face. Part-level targets may benefit from hierarchical target geometry (Graph-JEPA MUTAG +4.2). But the evidence is mixed: other datasets are about equal.

---

## 4. Pitfalls (E)

- **Kinematic and positional leakage.** These are the skeleton equivalents of Point-MAE's and Point2Vec's centre leakage.
  - Hand coordinates are usually wrist-relative or absolute. If the hand is masked but the wrist (a body joint) and elbow stay visible, the hand's global position is given away.
  - If the predictor's target queries carry the hand's centroid or bounding box, the shape is given away.
  - Rule: give target queries only *part identity × time* positional codes, never coordinates. Mask the wrist with the hand, and optionally the forearm (elbow) too in a fraction of samples.
- **Left/right symmetry.** In two-handed symmetric signs (a large fraction of the lexicon) the masked hand is mirrored by the other hand. Two mitigations:
  - Always also sample masks that drop *both* hands for a span, so the model must use arm, face and context.
  - Report a stratified evaluation (one-handed vs symmetric two-handed signs) to show the representation is not just a mirror copy.
- **Temporal copying.** A part masked in frame t but visible in t±1 is solved by interpolation (VideoMAE frame masking; V-JEPA full-extent masks). Use part-tubes of ≥8–16 frames, or the whole clip.
- **Detector-noise shortcut.** Pose detectors already zero out or jitter occluded hands. If the mask token looks like detector dropout, the model learns to impute noise. SignBERT+ deliberately mimics detector failures; that is a different goal. Keep the mask token distinct from the "missing keypoint" encoding, and handle missing-joint confidences explicitly.
- **Masking the part that carries no information.** BEST body-only masking falls *below* no pretraining. AttMask-Low falls below random. In sign language, masking only legs or torso teaches nothing. Do not include lower-body joints; mask budget should go to hands, face and arms.
- **Too-hard whole-part masking early on.** SemMAE 52.9, EPM 75.71, and HPM hard→easy at −0.78 all show that whole-part masking from step 0 hurts. Use easy→hard (intra-part → whole-part).
- **Masker shortcuts if the mask is learned** (CL-MAE: it masks nothing or everything) → enforce a fixed budget.
- **Collapse with small, similar targets** (Graph-JEPA variance; S-JEPA without EMA → 1.6%). Part targets are few (about 4–6 per frame), so the predictor sees low-entropy positional queries. Use an anti-collapse regularizer (EMA + centering, as in S-JEPA, or SIGReg, as in LeJEPA) and monitor target-embedding rank.
- **Low-dimensional tokens and the mask ratio.** For 2–3D coordinate tokens, per-joint random masking at a high ratio still leaves easy interpolation (MAMP needed a motion target to fix this). Per-part masking at a moderate ratio (about 40–60% of upper-body tokens), extended in time, is the analogue of SimMIM's "large mask unit, moderate ratio" regime.
- **Linear probe vs fine-tuning.** Structured or semantic masking shows up mainly in LP and kNN (MAE, AutoMAE, SemMAE, AttMask). With full fine-tuning differences shrink to ≤0.5 points (AutoMAE, MaskAlign, CrossMAE). Pre-register which protocol the claim is about.

---

## 5. Design rules for anatomical masking on a sign skeleton
*Joints: hands 2×21, face about 68 or a subset, body 11–23 joints, upper body only.*

**R1. Tokenize by anatomical group.** One token per (group, short time segment). Suggested groups:
- left hand (21), right hand (21);
- left arm and right arm (shoulder–elbow–wrist; the **wrist belongs to both the arm and the hand mask sets**);
- face split into 2–3 sub-groups (eyebrows/eyes, mouth, contour/nose); use a subset of about 20–30 if 68 is too many;
- torso/head (shoulders, neck, nose).

Optional finer level: per-finger tokens (5 per hand), giving two granularities for an EPM/SemMAE-style curriculum.

**R2. Mask families.** Sample one family per sample (or per target block) from a mixture. BEST and SignBERT+ show that mixtures beat any single family.

| Family | Share | What it removes |
|---|---|---|
| M1 intra-part | ~30% | 50–75% of the finger/face tokens *inside* a part |
| M2 whole-part tube | ~40% | one group (dominant hand, non-dominant hand, mouth, or eyes/brows) over a contiguous span of 8–32 frames or the whole clip |
| M3 cross-part | ~20% | both hands, or dominant hand + mouth, or arm+hand chain (shoulder→fingertips) |
| M4 random-block control | ~10% | spatio-temporal random blocks (keeps the V-JEPA/I-JEPA prior) |

- M1 is the easy, intra-part task, analogous to SemMAE's "75% patches per part".
- For M2, V-JEPA shows the full temporal extent matters.
- M3 kills mirror and kinematic shortcuts.
- Optionally weight M1/M2 by motion (MAMP, +1.2 points) or by attention (AttMask), keeping 1–5% hints.

**R3. Curriculum.** Anneal from M1-heavy to M2/M3-heavy with α=(epoch/E)^γ, γ≈2 (SemMAE γ=2 best; EPM γ=2 best). Never run the reverse order (HPM −0.78; SemMAE reverse curriculum 63.3).

**R4. Keep the context informative.** About 50–75% of upper-body tokens stay visible per sample, so the context is never only the torso (I-JEPA context scale 0.85 vs 0.40: 54.2 vs 31.2). Predict about 2–4 target blocks per clip, for example dominant-hand tube + mouth tube (I-JEPA: 4 targets 54.2 vs 1 target 9.0).

**R5. Target construction.**
- Take targets from the EMA target encoder run on the **full** skeleton, then select the masked-part tokens at the *output* (I-JEPA: 67.3 vs 56.1 input masking).
- Optionally add a motion or velocity target head (MAMP +10.1 points). Or predict part-level pooled latents in addition to token latents, for multi-granularity.
- Consider CE-on-softmax with centering over MSE (S-JEPA: 85.3 vs 83.7).

**R6. Predictor queries.**
- Each query = (part ID embedding + temporal position), reading out by cross-attention to context tokens (CrossMAE / SiamMAE style).
- **No coordinates, centroids or bounding boxes of masked parts** (Point-MAE, Point2Vec).
- Express visible-part coordinates in a body-centred frame. Do not give hands absolute image coordinates if the wrist is masked.

**R7. Consistency across streams.** If the encoder has separate hand, face and body streams (SignRep-style multi-stream or hierarchical), apply the *same* mask to every stream that contains a joint. The wrist in the body stream and the hand root must be masked together (Point-M2AE: −4.5 when masks are inconsistent).

**R8. Exclude legs and low-information joints from the mask budget.** BEST body-only masking falls below baseline; AttMask-Low falls below random.

**R9. Required controls for a credible claim.**
- (a) Random joint masking at a matched token ratio.
- (b) Random spatio-temporal block masking (V-JEPA style).
- (c) **Shuffled-anatomy masks**: the same number of joints and temporal extent, but joints drawn across parts (ADIOS shuffle control).
- (d) Whole-part masking without curriculum.
- (e) Wrist-visible vs wrist-masked, for the leakage probe.
- (f) Both-hands vs one-hand, for the symmetry probe.

Report both the frozen protocol (LP/kNN, retrieval) and fine-tuning. Expect structured masking to help mainly the frozen protocol.

**R10. Diagnostics to log.**
- Per-family prediction loss: if M2 loss ≈ M1 loss, the part is being inferred trivially.
- Linear decodability of masked-hand *handshape class* vs *wrist position* from the predictor output: a high wrist R² with low handshape accuracy signals a kinematic shortcut.
- Target-embedding effective rank, for collapse.
- AvgDist-like mean hop distance from masked joints to the nearest visible joint, per family.

**Expected effect size.** Mask-structure gains in the closest analogues (SkeletonMAE, MAMP) are **+1–1.5 points**. In BEST and SignBERT+ the large gains (+5–8.6) are relative to *no pretraining*, not to random masking. A realistic hypothesis is +1–3 points on frozen evaluation over a random-block JEPA, with larger effects on hand-dependent sign subsets. Target design (motion or velocity, part-pooled latents) is likely to matter more than mask design.

---

## 6. Source files
- Raw text: `raw/L2/txt/{mae,simmim,beit,ijepa,vjepa,videomae,stmae,mgmae,motionmae,adamae,semmae,attmask,adios,automae,hpm,clmae,epm,kong,howmask,crossmae,maskalign,lejepa,audiomae,graphmae,graphmae2,maskgae,graphjepa,pointmae,pointm2ae,point2vec,pointjepa,skelmae_yan,skelmae_wu,mamp,pstl,motionbert,posebert,momask,bamm,signbert,signbertp,best,masa,siammae,croco,hiera,multimae,msn,...}.txt`
- S-JEPA: `raw/L3/sjepa_eccv.txt`
- arXiv metadata used for venues: `raw/L2/q/*.xml`
