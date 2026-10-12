# L1: JEPA and Current SSL Paradigms, as Background for a Skeleton/Keypoint JEPA for Sign Language

Compiled 2026-10-11. Raw arXiv HTML/abs pages and text dumps are in `scratchpad/litreview/raw/{html,txt}`. The official LeJEPA code files are in `raw/gh_lejepa`. I also read the local vendored code `method/jepa_targets/vendor/vjepa2_src` (facebookresearch/vjepa2 @204698b).

**Tags.** **[VR]** = verified-raw: I grepped the number or quote from the arXiv HTML/abs text or the code. **[FS]** = from-summary: it comes from a web-search summary or secondary page and is not re-verified. **[MEM]** = my prior knowledge, not re-checked this session (used only for a few venues and code URLs). Venue tags marked "cit" were verified from citation strings in other downloaded papers.

---

## 0. Framing in one paragraph

A JEPA trains an encoder E and a predictor P. P maps the embedding of a context x, plus a conditioning variable z (usually mask-token positions), to the embedding of a target y. The loss is computed in representation space. Any such system has to stop E from collapsing. Four families of anti-collapse mechanisms exist today:
1. **Asymmetric self-distillation.** Use an EMA target encoder with stop-gradient and a predictor (BYOL → data2vec → I/V-JEPA).
2. **Frozen teacher.** Train a teacher once, freeze it, then distil (SALT, and MVD/BEiT-v2-like pipelines).
3. **Explicit distributional regularisers** on a batch of embeddings: VICReg (C-JEPA, VJ-VCR, MC-JEPA, PLDM), SIGReg (LeJEPA, LeWM, Laya, PointTube-JEPA), or VISReg.
4. **Clustering / optimal-transport targets** (DINO/iBOT/DINOv2 heads, SIGMA).

Separately, the *target* can be raw input (MAE), a hand-crafted descriptor (MaskFeat HOG, MAMP motion), an online latent (JEPA), or a frozen latent (SALT). Most of the design space for a skeleton JEPA comes down to four choices: target, mask, anti-collapse mechanism, and probe.

---

## 1. Core vision JEPAs

### I-JEPA
Assran et al., arXiv 2301.08243, **CVPR 2023** [VR-cit]. Code: github.com/facebookresearch/ijepa [MEM].
- **Objective.** Average L2 between predicted and target patch representations [VR]. Targets come from an EMA target encoder over the full image. Target blocks are masked at the *output* of the target encoder, not the input. Output masking gives 67.3 IN-1% vs 56.1 for input masking (ViT-H, 300 ep) [VR].
- **EMA.** Momentum 0.996, linearly increased to 1.0 [VR].
- **Masking.** 4 target blocks, scale (0.15, 0.2), aspect ratio (0.75, 1.5). One context block, scale (0.85, 1.0), with target overlaps removed [VR].
  - Target scale (ViT-B, 300 ep, IN-1% linear): (0.075, 0.2) → 19.2, (0.1, 0.2) → 39.2, (0.125, 0.2) → 42.4, **(0.15, 0.2) → 54.2**, (0.2, 0.25) → 38.9, (0.2, 0.3) → 33.6 [VR].
  - Context scale: 0.40 → 31.2, 0.65 → 47.1, 0.75 → 49.3, 0.85 → 54.2 [VR].
  - Number of targets: 1 → 9.0, 2 → 22.0, 3 → 48.5, 4 → 54.2 [VR].
  - Masking type: multi-block 54.2, rasterized 15.5, single block 20.2, random 17.6 [VR].
- **Predictor.** Narrow ViT with width fixed at 384 and heads equal to the backbone's [VR]. Depth 6 vs 12 gives 64.0 vs 66.9 (ViT-L) [VR]. Width 384 vs 1024 gives 70.7 vs 68.4 (IN-1% fine-tune) [VR], so the bottleneck helps.
- **Target type.** Pixel targets give 40.7 vs 66.9 for latent targets (IN-1%, ViT-L) [VR].
- **Weight decay.** A 0.04→0.4 ramp vs a fixed 0.05 trades linear eval (77.8 vs 76.4) against IN-1% fine-tune (69.4 vs 70.7) [VR].
- **Compute.** ViT-H/14 trains in under 1200 GPU-h, more than 10× less than MAE ViT-H. Iterations are about 7% slower than MAE, but it converges in about 5× fewer iterations [VR].
- **Lesson.** The results are extremely mask-sensitive. With a bad mask, the "JEPA" collapses to near-useless features (9–20% vs 54%) without diverging. **Mask geometry is the main hyperparameter.**

### V-JEPA
Bardes et al., arXiv 2404.08471, **TMLR 2024** [VR-cit]. Code: github.com/facebookresearch/jepa [VR].
- **Objective.** Uses an **L1** loss ("more stable" than L2) on EMA targets [VR]. Momentum starts at 0.998 and rises to 1.0. Schedules are computed for 112.5k iterations, but training stops at 90k [VR].
- **Masking.** Short-range masks are the union of 8 blocks covering 15% of each frame. Long-range masks are the union of 2 blocks covering 70%. Both extend over all frames, giving an average masking ratio of about 90% [VR]. Tokens are 16×16×2 tubelets.
- **Predictor.** 12 blocks at width 384, taking mask tokens plus positional embeddings [VR].
- **Ablations (ViT-L, attentive probe).**
  - Feature vs pixel targets (K400 / SSv2 / IN1K): 73.7 / 66.2 / 74.8 vs 68.6 / 66.0 / 73.3. K400 fine-tuning is equal (85.6 vs 85.4) [VR].
  - Masking: multi-block 72.9 / 67.4 / 72.8; causal multi-block[12] 71.9 / 63.6 / 72.2; causal[6] 61.3 / 49.8 / 66.9; **random-tube 90%: 51.5 / 46.4 / 55.6** [VR].
  - Appendix: several blocks per mask beat one large block. Two masks per clip (short + long) beat one. Low spatial or temporal coverage makes the task trivial [VR].
- **Probe protocol** (key for us):
  - Average vs attentive pooling: K400 56.7 → 73.7, SSv2 50.1 → 66.2 [VR].
  - Linear vs attentive, ViT-L: VideoMAE K400 52.5 → 77.8, SSv2 41.3 → 61.2. V-JEPA K400 56.7 → 80.8, SSv2 50.1 → 69.5 [VR].
  - DINOv2-g on SSv2: 38.3 → 50.0 [VR].
  - The authors argue that the unnormalised regression objective gives no a priori reason for linear separability [VR].
- **Label efficiency.** The gap to pixel methods grows as probe labels shrink [VR].

### V-JEPA 2 and V-JEPA 2-AC
Assran et al., arXiv 2506.09985, 2025 tech report [VR]. Code: github.com/facebookresearch/vjepa2 [VR].
- **Scaling ingredients.** Relative to a ViT-L/VM2M baseline: data 2M→22M videos gives +1.0, ViT-L→ViT-g (1B) gives +1.5, 90k→252k iterations gives +0.8, and resolution 256→384 plus 16→64 frames gives a cumulative +4.0, reaching 88.2 average over 6 tasks [VR].
- **Recipe changes.**
  - 3D-RoPE instead of sincos, which "stabilises training for the largest models" [VR].
  - Warmup-constant-cooldown LR schedule.
  - **Fixed EMA and weight decay** instead of ramps, which had "minimal impact" [VR].
  - Progressive resolution: up to 8× faster [VR].
  - Probes are 4-layer attentive probes [VR].
- **Local code check** (`configs/train/vitl16/pretrain-256px-16f.yaml`, `app/vjepa/train.py`):
  - Masks: 8 blocks at spatial_scale 0.15 plus 2 blocks at 0.7, temporal_scale 1.0, aspect ratio [0.75, 1.5] [VR].
  - Predictor: depth 12, width 384, 12 heads. Mask tokens are zero-initialised [VR].
  - EMA fixed at 0.99925. `loss_exp: 1.0`, i.e. `mean(|z−h|)` [VR].
  - **Targets are `F.layer_norm` over the feature dimension of the EMA output** [VR].
  - The collator truncates all context and target index sets in a batch to the batch minimum [VR]. Variable-length skeleton clips need the same trick or padding.
- **V-JEPA 2-AC.** The encoder is frozen and a new ~300M action-conditioned predictor is trained (24 layers, 1024 wide) [VR]. Training uses under 62 h of DROID video [VR] with an L1 teacher-forcing loss plus a 2-step rollout loss [VR]. Planning uses CEM on L1 latent goal distance [VR].

### V-JEPA 2.1 (dense features)
Mur-Labadia et al., arXiv 2603.14482, Mar 2026 [VR]. Code: the vjepa2 repo, `app/vjepa_2_1` [VR, local].
- **Diagnosis.** In V-JEPA 2, visible tokens act as "global aggregators". Dense features are noisy: 22.2 mIoU on ADE20K and 0.682 RMSE on NYU with linear probes [VR].
- **Fix 1: context loss.** The predictor also outputs predictions at *visible* tokens, regressed to EMA targets with weight λ_i = λ/√d_min(i, M), where d_min is the distance to the nearest masked token [VR]. This raises ADE20K 22.2→33.9 and lowers NYU RMSE 0.682→0.473, but **hurts SSv2** [VR].
  - Constant λ in {0, 0.05, 0.2, 0.5, 1.0} trades segmentation for classification.
  - Warming λ up over epochs 50–100 "greatly stabilizes" training [VR].
- **Fix 2: deep self-supervision.** Outputs of 3 intermediate layers plus the last are concatenated, MLP-fused, and predicted as 4 levels. This recovers SSv2 72.0 / IN1K 80.8 and raises ADE to 38.6 [VR].
  - Code: each level gets its own LayerNorm. λ = 0.5 for both images and video. Predictor depth is 24 [VR].
- **Results.** Scaling to ViT-G (2B) gives SSv2 77.7 and NYU 0.307 [VR].
- **Distillation.** Distilling into ViT-L/B uses a **frozen teacher instead of EMA** [VR].
- **Lesson.** Masked-only latent losses give good *global* features but poor *per-token* features. If per-frame or per-joint features matter (sign boundaries, fingerspelling), supervise context tokens as well.

### SALT ("Rethinking JEPA: frozen teachers")
Apple, arXiv 2509.24317, 2025 [VR]. Listed as an ICLR 2026 poster [FS].
- **Method.**
  - Stage 1: a "V-Pixel" teacher is trained with VideoMAE-style pixel reconstruction under V-JEPA multi-block masking.
  - Stage 2: the teacher is frozen and a student plus predictor regress its latents on masked regions.
  - No EMA and no stop-gradient machinery [VR].
- **Claims.** SALT beats V-JEPA 2 at matched FLOPs; on V-3.6M at 240k steps, ViT-L SALT averages +2.3 over 6 tasks [VR].
  - **Small or under-trained teachers suffice.** The best ViT-L student (77.4 avg) came from a ViT-L teacher, not ViT-H or ViT-G. The best split was 40k teacher steps and 200k student steps [VR].
  - Student loss predicts downstream accuracy (R² = 0.951) [VR].
  - Student quality is robust to the teacher's data mixture, except an ImageNet-only teacher [VR].
- **Lesson for us.** A cheap two-stage recipe is possible: a skeleton MAE or motion predictor as teacher, then a latent student. It also gives an interpretable loss.

### C-JEPA (contrastive JEPA)
Mo & Tong, arXiv 2410.19560, **NeurIPS 2024** [VR-cit].
- **Method.** I-JEPA plus a VICReg variance/covariance term, and invariance of mean patch representations across mask views [VR].
- **Claim.** EMA does not fully prevent collapse in I-JEPA [VR].
- **Gain.** Modest: ViT-B 600 ep linear 72.9 → 73.7, fine-tune 83.5 → 84.5 [VR].

### Other image/video variants (brief)
- **DMT-JEPA** (arXiv 2405.17995) [VR]. Targets for a masked patch are aggregated from semantically similar neighbouring patches, via cross-attention heads. ADE20K +1.4 (47.6→49.0), DAVIS +1.7 [VR].
- **VJ-VCR** (Drozdov et al., arXiv 2412.10925) [VR]. Video JEPA with variance-covariance regularisation instead of EMA, plus latent variables for future uncertainty. Beats a generative baseline on dynamics-probing tasks.
- **MC-JEPA** (Bardes et al., arXiv 2307.12698) [VR]. Shared encoder trained jointly on self-supervised optical flow (PWC-Net-style) and VICReg content features. Motion and content objectives help each other.
- **Factorized Latent Dynamics for Video JEPA** (arXiv 2605.17165, 2026) [VR]. Small-scale study of 18 auxiliary objectives. Most trade one capability for another. Factorising appearance and dynamics subspaces with hard-region weighting gave +5.92 on IN-100 and +3.21 on SSv2 [VR].

---

## 2. Collapse, theory, and regulariser-based JEPAs

### LeJEPA / SIGReg (exact formulation, as requested)
Balestriero & LeCun, arXiv 2511.08544, Nov 2025 [VR]. Code: github.com/rbalestr-lab/lejepa [VR].
- **Thesis.** An isotropic Gaussian embedding uniquely minimises worst-case downstream linear and nonlinear probe risk [VR].
- **SIGReg.**
  - Sample M random unit directions a. Project the batch: z = a^T f(x).
  - Compute the univariate **Epps–Pulley** statistic EP = N ∫ |φ̂_z(t) − e^{−t²/2}|² w(t) dt, with Gaussian window w(t) = e^{−t²/σ²}, σ = 1 [VR].
  - Average over directions (the paper uses an average, not a max, to avoid sparse gradients) [VR].
  - Gradients are bounded: |∂EP/∂z_i| ≤ 4σ²/N. Cost is O(N). Mini-batch bias is O(1/N), "not a concern even for minibatches as small as 16" [VR].
- **Paper code.**
  - `num_slices=256` by default. Integration points `t = linspace(-5, 5, 17)`, trapezoid rule. Slices are synced across GPUs by seeding with `global_step` [VR].
  - Recommended starting point: **1024 slices, 17 knots, domain [−5, 5]**. 512 slices are already competitive [VR].
- **Repo code.**
  - `MINIMAL.md` and `lejepa/univariate/epps_pulley.py` use the symmetric half-domain: t ∈ [0, 3] with 17 knots and doubled weights. The weight vector folds in φ(t) = e^{−t²/2} [VR].
  - `err = (mean_N cos(t·z) − φ)² + (mean_N sin(t·z))²`. Statistic = (err · weights) × N, then averaged over 256 fresh `randn` slices per call [VR].
- **Where SIGReg is applied.**
  - On the **projector output**, a 3-layer MLP 2048-2048-proj_dim with BatchNorm, *per view*: `proj` has shape (V, N, d) and the mean is taken over the batch dimension N [VR].
  - It is not applied to the backbone embedding, which is what gets probed.
  - The LeJEPA paper's `LeJEPA()` averages SIGReg over views [VR].
- **Loss and λ.**
  - Loss = (1−λ)·pred + λ·SIGReg [VR].
  - **λ = 0.05 recommended** [VR]. The repo's MNIST example uses λ = 0.02, V = 4, proj_dim = 16 [VR].
- **The "prediction" in the paper is multi-crop invariance, not masked latent prediction.** Every view's embedding regresses the mean of the V_g = 2 global-view embeddings (L2). Defaults are V_g = 2, V_l = 8 with 224/96 crops [VR].
  - **No predictor network and no teacher.** Removing them does not collapse, unlike in other methods [VR].
  - SWA or a teacher gives a small boost for ViTs [VR]. Registers are not needed [VR].
- **Informative loss.** Spearman correlation between training loss and linear-probe accuracy is about 85%, rising to about 99% with loss/λ^0.4 [VR].
- **Small-domain results.** In-domain pretraining on Galaxy10 (11k images, 400 ep) beats DINOv2/v3 transfer [VR].
- **Scale results.** IN-1k linear: ViT-L 77.1, ConvNeXtV2-H 78.5 [VR]. That is below DINO-family numbers, so LeJEPA's selling point is stability and few hyperparameters, not SOTA.

### Works that combine *masked/token-level latent prediction* with SIGReg instead of EMA (checked 2025–2026)
- **PointTube-JEPA** (Lee & Chen, arXiv 2608.24093, Aug 2026; code github.com/Leoxu3/PointTube-JEPA) [VR]. 4D point-cloud *human action* video (MSRAction-3D, NTU RGB+D point clouds).
  - Shared encoder, no EMA. Target = LayerNorm(full-sequence latents at masked tokens). Smooth-L1 loss. **SIGReg on the full-sequence token latents**, λ = 0.01 best. 75% random target masking (85% too much) [VR].
  - Ablation (MSRAction-3D): full 94.08; without SIGReg 89.90 (target std collapses early); **EMA target encoder 88.50**; **stop-grad on targets 79.44**; without target-position query 93.03 [VR].
  - The closest published analogue to an EMA-free skeleton JEPA.
- **Laya** (EEG, arXiv 2603.16281, Mar 2026) [VR].
  - Masked contiguous temporal blocks are predicted in latent space with a shared encoder and *stop-grad targets*.
  - **SIGReg only on the projected global CLS embedding** [VR].
  - **λ < 0.02 collapses to a constant embedding.** λ was reduced from 0.05 to 0.02 for the longer/larger run. Removing the projector degrades accuracy to 0.476. Stop-grad matters more as training gets longer [VR].
- **LeWorldModel (LeWM)** (Maes, Le Lidec, Scieur, LeCun, Balestriero, arXiv 2603.19312; code github.com/lucas-maes/le-wm) [VR].
  - Next-embedding prediction (action-conditioned, not masked) plus SIGReg on Z ∈ R^{N×B×d}: history × batch × dim, i.e. the CLS embedding after a **1-layer MLP with BatchNorm**. That projector is needed "because the final ViT LayerNorm prevents our anti-collapse objective from being optimized effectively" [VR].
  - M = 1024 directions, λ = 0.1. The number of directions and knots barely matters [VR].
  - Stated limitation: **"low data diversity weakens SIGReg in simple, low-dimensional environments"** [VR].
- **Partial or other uses.**
  - AeroJEPA (2605.05586): token-level latent matching plus SIGReg with λ_s = 0.001 [VR].
  - ProteinJEPA (2605.07554): main recipe is an EMA teacher, half-depth targets, and VISReg. A teacher-free SIGReg variant appears only in the wall-clock study. **JEPA-only training collapses downstream; JEPA only helps as an add-on to MLM** [VR].
  - EB-JEPA library (2602.03604): a learned bottleneck projector gives about +3 points over regularising encoder outputs directly [VR].
  - "Spectral-Target Physical Latent Structuring" (2609.04264): SIGReg prevents global collapse but not **"lazy" encoders** that omit task-relevant information [VR]. It also cites a "temporally centered SIGReg" for multi-task world models [VR, title only].
- **VISReg** (arXiv 2606.02572, Jun 2026) [VR, abstract]. Keeps the VICReg variance term and replaces covariance with sliced-Wasserstein sketching. Claims SIGReg's gradients vanish under collapse and that it couples scale and shape.
- **Not found.** No 2025–26 *image or video ViT* paper (I-JEPA/V-JEPA scale) reporting SIGReg replacing EMA for multi-block token prediction. The evidence is small-scale (point clouds, EEG, aero fields, world models).

### Theory and collapse analyses
- **Littwin et al., "How JEPA avoids noisy features"** (arXiv 2407.03475, **NeurIPS 2024** [VR-cit]).
  - In deep linear self-distillation, JEPA is greedily biased toward features with high *regression coefficient* ρ ("influential"). MAE prioritises high-covariance features [VR].
  - The bias **depends on encoder depth L**: "When L > 1, JEPA will suppress … noisier directions … increases with depth". At large depth the encoder becomes low-rank [VR].
  - A 1-layer JEPA's dynamics match an infinite-depth MAE [VR].
- **Sobal et al., "JEPAs focus on slow features"** (arXiv 2211.10831, NeurIPS 2022 SSL workshop) [VR].
  - VICReg- and SimCLR-JEPAs ignore distractor noise that changes every step. They **fail when the distractor is fixed over time**: the "slowest features" are the noise, so "the representation will only contain the noise" [VR].
- **Tian et al., DirectPred** (arXiv 2102.06810, **ICML 2021** [VR-cit]).
  - In non-contrastive SSL, the predictor's eigenspace aligns with the input correlation. Weight decay balances predictor and encoder [VR].
  - **EMA is not necessary if the predictor has a higher relative LR** [VR]. Setting the predictor by PCA works [VR].
- **SimSiam** (arXiv 2011.10566) [VR]. Removing stop-grad gives 0.1% (chance). No predictor gives 0.1%. A fixed random predictor gives 1.5% [VR].
- **Balestriero & LeCun, "Learning by reconstruction produces uninformative features"** (arXiv 2402.11337, **ICML 2024** [VR-cit]) [VR]. TinyImageNet using only the top-variance subspace (90% of pixel variance) gives 45% accuracy; the bottom subspace (20% of variance) gives 55%. Reconstruction spends capacity on variance, not on discriminative content.
- **"Gaussian Embeddings: JEPAs secretly learn your data density"** (arXiv 2510.05949) [VR]. The anti-collapse term provably estimates sample density, which can be used for outlier detection and data curation.

---

## 3. The teacher-target lineage (data2vec → DINOv3) and descriptor/pixel targets

- **data2vec** (arXiv 2202.03555; ICML 2022 [MEM]) [VR].
  - Targets: **average of the top-K blocks**, each normalised before averaging. K = 6 for ViT-B/L, K = 8 speech, K = 10 NLP.
  - Normalisation: **instance norm over the sequence for speech** (because adjacent targets are highly correlated), parameter-free LayerNorm for vision and NLP.
  - Loss: smooth-L1 with β = 2 for vision. τ = 0.9998 constant for vision; 0.999→0.9999 annealed for speech [VR].
  - Speech loss ablation: L2 17.1, L1 17.2, smooth-L1 16.8–17.3 WER, so the choice barely matters [VR].
  - Target layer: **self-attention output as target fails (100 WER)**, FFN output gives 13.1 [VR].
  - Multi-layer averaging is more robust than K = 1 [VR].
  - Collapse risk factors [VR]: LR too high or warmup too short; τ too low (student collapse propagates to teacher); **modalities with highly correlated adjacent targets and long masked spans**. Skeleton sequences are exactly that.
- **data2vec 2.0** (arXiv 2212.07525, **ICML 2023** [VR-cit]) [VR].
  - Masked tokens are not encoded. A conv decoder replaces the transformer decoder.
  - **Multi-mask: M = 8 masked versions per sample reuse one teacher target** (M = 16 for the longest run). Inverse block masking.
  - Trains with batch 512 instead of 2048–4096. Batch 64 with M = 1 diverged.
  - 2–16× faster [VR]. A global CLS loss helps vision [VR].
- **iBOT** (arXiv 2111.07832, **ICLR 2022** [VR-cit]; github.com/bytedance/ibot) [VR]. MIM as self-distillation with an online tokenizer (EMA teacher on unmasked image) plus a CLS distillation loss. ViT-B: k-NN / linear / fine-tune 77.1 / 79.5 / 84.0 [VR].
- **DINOv2** (arXiv 2304.07193, **TMLR 2024** [VR-cit]) [VR].
  - **KoLeo** = −(1/n) Σ log d_{n,i}, where d is the nearest-neighbour distance within the batch. It improves Oxford-M retrieval 55.6→63.9, other metrics roughly flat [VR].
  - The iBOT MIM term is "critical for dense tasks", about +3 [VR].
- **DINOv3** (arXiv 2508.10104) [VR]. Dense features **degrade with long training**: VOC mIoU falls after about 200k iterations while classification keeps rising.
  - **Gram anchoring** fixes this. It pushes the student's patch-similarity Gram matrix toward that of an early-checkpoint "Gram teacher", which constrains similarity structure rather than the features themselves [VR].
- **MaskFeat** (arXiv 2112.09133, **CVPR 2022** [VR-cit]) [VR]. MViT-S on K400, fine-tune: scratch 81.1, RGB pixel targets 80.7, **HOG 82.2**, dVAE 81.7, DINO features 82.5, supervised features 81.9. A cheap hand-crafted local descriptor gets most of the gain of learned targets.
- **VideoMAE** (arXiv 2203.12602, **NeurIPS 2022** [VR]) [VR].
  - **90–95% tube masking**. Tube masking from 75%→90% gives SSv2 68.0→69.6.
  - Works on **3–4k videos**. "Data quality > quantity", and domain shift matters.
- **VideoMAE V2** (arXiv 2303.16727, **CVPR 2023** [VR], abs only). Dual masking (encoder subset and decoder subset), billion-parameter ViT. K400 90.0, SSv2 77.0 [VR].
- **SIGMA** (arXiv 2407.15447, **ECCV 2024** [VR]) [VR]. Learns the target space with a projection network. Collapse is avoided by Sinkhorn-balanced assignment of space-time tube features to learnable clusters, and the model predicts the cluster assignments symmetrically.

**What latent targets buy over pixel/coordinate targets.** The direct ablations above are: I-JEPA 66.9 vs 40.7 (IN-1%); V-JEPA +5.1 on K400 attentive, about 0 on SSv2, 0 under fine-tuning; MaskFeat HOG ≈ DINO-feature targets. The gain is large under frozen probing in high-dimensional, nuisance-heavy inputs (pixels). It shrinks under fine-tuning, and shrinks when the "raw" target is already a good descriptor (HOG, motion). See §5.

---

## 4. Domain JEPAs (non-image) and skeleton-specific work

- **A-JEPA** (audio, arXiv 2311.15830) [VR]. I-JEPA on spectrograms with a **curriculum from random block masking to time-frequency-aware masking**, because local time-frequency correlation makes block masking easy. Fine-tuning uses regularised masking.
- **Point-JEPA** (arXiv 2404.16432, **WACV 2025** [VR-cit]; github.com/Ayumu-J-S/Point-JEPA [VR]) [VR].
  - A **greedy sequencer** orders unordered patch embeddings so that contiguous indices are spatially contiguous. Contiguous blocks are then sampled.
  - Target ratio (0.15, 0.2), context (0.4, 0.75), smooth-L1 loss. ModelNet40 linear SVM 93.7 ± 0.2.
  - Increasing target blocks helps up to a point; 1 → 93.0, 2 → 93.5 [VR].
  - **Lesson:** impose a topology-aware ordering so that "block" means something. For skeletons, use kinematic chains or body parts.
- **Graph-JEPA** (Skenderi et al., arXiv 2309.16014, **TMLR** [VR]; github.com/geriskenderi/graph-jepa) [VR].
  - METIS partition into subgraphs with 1-hop expansion, each encoded by a GNN.
  - **One context subgraph predicts m target subgraphs.**
  - Targets are mapped to a 2D unit-hyperbola coordinate (cosh/sinh of an angle) to encode hierarchy.
  - Positional conditioning uses node-level random-walk structural encoding (RWSE); patch-level relative RWSE is worse.
  - High run-to-run variance, blamed on trivial context subgraphs and collapse with small, similar subgraphs [VR]. Random partitioning is also competitive [VR].
- **T-JEPA** (tabular, arXiv 2410.05016, **ICLR 2025** [VR]) [VR].
  - Predicts latents of one feature subset from another, with no augmentation.
  - **An initial collapse (loss → ~0) occurs with EMA**; the authors reproduced it in official I-JEPA.
  - **Appended [REG] tokens were required to escape collapse** on low-dimensional tabular inputs.
- **Brain-JEPA** (fMRI, arXiv 2409.19407, **NeurIPS 2024 spotlight** [VR]) [VR].
  - **Brain-gradient positioning**: functional-connectivity gradients as ROI coordinates. This beats sin-cos and anatomical positional encodings.
  - Spatiotemporal masking into Cross-ROI, Cross-Time, and Double-Cross target regions, with overlapped sampling for a variable masking ratio.
- **ECG-JEPA** (arXiv 2410.08559) [VR]. Cross-Pattern Attention (CroPA) is a masked self-attention that restricts attention to clinically relevant lead/time patches. It avoids the "naive L2 on raw signal" problem.
- **ACT-JEPA** (arXiv 2501.14622, IEEE Access 2026 [VR]). Jointly predicts action chunks and latent observation sequences.
- **seq-JEPA** (arXiv 2505.03176; github.com/hafezgh/seq-jepa) [VR]. A sequence of views, each concatenated with an action/transformation embedding and aggregated by a transformer, predicts the next view. Per-view encodings become equivariant; the aggregate becomes invariant.
- **LLM-JEPA** (arXiv 2509.14252) [VR, abstract]. Adds a JEPA term to LLM fine-tuning and pretraining.
- **DINO-WM** (arXiv 2411.04983, **ICML 2025** [VR-cit]) [VR]. Frozen DINOv2 patch features with a learned predictor for planning.
- **Intuitive physics** (Garrido et al., arXiv 2502.11831) [VR]. V-JEPA-type latent prediction shows violation-of-expectation physics. Pixel predictors and MLLMs are near chance.
- **Human-JEPA** (arXiv 2608.21160, Aug 2026; coordinator request) [VR].
  - **Input:** RGB video only; *no pose or keypoint input or targets*. Keypoints appear only (i) as a data-curation gate, where a person detector with keypoints filtered 536,699 K700 clips to 164,431 full-body clips, and (ii) as downstream probes (COCO keypoints AP 0.620, Market-1501 ReID 0.4635 mAP).
  - **Initialisation:** continued pretraining from V-JEPA 2.1-L.
  - **Targets:** the masked term uses EMA-teacher L1. The dense context term is **anchored to a frozen copy of the initialisation** (A = f₀). Otherwise targets drift and dense quality "silently collapses" while the loss looks fine.
  - **Masking:** block masks are replaced by a **pure past→future split**: context is the first half of the tubelets, target the second half. Block masks act as a "copy task" that cost 4.8 points on NTU-120 action. The new recipe still pays 2.7 points there, plus 2.9 on K700. Per-batch mixtures with block masks are studied [VR].
  - Part-guided (body-part) masking was tested only as a control [VR].
- **Skeleton-specific prior art.**
  - **S-JEPA** (Abdelfattah & Alahi, **ECCV 2024**, LNCS pp. 367–384 [VR-cit]; no arXiv version found). Predicts latents of masked joints. A "centering" step aids stability [FS].
    - Third-party table (GFP, ICCV 2025) [VR], NTU-60 linear x-sub / x-view: SkeletonMAE 74.8/77.7, **MAMP 84.9/89.1, S-JEPA 85.3/89.8**, GFP 85.9/92.0.
    - **S-JEPA needs 1200 epochs vs 400; 90 h 57 m vs 20 h 27 m for MAMP** on one RTX 4090.
    - NTU-120: S-JEPA 79.6/79.9 vs MAMP 78.6/79.1. PKU-II: 53.5 vs 53.8 [VR].
  - **MAMP** (arXiv 2308.07092, **ICCV 2023**; github.com/maoyunyao/MAMP) [VR].
    - Predicts the *temporal difference (motion)* of masked joints from the joint sequence.
    - NTU-60 / NTU-120 linear: joint→joint 74.8/72.5, **joint→motion 84.9/78.6**, motion→joint 76.5/71.0, motion→motion 75.9/70.5.
    - Motion-aware masking beats random by +1.2/+1.3. 90% masking is best. Segment length l = 4 is best.
  - **GFP** (arXiv 2509.03609, **ICCV 2025**) [VR]. A co-trained lightweight target-generation network with variance regularisation against collapse. 6.2× faster than MAE-style skeleton methods.
  - **SLiM** (arXiv 2603.10648, 2026) [VR].
    - Compact tokens, decoder-free EMA teacher feature prediction, plus contrastive learning.
    - **Independent joint/patch masking is solvable by local interpolation.**
    - Semantic Tube Masking (connected anatomical group over a consecutive time span) beats conventional joint masking by up to 3.5 points (87.7 vs 84.2 x-sub).
  - **A4Mer / Action Motifs** (arXiv 2604.28173, **CVPR 2026 Highlight** [VR]). Nested latent transformer over variable-length pose segments ("Action Atoms"), with masked token prediction in latent spaces.
  - **Sign language.** No JEPA on sign-language keypoints was found [FS search]. Existing pose-based sign SSL is masked reconstruction (SignBERT/+, BEST, SignMAE 2026) or contrastive.

---

## 5. Representation diagnostics and probe protocol

- **RankMe** (arXiv 2210.02885, **ICML 2023** [VR]). Effective rank (entropy of normalised singular values) of embeddings is a label-free predictor of performance for JE-SSL.
- **LiDAR** (arXiv 2312.04000, **ICLR 2024** [VR-cit]). Rank of the LDA matrix of the surrogate SSL task, with views of a sample treated as a class. Discriminates informative from uninformative dimensions.
  - **On I-JEPA, varying target-mask scale, RankMe "correlates poorly with downstream performance"; LiDAR correlates highly** [VR].
  - I-JEPA and data2vec are "extremely sensitive to the masking hyperparameters" [VR].
- **SALT** (above). The student loss, the teacher's RankMe, and teacher accuracy were studied as teacher-selection signals [VR].
- **Attentive vs linear probing.**
  - Attentive probing changed V-JEPA's conclusions by 16–24 points (above).
  - Efficient probing (Psomas et al., arXiv 2506.10178, **ICLR 2026** [VR]), IN-1K linear → efficient attentive probe [VR]:
    - MAE ViT-B 67.7 → 75.6 (+7.9); MAE ViT-S 47.4 → 64.6 (+17.2); SimMIM +13.6.
    - DINO +0.5, iBOT +0.5, BYOL +0.8.
    - "ep changes the relative ranking … MAE surpasses BYOL."
  - Methods that optimise *patch tokens* rather than a global vector benefit most [VR].
  - "Beyond [cls]" (arXiv 2412.03215, abs) [VR]: MIM attention is spread uniformly over patches, so the [cls] token aggregates poorly; selective aggregation recovers much of the gap.
- **Protocol consequence.** Linear-on-mean-pool *penalises* masked-prediction models (MAE, JEPA) relative to invariance models. Attentive probes *close* the gap. Report both, and compare methods only within a protocol.

---

## 6. Design rules for a JEPA on low-dimensional structured inputs (skeleton graphs)

Inputs here are, for example, about 50–130 keypoints × (x, y[, z, conf]) per frame, at 25–60 fps.

**R1. Tokenise coarsely and topologically. Do not use one token per joint per frame.**
- Make tokens like (body-part group × short time segment): for example, left hand, right hand, arms/torso, face × l = 4 frames. MAMP found l = 4 best [VR].
- Order tokens along kinematic chains so that "contiguous block" is meaningful, the analogue of Point-JEPA's sequencer [VR].
- Dense joint tokens cost compute and invite interpolation shortcuts (SLiM) [VR].

**R2. Mask structure, not points. Mask a lot.**
- Use multi-block masks in (part × time). Each block covers a connected anatomical group over a contiguous time span (Semantic Tube Masking: +3.5) [VR].
- Use two mask families per sample, V-JEPA style: short-range (many small part-time blocks) and long-range (few large spans). Total masked fraction about 75–90%.
  - Supporting values: V-JEPA ~90% [VR]; MAMP 90% best [VR]; PointTube 75% best, 85% too much [VR]; Point-JEPA context 0.4–0.75 [VR].
- Never use independent random joint masking. It is the analogue of V-JEPA's random-tube result, 51.5 vs 72.9 [VR].
- Add a **past→future (forecasting) family** for a fraction of batches. It forces dynamics rather than copying (Human-JEPA) [VR], but it costs global/appearance performance, so mix rather than replace.
- Sign-specific: whole-hand-over-time masks force cross-articulator inference. Watch R8, because some of that information is genuinely unpredictable.

**R3. Targets: normalise, prefer multi-layer, and consider a hybrid with a cheap descriptor.**
- EMA-teacher targets, masked at the teacher *output* (+11.2 in I-JEPA) [VR], with LayerNorm per token over features (vjepa2 code) [VR].
- Because adjacent skeleton targets are highly correlated, data2vec's advice applies: **normalise over the sequence (instance norm) or the batch** to promote variance [VR].
- Average the top-K layers (data2vec) [VR], or use V-JEPA 2.1 deep self-supervision (4 levels) [VR].
- Never use raw attention-output features [VR].
- Strong low-cost alternative or auxiliary target: **joint velocity/motion** (MAMP: +10.1 over joint targets) [VR]. This is the skeleton analogue of MaskFeat HOG (82.2 vs pixel 80.7) [VR].
- ProteinJEPA [VR] and MAMP vs S-JEPA [VR] both suggest that **latent prediction is best as a complement**: motion or coordinate head plus latent head.

**R4. Predictor: narrow, deep enough, richly position-conditioned.**
- Width ≈ 1/2–1/3 of the encoder: I-JEPA 384 vs 1024 gave 70.7 vs 68.4 [VR]. Depth ≥ 6, since 12 beat 6 by 2.9 [VR].
- Mask tokens carry (joint-group id, kinematic or graph positional encoding, temporal RoPE). V-JEPA 2 found 3D-RoPE stabilising [VR].
- Positional design is measurable: Brain-JEPA gradient positioning beat sincos and anatomical encodings [VR]; Graph-JEPA node-level RWSE beat relative RWSE [VR]; PointTube target-position query 94.08 vs 93.03 [VR].
- For hands, use learned joint embeddings plus graph-Laplacian eigenvectors or hop distance to the wrist.

**R5. Loss.**
- L1 (V-JEPA, "more stable") [VR] or smooth-L1 (data2vec, Point-JEPA, PointTube) [VR] on normalised targets. Loss choice is second order (data2vec ±0.3 WER) [VR].
- If per-frame or per-joint features matter (sign segmentation, fingerspelling, handshape retrieval), add the V-JEPA 2.1 **distance-weighted context loss**: λ_i = λ/√d, λ ≈ 0.5, warmed up [VR]. Pair it with deep supervision, otherwise the global task suffers [VR].

**R6. Anti-collapse: pick one deliberately and instrument it.**
- **(a) EMA.** Use τ in roughly 0.996–0.9999. On a small dataset with few steps, a very high fixed τ (0.99925 is tuned for 252k-step runs) makes the teacher nearly static. Too-low τ collapses (data2vec) [VR].
  - Expect an early near-zero-loss phase (T-JEPA, I-JEPA reproduction) [VR]. Consider [REG] tokens for low-dimensional inputs [VR].
- **(b) Frozen teacher (SALT).**
  - Train a skeleton MAE or MAMP teacher briefly, freeze it, and train the student longer. In SALT a teacher with 40k of 240k total steps was best [VR].
  - It gives an informative loss (R² 0.951) [VR] and removes EMA tuning. This is the most compute-sane choice for a small lab.
- **(c) SIGReg (EMA-free).**
  - Apply it to a **BatchNorm MLP projector** output, not to LayerNorm'd encoder outputs (LeWM, EB-JEPA) [VR].
  - Choose deliberately between token-level latents (PointTube) and global CLS (Laya).
  - λ ≈ 0.01–0.05: below 0.02 collapsed in Laya; 0.01 was best in PointTube [VR]. 256–1024 slices, 17 knots [VR].
  - **Stop-grad on targets is unresolved.** It hurt badly in PointTube (79.44 vs 94.08) but helped long runs in Laya [VR]. Ablate it.
  - Caveat: SIGReg weakens with low data diversity and low-dimensional environments (LeWM) [VR], and does not prevent "lazy" encoders [VR].
- Always log target-latent std per dimension, RankMe, and LiDAR on held-out clips.

**R7. Data amortisation.**
- Use data2vec 2.0 multi-mask: M = 4–8 masks per clip reusing one teacher forward pass [VR]. This acts as augmentation and lets small batches train.
- Avoid S-JEPA's cost profile (1200 epochs, 4.5× MAMP wall-clock) [VR].

**R8. Guard against "slow features" = signer identity.**
- Body proportions, camera framing, and signer-specific posture are constant within a clip. These are exactly the fixed distractors that JEPAs latch onto [VR, Sobal].
- Mitigations:
  - Normalise skeletons: root-centre, scale by shoulder width, optionally canonicalise bone lengths.
  - Per-sequence instance norm on targets.
  - Track a **signer-ID linear probe as a negative metric**. Rising signer decodability with flat sign accuracy is a red flag.
- Littwin's result implies JEPA suppresses low-ρ directions [VR]. Fine handshape detail is lexically essential but may be "low-influence" relative to arm trajectories, so probe handshape explicitly.

**R9. Evaluation protocol.**
- Primary: **attentive probe**, with 1–4 cross-attention queries, plus linear-on-mean-pool. Add kNN or retrieval for sign lookup, and a dense per-frame probe if R5 is used.
- Report all of them. MAE-style and JEPA rankings flip between probes [VR].
- Use the same probe head and budget across baselines, including MAMP and a skeleton MAE.

---

## 7. Risks and critique

1. **The core JEPA argument is weaker for skeletons.** JEPA's advantage is explained by avoiding high-variance, uninformative input detail: pixels (Balestriero & LeCun 2024) and noisy low-ρ features (Littwin) [VR]. Keypoints are already a low-dimensional, denoised abstraction. The empirical record matches: **S-JEPA ≈ MAMP** (85.3 vs 84.9 NTU-60 linear) at about 3× the epochs and about 4.5× the wall-clock [VR]. A hand-crafted motion target closes most of the gap. Expect JEPA's margin over a well-designed motion-MAE baseline to be small. Design experiments so that a null result is informative.
2. **Small data and low dimensionality favour collapse and instability.**
   - T-JEPA needed [REG] tokens to escape collapse [VR].
   - Graph-JEPA reports high variance from trivial contexts [VR].
   - LeWM says SIGReg weakens in low-diversity, low-dimensional settings [VR].
   - ProteinJEPA's JEPA-only variant collapsed downstream [VR].
   - On the other side, VideoMAE trains on 3–4k videos [VR] and LeJEPA trains in-domain on about 1–11k images [VR]. Reconstruction or regularised objectives may simply be the safer small-data choice.
3. **EMA sensitivity and the uninformative loss.**
   - EMA schedules interact with LR, warmup, and weight decay (DirectPred, data2vec) [VR].
   - EMA-JEPA training loss is "a poor proxy for representation quality" and may not decrease monotonically (SALT, LeJEPA) [VR]. Model selection then needs labelled probes or LiDAR; RankMe failed for I-JEPA [VR].
   - V-JEPA 2 found EMA ramps unimportant at scale [VR], but that does not transfer to small runs.
4. **Hidden failure of dense features.**
   - Global metrics can rise while per-token quality decays: DINOv3 after about 200k iterations [VR]; Human-JEPA's "silent collapse" during continued pretraining [VR]; V-JEPA 2's noisy dense maps [VR].
   - For sign language, per-frame features (boundaries, co-articulation) need their own probe and possibly anchoring (Gram anchoring, or a frozen-init dense anchor).
5. **Mask-induced shortcuts.**
   - Interpolation from neighbouring joints (SLiM) [VR].
   - Copy-across-time when block masks repeat over frames (Human-JEPA's "copy task"; V-JEPA's tube rationale) [VR].
   - Conversely, too-small targets or contexts collapse quality (I-JEPA 9–20% vs 54%) [VR].
   - Mask hyperparameters are the most sensitive knob (LiDAR: "extremely sensitive") [VR]. Budget a mask sweep before anything else.
6. **The probe protocol can manufacture conclusions.** Attentive probes add +17 (V-JEPA K400), +7.9 (MAE-B), +17.2 (MAE-S), but only +0.5 for DINO [VR]. A JEPA-vs-contrastive comparison under linear probing alone is biased toward contrastive and invariance methods. Under attentive probing it shifts toward masked-prediction methods.
7. **Contradictory evidence to ablate, not assume.**
   - Stop-grad on targets without EMA: harmful in PointTube [VR], helpful in Laya [VR].
   - Single vs multi-family masks: forecasting vs block involves trade-offs (Human-JEPA) [VR].
   - Context loss helps dense tasks but hurts global ones without deep supervision (V-JEPA 2.1) [VR].
   - Auxiliary objectives trade capabilities (Factorized-LD study) [VR].
8. **Confounds specific to sign data.** Signer identity and recording setup are slow features (Sobal) [VR], and keypoint-detector noise is structured, not i.i.d. JEPA may encode detector artefacts that are temporally stable (e.g., systematic hand-keypoint jitter on certain signers). Evaluate signer-disjoint splits only.
9. **Evidence base.** The SIGReg-instead-of-EMA evidence for *masked token prediction* is small-scale (MSRAction-3D ~94%, EEG) and 2026-only. The LeJEPA paper itself validates multi-crop invariance, not masked prediction [VR].

---

## 8. Minimal ablation grid suggested by the literature (for a skeleton JEPA)

The first two are the first runs to do.
1. **Mask:** semantic part-time tubes vs random joint masking vs a past→future mixture, at 75% and 90%.
2. **Target:** EMA latent (LayerNorm, top-K average) vs motion vector vs hybrid; frozen MAMP-teacher latent (SALT-style).
3. **Anti-collapse:** EMA τ ∈ {0.99, 0.996→1} vs SIGReg λ ∈ {0.01, 0.02, 0.05} on a BatchNorm projector, with and without target stop-grad.
4. **Predictor:** width 1/2 vs 1/4 of the encoder, depth 4 vs 8, kinematic positional encoding vs learned.
5. **Diagnostics in every run:** attentive and linear probe, signer-ID probe, handshape probe, RankMe, LiDAR, and target std.

---

## 9. Source index

ArXiv IDs, all verified by fetching the abs page title:
- **Core vision JEPAs:** 2301.08243 I-JEPA · 2404.08471 V-JEPA · 2506.09985 V-JEPA 2 · 2603.14482 V-JEPA 2.1 · 2307.12698 MC-JEPA · 2410.19560 C-JEPA · 2509.24317 SALT · 2405.17995 DMT-JEPA · 2412.10925 VJ-VCR · 2605.17165 Factorized-LD.
- **Domain JEPAs:** 2311.15830 A-JEPA · 2404.16432 Point-JEPA · 2309.16014 Graph-JEPA · 2410.05016 T-JEPA · 2409.19407 Brain-JEPA · 2505.03176 seq-JEPA · 2501.14622 ACT-JEPA · 2410.08559 ECG-JEPA · 2509.14252 LLM-JEPA · 2502.11831 intuitive physics · 2411.04983 DINO-WM.
- **SIGReg line and 2026 variants:** 2511.08544 LeJEPA · 2603.19312 LeWM · 2606.02572 VISReg · 2603.16281 Laya · 2608.24093 PointTube-JEPA · 2605.07554 ProteinJEPA · 2605.05586 AeroJEPA · 2602.03604 EB-JEPA · 2609.04264 Spectral-target · 2608.21160 Human-JEPA.
- **Theory and collapse:** 2407.03475 Littwin · 2211.10831 slow features · 2102.06810 DirectPred · 2011.10566 SimSiam · 2402.11337 reconstruction-uninformative · 2510.05949 Gaussian embeddings.
- **Teacher-target lineage:** 2202.03555 data2vec · 2212.07525 data2vec 2.0 · 2111.07832 iBOT · 2304.07193 DINOv2 · 2508.10104 DINOv3 · 2112.09133 MaskFeat · 2203.12602 VideoMAE · 2303.16727 VideoMAE V2 · 2407.15447 SIGMA · 2111.06377 MAE.
- **Diagnostics and probing:** 2210.02885 RankMe · 2312.04000 LiDAR · 2506.10178 efficient probing · 2412.03215 Beyond [cls].
- **Skeleton:** 2308.07092 MAMP · 2509.03609 GFP · 2603.10648 SLiM · 2604.28173 Action Motifs.
- **Not on arXiv:** S-JEPA (ECCV 2024, DOI 10.1007/978-3-031-73411-3_21; numbers taken from the GFP and SLiM tables).
- **Not found:** a specific "causal / latent-action JEPA" paper beyond V-JEPA 2-AC, LeWM, and DINO-WM; any sign-language keypoint JEPA.
