# Q1 - JEPA strategy zoo: what is NOT yet covered (as of 2026-10-17)

Scope: gaps left by `proposal12/literature/L1-L3`, `research/literature/B,C`, `proposal16/literature/P2`. Path-JEPA / signature tokens are excluded on purpose.
Raw sources were fetched with curl (`arxiv.org/html/<id>` or `/pdf/<id>` via PyMuPDF) into the scratchpad `q1/raw/`. Numbers were read from table text, not from a summariser.

Tags: **[V]** verified in raw source; **[S]** secondary (search snippet / abstract only, not audited); **[I]** my inference; **[V-local]** verified by reading this repo's code.
Table flattening caveat: where a table came out column-shifted I re-derived the row/column mapping from the surrounding prose; this is noted per item.

## 0. Ten-line summary

1. Theory (Littwin, NeurIPS'24) says a deep JEPA ranks features by **regression coefficient rho (predictability)**, not variance, and an infinitely trained deep JEPA collapses toward a single dominant direction [V]. A clip-constant signer descriptor is the extreme high-rho feature [I].
2. Three independent toy studies (Sobal'22, Gulati-Nemenman'26, MotionJEPA'26) show predictive-latent objectives encode **trajectory-constant nuisance** and suppress fast dynamics; the remedy that works is to change the contrast (**within-trajectory negatives**) or add a dynamics-forcing term [V]. No one has tested this for masked-token JEPA with EMA + centered CE on skeletons [V as absence].
3. ContentVec (ICML'22) is the closest **working recipe** against an identity shortcut in masked prediction: voice-converted teacher input, speaker-perturbed student input with within-utterance negatives, and a **speaker-conditioned predictor**; each removal costs 10-12% relative ABX [V]. Direct analogue: signer-canonicalised teacher + signer-conditioned predictor.
4. IWM (ICLR'24 line): conditioning the predictor and giving it capacity makes the encoder equivariant; a weak/unconditioned predictor makes it invariant. For **frozen linear probes the invariant regime wins** (74.5 vs 67.5 IN-1k linear) [V]. Conditioning on a nuisance that is class-uncorrelated costs accuracy (seq-JEPA: 87.4 -> 79.3) [V].
5. Multi-depth targets are the best-evidenced "free" upgrade outside images: Bootleg (concat of 4 spaced blocks, +15 pts Patch-probe over pixel targets), EEG-JEPA (+3.66 frozen macro-BA from multi-level prediction), V-JEPA 2.1 (restores SSv2 62.5 -> 72.1) [V]. Bootleg also shows **final-layer-only targets degrade with long training** (X-Blk -9 pts by 1200 ep) [V].
6. SIGReg (LeJEPA) is not a drop-in loss for masked-token prediction. Its recipe needs multiple global views (1 global view: 53.1 vs 72.3 at V=4), batch >= 256, SIGReg on the *same* embeddings that enter the alignment loss, small projector [V]. This repo's L1+SIGReg regularises a **side head on student visible tokens**, not the predicted/target space [V-local] - a likely cause of the -13.8.
7. Known SIGReg limits in 2026: low-diversity/low-dimension data weakens it (LeWM), and it keeps slow features while dropping fast ones (MotionJEPA Fig. 1) [V]. Both describe our regime.
8. Online-vs-EMA encoder choice matters: CF-JEPA reports the online encoder is +3.9 pts better for classification than the EMA encoder from the same run (time series) [V]. Check which one we evaluate.
9. Human-JEPA (2608.21160) shows the **mask family sets how much identity survives**: block masks lose 17 mAP re-ID vs forecasting [V]. Mask geometry is an identity-control knob, not only a difficulty knob.
10. No 2026 JEPA paper on sign / hand / skeleton-for-sign was found (searched arXiv, ECCV/ICCV/NeurIPS-2026 pages). Nearest: Action Motifs (3D pose, nested latent masking), SignMAE (reconstruction), Human-JEPA (RGB). SHuBERT is reference-only per project rule.

---

## 1. Q1 - What does JEPA actually learn? (implicit bias, slow features, identity shortcut)

### 1.1 Littwin et al., "How JEPA avoids noisy features" - arXiv 2407.03475, NeurIPS 2024 [V]
- **Setting.** Deep *linear* encoder of depth L, linear decoder; MAE loss vs JEPA loss (stop-grad on target branch). Data: Gaussian, with Sigma_xx and Sigma_yx simultaneously diagonalisable (explicit assumption, authors flag it as the biggest gap). Per-feature parameters: covariance lambda_i and regression coefficient rho_i = lambda_i / sigma_i^2.
- **Result 1 (dynamics).** Both objectives learn greedily by lambda. Only JEPA is also greedy in rho, and the dependence on rho grows with depth: critical-time ratio between two rho values contains a term (Delta-rho)(eps^{1/L}) for JEPA and none at leading order for MAE.
- **Result 2 (fixed point).** For L>1 JEPA suppresses low-rho directions, "this suppression increases with the network depth. At large depths, the encoder becomes low-rank to the point a single dominating eigenvalue ... remains". The dynamics of a 1-layer JEPA match an infinite-depth MAE.
- **Authors' own caveat (Discussion).** The same bias "may shed light on JEPAs vulnerability to spurious or slow features". Also: "predictive yet low-variance features" are favoured.
- **Caveats.** Linear, small-init, diagonalisable; no nonlinear evidence. Depth enters through encoder depth L, not predictor depth.
- **Transfer [I].** A signer-morphology component is constant over a clip, hence almost perfectly predictable from visible tokens (rho -> 1) even though its variance in teacher space may be moderate. Lexical movement features are hard to predict for masked far-time tokens (low rho). Theory therefore predicts: *longer training sharpens signer/slow features and starves movement*. This is consistent with our linear-probe R2<0 on movement and with the 160->200 ep plateau (+0.2 val / -0.05 test), but it is not proof.
- **Minimal test (no new training).** From the stored 40/80/120/160/200-epoch checkpoints, plot per-factor probe score vs epoch: signer-ID acc, handshape R2, movement R2, location R2. Prediction under Littwin: signer-ID and handshape flat/up, movement flat/down after ~100 ep. Cost S (existing eval harness).

### 1.2 Sobal et al., "JEPAs focus on slow features" - arXiv 2211.10831 (NeurIPS 2022 SSL workshop) [V]
- JEPA with VICReg or SimCLR objectives vs reconstruction, moving-dot world, linear probe of dot position. "JEPA methods perform on par or better than reconstruction when distractor noise changes every time step, but fail when the noise is fixed." Also in the 3-dots dataset (action-controlled dot, uncontrollable dot, stationary dot) VICReg/SimCLR "focus only on the stationary dot" while reconstruction captures the controllable and stationary ones.
- Mechanism (their Sec. on spurious correlation): if the forward model converges to identity and a feature is constant over time, a trivial solution that encodes only that feature satisfies both the VICReg variance/covariance terms and the InfoNCE term.
- **Caveats.** Toy, no augmentations, only VICReg and SimCLR JEPAs, no EMA-teacher masked prediction. Authors state the limitation themselves.
- **Relevance.** Closest statement of the "constant-over-time nuisance is the cheapest predictive feature" claim. Our objective is not VICReg/SimCLR, so it is a hypothesis, not a finding, for us.

### 1.3 Gulati & Nemenman, "Contrast encodes inductive bias: separating slow noise from dynamics" - arXiv 2606.07770 (2026-06) [V]
- **Claim.** Failure under fixed noise is a property of **across-trajectory negatives (ATN)**: a trajectory-specific constant separates positives from negatives, so encoding it minimises InfoNCE. Replacing negatives by **within-trajectory negatives (WTN)** (other time points of the same trajectory) removes the shortcut.
- **Evidence.** SimCLR-predictor on moving dot and DySIB on experimental pendulum video. Under fixed noise ATN error saturates near random guessing as noise strength grows; WTN tracks the supervised baseline over the full noise range. The 2-D pendulum latent breaks into trajectory-specific clusters under ATN and stays a clean angle ring under WTN.
- **Limits stated by the authors.** (i) WTN works only if each trajectory spans enough phase space; stationary trajectories cannot help. (ii) Residual error **peaks at noise correlation time tau ~ T** (signal and nuisance share a time-scale), shrinking as T grows (power law in T). (iii) Cost WTN O(T^2 N) vs ATN O(T N^2). (iv) Moving-dot/pendulum only; JEPA/EMA integration listed as future work. They also suggest mixing within- and across-trajectory negatives on a schedule.
- **Transfer [I].** Our centered-CE target is over a batch-centred softmax across feature dimension; it has no explicit negatives, but the *centre* is a batch statistic (so the between-clip signer axis is not removed). The WTN principle translates to "subtract/contrast within the clip". **Risk from (ii):** a held handshape is also nearly clip-constant (tau ~ T), so within-clip contrast will also under-weight held lexical features. Needs a partial (alpha<1) variant.
- **Minimal test.** Offline, on frozen teacher latents of val clips: split variance into between-clip and within-clip parts per channel; probe signer-ID from the clip-mean component vs from the residual; probe sign class from both. If the residual keeps >=90% of the sign-class accuracy while dropping signer-ID, within-clip targets are safe (S, CPU).

### 1.4 MotionJEPA (DISReg) - arXiv 2609.23881 (2026-09-20) [V]
- Claim: "standard JEPA training exhibits a strong inductive bias towards slow features, causing feature suppression". Fig. 1 (Pong: fast ball, bats, slow scoreboard): SIGReg-based LeWM encodes scoreboard and bats but **suppresses the ball**; inverse-dynamics (SMWM) keeps only action-controlled bats; DISReg keeps all three.
- Method: DiffEnc/DiffPred predict the embedding of the *temporal difference image* from (z_t, z_t+1); SIGReg on z and on d; weights lambda_z=0.25, lambda_pred=0.5, lambda_d=2 (fixed for all environments).
- Numbers (Table 3, 50-step planning block, mean over 4 envs; column order inferred from flattened text): LeWM 11.8, LeWM-Flat 31.9, IDM 59.3, DISReg 70.3, DISReg without lambda_z 49.2, DISReg+IDM 71.6 [V, column mapping I].
- Baselines that did **not** fix it: SIGReg along time (LeWM-Time), flattened batch x time (LeWM-Flat), detach/stop-grad; a hyper-parameter search per baseline "to demonstrate that the collapse is not an issue of hyperparameter selection". Loss curves: baselines plateau in probe NMSE despite continued training.
- Cited related: Gulati'26 (2606.07770) for "reproduced across noise timescales", SMWM 2606.20104 (inverse dynamics regulariser; abstract [V]).
- **Caveats.** Synthetic games, MLP probe (not linear), planning success as downstream; no masked-token JEPA.
- **Transfer.** Motion is the thing we under-encode (R2<0). A DISReg-lite on skeletons: predict the embedding of the movement/delta-channel token of bin t+1 from embeddings of adjacent bins (see strategy 5 in Sec. 7).

### 1.5 Provable benefits and the augmentation-alignment condition - Van Assel et al., arXiv 2505.12477 (NeurIPS 2025) [V abstract only]
- Closed-form linear analysis: "both SSL paradigms require a minimal alignment between augmentations and irrelevant features to achieve asymptotic optimality"; when irrelevant features have large magnitude, joint-embedding needs a strictly weaker alignment condition than reconstruction.
- **Relevance [I].** Signer morphology is a large-magnitude irrelevant feature. The theorem says the *augmentations* must act on it (bone-length / body-proportion perturbations); without that, neither MAE nor JEPA is guaranteed to discard it. Only the abstract was audited (HTML fetch was partial).

### 1.6 Idealised-dynamics view - Ruiz Morales et al., 2511.09783 [V abstract]
JEPA with a near-identity linear predictor recovers regime indicator functions (Koopman eigenfunctions). Already in C Sec. 1.3; kept here only to note it is consistent with 1.1-1.3: a near-identity predictor makes "the same across time" the optimum.

### 1.7 Is there evidence for identity shortcut in data2vec/BYOL/HuBERT-style masked prediction?
- **HuBERT-style masked cluster prediction retains speaker info** - ContentVec, 2204.09224 (ICML 2022) [V]: speaker-ID accuracy of ContentVec is "as high as 36%" lower than HuBERT-iter; HuBERT-iter is itself slightly lower than HuBERT (iterating the teacher removes some speaker info). So masked prediction does carry speaker identity by default, and it takes explicit machinery to remove it.
- **Speech S-JEPA** (2606.19398, soft-GMM targets): reports linear-probe Speaker ID and Gender at ceiling and Chapter-ID 93.2 vs HuBERT 88.2 [V]. Speaker information is not removed by soft cluster targets; no controlled ablation (authors say so).
- **Human-JEPA** (2608.21160) [V]: identity (Market-1501 re-ID, frozen BNNeck probe) is *added or destroyed* by the mask family: forecasting split 0.4635 mAP vs base V-JEPA 2.1-L 0.4370 vs **block-mask variant 0.2666** (same data/anchor/budget). Block inpainting is "a temporally local copy task" per the authors.
- **Missing: any controlled test of identity decodability in an EMA + masked-token JEPA vs MAE vs random-init encoder.** Our 94% signer decodability is uninterpretable without an **untrained-encoder baseline and a handcrafted-descriptor baseline** (MotionJEPA includes a "Random" arm for exactly this reason).
- **Minimal test (S).** Signer-ID linear/kNN accuracy for (a) random-init PLctx, (b) B0 handcrafted descriptor, (c) PLctx-160, (d) an MAE-target variant if one exists. Report as retention ratio vs (a).

### 1.8 Verdict on the key question
Does masked-latent prediction *structurally* favour clip-constant nuisance? Evidence level: **plausible, supported by theory (1.1) and by three toy experiments with other objectives (1.2-1.4), contradicted by none, but untested for EMA + centered-CE masked tokens.** A skeleton adds a confound: signer morphology is directly present in the keypoint geometry, so even random features decode it. The decisive pre-registered measurement is the retention ratio in 1.7.

---

## 2. Q2 - Conditioned / transformation-aware predictors

### 2.1 Image World Models (IWM) - Garrido et al., arXiv 2403.00504 (2024-03) [V]
- **Setup.** I-JEPA-style EMA target; source view is the target view plus further colour jitter and destructive aug (grayscale, blur, solarisation) plus I-JEPA masking. Predictor receives mask tokens plus transformation parameters a_{x->y}. Target never gets destructive aug.
- **Three controls of the world model** (all [V], MRR = rank of the true augmented view in a 256-bank; 0 = no equivariance, 1 = perfect):
  - Conditioning: none 0.00 / sequence 0.82 / feature 0.79 (Table 1).
  - Capacity x aug (Table 2): I-JEPA predictor (12,384): jitter 0.00, +destructive 0.00, +strong jitter 0.00; IWM (12,384): 0.11 / 0.09 / 0.81; IWM (18,384): 0.25 / 0.79 / 0.85. "For 12 layers, jitter equivariance is achieved 1 out of 5 times whereas for 18 layers 4 out of 5 times."
  - Stronger augmentation strength makes a stronger world model; a weak task makes the predictor useless.
- **Invariance vs equivariance and the readout (Table 8, ViT-B, 300 ep, IN-1k)**: MoCo v3 76.3 lin / 76.4 attn; MAE 60.2 / 73.5; I-JEPA 70.0 / 75.0; **IWM-Inv (12,384) 74.5 / 77.0**; **IWM-Equi (18,384) 67.5 / 75.1**. "More invariant world models excel in linear evaluation and equivariant world models shine with larger evaluation heads"; Fig. 4: equivariance is hardly correlated with attentive-probe accuracy.
- **Caveats.** ImageNet ViT-B, photometric augmentation only, 300 epochs, one seed per cell. The "invariance" is invariance to *the augmentation family*, not to an arbitrary nuisance.
- **Transfer verdict.** Our readout (mean-pool -> L2 -> cosine, class = max over gallery) is even simpler than a linear probe, so the invariant regime is the right one **with respect to signer** - which is why a *signer-conditioned* predictor (Sec. 2.3) is more apt than IWM's augmentation-conditioning (which would add equivariance to nuisance). Predictor depth is the knob: do not deepen. 
- **Minimal test (S).** Predictor depth sweep {1,2,4,8} at the 4k-step pilot scale; report retrieval R@1 together with signer-ID and per-factor R2. IWM predicts a monotone invariance -> equivariance drift with depth; if R@1 falls with depth the regime assumption holds.

### 2.2 seq-JEPA - arXiv 2505.03176 (NeurIPS 2025) [V]
- View sequence + relative-action embeddings -> transformer aggregate -> predictor conditioned on the next action. Two separate readouts: per-view encoder output (equivariant) and aggregate token (invariant).
- **3DIEBench (Table 1)**: BYOL 82.90 / rel-rot R2 0.12; SimCLR 81.13 / 0.35; EquiMod 84.29 / 0.32; SIE 77.49 / 0.58; Conditional BYOL 82.61 / 0.31; **seq-JEPA(3,5) 87.41 / 0.71**; seq-JEPA without action conditioning 86.05 / 0.29. So action-conditioning moves equivariance 0.29 -> 0.71 with no accuracy cost (86.05 -> 87.41); longer view sequences carry additional classification gain ((1,1) 84.08 -> (3,5) 87.41).
- **Table 4 ablation** (3 seeds): no predictor conditioning 87.17 / 0.37; no transformer conditioning 86.33 / 0.53; default 87.41 / 0.71.
- **Nuisance-conditioning risk (Table 7, appendix)**: conditioning on rotation *and colour* (colour is weakly class-correlated): SEN 82.17, EquiMod 82.58, SIE 75.34, **seq-JEPA 79.31** (from 87.41), colour R2 ~0.95-0.97. The authors cite Wang et al. 2024 "Principle II": forcing the encoder to encode class-irrelevant factors costs class information.
- **Caveat.** Conditioning *adds* nuisance information to the encoder's output (equivariance). Opposite sign of what ContentVec does (below).

### 2.3 Conditioning the predictor to *release* the encoder from a nuisance - ContentVec [V]
- Mechanism (Sec. 3.5): teacher labels still leak some speaker information, so "student representations are undesirably forced to carry the same amount of speaker information ... To break this entailment ... we feed the speaker embeddings to the predictor" (pretrained GE2E embedding of the same utterance).
- **Ablation (Table 4; ABX-within / ABX-across lower is better, PNMI higher)**: full 5.13 / 6.32 / 0.590; No-Disentangle-Teachers 5.67 / 6.92 / 0.576; No-Disentangle-Students 5.62 / 6.93 / 0.577; **No-Conditioning 5.76 / 7.08 / 0.593** (PNMI barely changes, ABX degrades 12%). Contrastive-loss weight is robust (ABX(w) 5.10-5.22 over lambda 1e-6..5e-5) but its **layer position matters**: at layer 1 ABX(w) 5.76, layer 4 5.19, layer 8 5.14.
- Student disentanglement uses **negatives drawn from the same utterance** (their Eq. 2) - the WTN principle of Sec. 1.3, arrived at independently.
- Random "speaker-only" transform = formant scaling x F0 scaling x random equaliser, rho in U[1,1.4] with 50% reciprocal.
- **Caveats.** Speech content tasks (ABX, PNMI, SUPERB content), HuBERT-style *hard cluster* targets, speaker embedding supplied by a *pretrained external* network (needs speaker-labelled data elsewhere). Gains are 0.5-0.7 ABX points; no seed variance reported.
- **Transfer verdict [I].** The skeleton analogue of GE2E is a deterministic **morphology descriptor** (median-frame bone-length ratios, shoulder/hip width, hand-to-forearm ratio, hand-size) or a learned clip-level signer embedding from a small encoder trained with signer-ID on the *training signers only*. Condition the predictor (AdaLN / extra token), not the encoder. Expected sign: lower signer decodability, higher retrieval. Risk: the descriptor may leak lexical info if computed from non-neutral frames; restrict to bone-length ratios that are posture invariant.
- **Minimal test (M).** One 4k-step pilot, +signer-descriptor token to the predictor only; compare to baseline on R@1, signer-ID, probe R2. Control: shuffle descriptors across clips (should equal baseline or worse).

### 2.4 Other conditioned-predictor evidence
- **EquiMod** (2211.01244) [V text]: predictor in projector space conditioned on augmentation parameters; adds equivariance to BYOL/SimCLR/Barlow; "EquiMod increases the performances of all the baselines used, except BYOL while trained on 1000 epochs" (linear eval, ImageNet/CIFAR10); table cells column-shifted in the flattened text, so exact numbers are not quoted. Note the paper's own failure mode: an invariant latent with an identity predictor is an admissible solution - their equivariance metrics (absolute / relative displacement) detect it.
- **AugSelf** 2111.09613 [V abstract]: auxiliary head predicts the *difference of augmentation parameters* between two views; transfer gain when downstream depends on the augmentation.
- **LEPA** 2603.07246 (2026) [V]: replicates the IWM finding for geometric transforms on satellite imagery: "conditioning the predictor on augmentation parameters markedly improves equivariance, MRR increases from around 0.2 to nearly 0.7"; predictor fine-tuned encoder-frozen reaches 0.8. Training with augmentation prediction as the *exclusive* objective fails (needs inpainting as well).
- **V-JEPA 2-AC** 2506.09985 [V]: the action-conditioned predictor is trained on a **frozen** encoder - so action conditioning there does not alter the encoder's invariance at all.
- **UniJEPA** 2608.07409 [S]: claims photometric prediction learns invariant structure and temporal prediction equivariant structure, ImageNet 74.9 vs IWM 73.5. Numbers look unusually clean and I did not audit the training details; treat as unreplicated.
- **LRC-JEPA** 2609.34375 [V abstract]: routes "predictable visual context" to a separate residual embedding u so the dynamics latent z stays small; claims sufficiency/minimality under assumptions. Same idea as a nuisance-side channel; unreplicated.
- **No 2025-26 paper found** that conditions a JEPA predictor on an *identity/nuisance* embedding for non-speech data.

### 2.5 Synthesis Q2
Predictor capacity and conditioning control what the encoder must carry: whatever the predictor can obtain from its conditioning vector (augmentation parameters, speaker embedding) the encoder is *free to drop*; whatever the predictor cannot undo (destroyed information) the encoder must keep to be equivariant. For frozen retrieval, (a) keep the predictor shallow, (b) condition on nuisance side-information that is **available at train time but not part of the lexical signal**, (c) avoid conditioning that *adds* class-irrelevant factors to the encoder output. Conflicting-regime caveat: Human-JEPA/IWM numbers are RGB; none of this is shown on tokens with 7 channels x 32 bins.

---

## 3. Q3 - Multi-scale / hierarchical / block-level targets

### 3.1 Bootleg - 2603.15553 (Lowe et al., 2026-03; NeurIPS'26 poster title "...Masked...") [V]
(Overlaps C Sec. 1.2 for Table 2 and the tau sweep; only the delta is given.)
- **Duration collapse (Table 27, ViT-S, IN-1k, final hparams; columns Patch / CLS / X-Blk / ADE kNN):**
  | Epochs | targets {1,4,8,12} | block-12-only |
  |---|---|---|
  | 100 | 59.0 / 59.4 / 70.0 / 17.7 | 61.0 / 60.2 / 69.9 / 17.1 |
  | 200 | 65.5 / 66.3 / 73.2 / 20.9 | 59.3 / 57.1 / 69.6 / 16.2 |
  | 600 | 69.8 / 70.4 / 75.3 / 23.5 | 47.9 / 46.4 / 64.4 / 7.5 |
  | 1200 | 69.4 / 70.2 / 75.0 / 22.5 | 43.8 / 41.2 / 60.0 / 5.5 |
  Final-layer self-distillation "peaks early (~100 ep) then degrades" with embedding-rank collapse; the training loss decreases monotonically while probes fall (Sec. K.3). Multi-target is stable to 1200 ep.
- **Masking must support deep targets**: with MAE random masks, hidden-layer targets deeper than block 4 collapse the model; the fix is block masks (their Sec. 3.2, Table 3: adding hidden self-distillation gives MAE/CrossMAE +10 Patch/CLS probe, I-JEPA +6, data2vec 2.0 concatenation +2 but 'every 4th layer' -1).
- **Relevance.** Our plateau at ~160 ep with final-layer teacher targets (centered CE over final teacher output) is the same shape as Bootleg's b12-only curve, but ours does not degrade (just flat) - [I] we may be at a milder stage of the same drift. The effective-rank trace over our checkpoints is the cheap diagnostic (C Sec. 1.7 template).
- **Caveat.** Images, ViT-S/B/L, ImageNet probes; encoder here is 12-block, ours is 6-block (use {1,3,6} or {2,4,6}).

### 3.2 V-JEPA 2.1 deep self-supervision - 2603.14482 [V]
- Table 1 (ViT-L, row labels follow the numbers in the flattened text; verified against the prose): V-JEPA 2: IN1K 82.2, SSv2 72.8, NYU RMSE 0.682, ADE 22.2 -> **+ context loss** 72.6 / 62.5 / 0.474 / 33.8 -> **+ multi-level prediction** 80.8 / 72.1 / 0.463 / 38.6. The text says deep self-supervision "restores the classification performance".
- Design: concatenate normalised outputs of 4 equally spaced encoder blocks -> MLP fuse -> predictor outputs 4 heads, each regressed on the matching teacher level; loss applied at all four levels and on both masked and context tokens (distance-weighted for context). Table 13 (App. D): with deep supervision, the last layer alone suffices for dense probes.
- **Relevance.** The attractive property for us: deep supervision makes *last-layer* features carry local detail, i.e. the frozen embedding we retrieve with is less low-level-starved.

### 3.3 EEG-JEPA (structured latent prediction) - 2608.00114 (2026-07-31) [V]
- Target design along content / support / **depth** (L3, L6, L9, final). Frozen 14-task macro BA (Table 6): MAE control 40.49; independent target 44.79; EMA target 45.12; + N-MET structured masking 46.47; **+ multi-level prediction 50.13 (+3.66)**; + context consistency 50.42; without VCReg 50.09; the multisource stage-2 continuation adds 2.52 to reach 52.94.
- The authors' rationale for latent targets on noisy biosignals: reconstruction "may encourage models to recover predictable background activity, acquisition effects, and artifacts" - the same worry as signer morphology.
- **Caveat.** EEG-FM-Bench, one seed per row (std 0.3-0.6 reported).

### 3.4 Time-series/audio JEPAs - what transfers to a 1-D token sequence
- **WavJEPA** 2509.23238 [V]: ablations on HEAR mean score s(m): context block count M_ctx = 5/10/15 -> 66.2/66.0/64.0 ("little impact"); target sampling probability p_target = 0.15/0.20/0.25/0.30 -> 64.8/65.9/66.0/63.0; **"highly distributed training targets were consistently suboptimal for scene analysis and speech tasks"** (contiguous spans better); top-K layer averaging of the teacher improves all HEAR tasks, optimum task-dependent (K=8 beat K=12 on LibriCount, ESC-50, CREMA-D). Note: Bootleg shows *averaging* is worse than concatenation (Sec. 3.1).
- **CF-JEPA** 2606.07031 [V] (time series, Knowledge-Based Systems; 126 UCR + 26 UEA): replaces masking by forward prediction at short/mid/long horizon zones. Ablation: 3-zone forward vs masked-latent (TS-JEPA style) 0.819 vs 0.789 UCR, 0.717 vs 0.706 UEA; single-zone vs 3-zone: classification equal (-0.3 / -0.2 pts), multivariate forecasting MSE +3.3%. **Online vs EMA encoder (Table 13)**: classification UCR 0.819 vs 0.780, UEA 0.717 vs 0.689 (online better); forecasting MSE 1.400 (online) vs 1.016 (EMA), kNN anomaly F1 0.389 vs 0.501. EMA encoder = lower effective rank, smoother. The paper's own diagnosis: "high rank for sample-level discrimination; smooth trajectories for per-timestep scoring".
- **MTS-JEPA / SC-JEPA** 2602.04643 (v1 titled MTS-JEPA; later retitled SC-JEPA) [V]: soft-codebook bottleneck + multi-resolution objective for anomaly *prediction*. Ablation Table 3: removing the codebook module collapses AUC to 43-51 on all five datasets (e.g. MSL F1 33.58 -> 21.82; PSM AUC 77.85 -> 46.61); removing only the auxiliary codebook losses costs far less; removing temporal downsampling (the multi-resolution term) costs 1.4-6.1 AUC pts depending on dataset. Task (anomaly prediction) is far from retrieval; the useful generic point is that a **bounded simplex target (KL on codes)** stabilises self-distillation, which our centered-CE already is.
- **TS-JEPA** 2509.25449 [V via C]: ~MAE on classification; not repeated.
- **A-JEPA** (2311.15830): curriculum random-block -> time-frequency masking; no direct ablation of the curriculum was found in the sources I could access [S]; not re-covered.
- **Human-JEPA** 2608.21160 [V]: pure past-to-future split vs block masks, same recipe: NTU-120 action tax 4.8 pts (block) vs 2.7 (residual) and **re-ID collapse of 17 mAP** with block masks; mixing the two mask families per batch "traverses a clean monotone trade-off"; Fig. 3: "naive continued pretraining collapses dense perception while its training loss improves"; dense targets pinned to a frozen copy of the initialisation ("anchored") prevent this but alone plateau.
- **H-JEPA** 2610.06805 (2026-10-05) [V abstract]: hierarchy of action-conditioned JEPAs, higher levels "discard fast, unpredictable detail and retain slower task-relevant state"; planning gains, SIGReg used. **Wrong direction for us**: the thing we under-encode is fast detail.
- **Pooled / span-level targets**: I found **no** paper that ablates pooling a teacher target over a window before regression in masked prediction. Closest are WavJEPA target-length (contiguous > distributed) and CF-JEPA multi-horizon. Treat "segment-level pooled target" as untested [I]; cost to test is S (a 2x/4x average-pool over the 32 time bins of the teacher output before the CE head).

### 3.5 data2vec 2.0 multi-mask amortisation - 2212.07525 [V]
- One teacher forward is reused for M masked student views. Figure 3: at batch 64, M=16 vs M=2 raises accuracy by 4.6% "keeping everything else equal"; inverse block masking (B=3 default) beats block and random masking; vision runs use M=8 (R=0.8 mask), speech M=8 (R=0.5). Targets: top-K FFN outputs, instance-normalised, averaged.
- **Relevance.** Small-batch, low-compute regimes (the pose_jepa full-ASLC config here uses per_rank_batch 32 [V-local]; PLctx batch not checked) gain from more masks per teacher pass; also yields many masked views of *the same clip*, which is the within-clip structure Sec. 1.3 wants. No evidence it was ever combined with within-clip contrast.

### 3.6 Synthesis Q3
(a) Multi-depth concatenated targets: 3 independent domains agree (+3.66 EEG frozen, +10-15 Bootleg Patch, V-JEPA 2.1 restoring classification). (b) Contiguous target spans beat scattered ones for speech (WavJEPA). (c) Online/student encoders are sometimes better for discrimination than EMA/teacher encoders (CF-JEPA). (d) Hierarchy/pooling toward slow state is wrong for fast-feature deficits (H-JEPA) but is unproven either way for pooled targets. (e) Mask family modulates identity retention (Human-JEPA).

---

## 4. Q4 - Collapse-proof objectives beyond EMA; why our SIGReg pilot may have failed

### 4.1 LeJEPA / SIGReg - 2511.08544 [V claims and ablations]
- **Loss (Eq. 9):** L = lambda * mean_v SIGReg({z_{n,v}}_n) + (1-lambda) * mean_n L_pred, with L_pred = (1/V) sum_{v'} || mu_n - z_{n,v'} ||^2, mu_n the mean of the **Vg global-view embeddings**; every view predicts the global-view mean (DINO multi-crop layout). SIGReg is applied **to the same embeddings** used in L_pred, across the batch dimension B.
- **Recommended defaults (Sec. 6):** lambda=0.05; Vg=2, Vl=8; **batch >= 128**; 1024 slices; 17 quadrature points; integration domain [-5,5].
- **Ablations, ViT-L/14, IN-1k, 100 ep, linear (Table 1):**
  - (b) views: V=4: Vg=1 -> **53.06**, Vg=2 -> **72.26**; V=10: Vg=1 68.97, Vg=2 74.06, Vg=4 75.08.
  - (c) batch size: 128 -> 72.20, 256 -> 74.15, 512 -> 74.72, 1024 -> 74.07.
  - (a) integration domain: [-1,1] 71.8-72.9 vs [-3,3]/[-5,5] 73.7-75.0; slices 512 vs 2048 +0.7-1.1.
  - (d) projector dim: 64 -> 75.29-75.65 (best), 1024 -> 73.65-74.79. Smaller projector better.
- **Claims to keep in perspective.** "No stop-gradient, teacher-student, predictor needed" is shown with multi-view *invariance* training on images; the paper does not train masked-token prediction. They state predictors/teacher mainly prevent collapse and "a teacher-student configuration does provide a small performance boost for ViT models". Loss correlates with downstream accuracy (Spearman up to 99% after a lambda^alpha rescaling of the loss) - for *their* setting.
- **When Does LeJEPA Learn a World Model?** 2605.26379 [V abstract]: linear identifiability proven for stationary additive-noise latent dynamics, and the Gaussian is the *unique* latent law that gives the guarantee. Skeleton/phonological latents are neither stationary nor Gaussian; no guarantee transfers.

### 4.2 Documented SIGReg failure modes (2026)
- **LeWorldModel** 2603.19312 [V]: "low data diversity weakens SIGReg in simple, low-dimensional environments where matching a high-dimensional Gaussian prior is harder"; SIGReg is applied to the [CLS] projector output per time step across the batch; embedding dim saturates beyond a threshold; lambda in [0.01, 0.2] robust (>80% success), peaks ~0.09, degrades sharply at 0.5 where the regulariser dominates dynamics modelling.
- **MotionJEPA** [V]: SIGReg-based LeWM encodes the slow scoreboard and bats and suppresses the fast ball (Fig. 1); SIGReg-over-time, flat and detach variants do not fix it.
- **Rectified LpJEPA** 2602.01456 (ICML 2026 per the authors' repo page [S]; numbers [V]): sliced distribution matching with *random* directions loses to VICReg on ImageNet-1K at 100 epochs: encoder top-1 57.95 (RGN with random projections) vs VICReg 60.52 vs **60.63 with eigenvector-augmented projections** (projector acc 46.07 / 46.23 / 48.84). Batch-size dependency (CIFAR-100, encoder acc): 24 -> 64.88, 64 -> 67.59, 128 -> 67.69, 256 -> 68.29, 512 -> 67.88 (saturates at 128-256). So random slicing in a strongly anisotropic space converges slowly; data-aware slices help.
- **TC-SIGReg, LeNEPA, GLaS, Laya, lambda-JEPA** (P2 Sec. 2.2, not re-verified here): token-level SIGReg works for sequence models only with temporal/residual formulations and with the projector small.

### 4.3 Why our L1+SIGReg pilot was probably worse (-13.8 R@1; 28.07 vs 41.87 ASLC300 retrieval R@1 at 4k steps, from `proposal12/PhonoJEPA_Strategies_Deep_Research_VI.md` Sec. 4.1 [V-local doc]) [I from code + papers]
Code read: `pj/train.py` lines 230-234: `proj = m.sig_tok(h_s[vis])` then `ls = sig(proj) + sig(sig_tok(cls_s))`; prediction loss `|pred - tgt|.mean(-1)` on teacher outputs. [V-local]
1. **Regulariser not on the loss path.** SIGReg sits on a side head of *student visible-token* features; the L1 alignment is between *predictor outputs* and *teacher targets*. LeJEPA regularises the same embeddings that are aligned. A side-head SIGReg only indirectly shapes the predicted/target space.
2. **No multi-view invariance term.** LeJEPA with one global view collapses to 53 vs 72 (Table 1b). Our L1 masked-prediction term is a single-view alignment.
3. **Token samples are not i.i.d.** SIGReg's guarantees assume i.i.d. samples; visible tokens of one clip are strongly correlated, so effective N ~ batch of clips (the pose_jepa full-ASLC config in this repo uses per_rank_batch 32; I did not check the PLctx pilot batch). LeJEPA and LpJEPA both show degradation below ~128-256 independent samples.
4. **Low-diversity/low-dim data** (LeWM limitation) and strongly anisotropic features (LpJEPA eigenvector finding): Gaussian marginals force noise dimensions to fill the budget; random slices under-sample the few informative directions.
5. **Short schedule.** 4k steps; the no-EMA SIGReg variant did better (33.07 vs 28.07) which suggests double regularisation (EMA + SIGReg) rather than too little. Laya (P2) reports stop-grad effects growing with training length; a 4k pilot cannot show the late-phase behaviour.
6. **Slow-feature bias is regulariser-dependent** (MotionJEPA): isotropy over the whole batch does not by itself rebalance signer vs movement.
- **A fair retry (not a replacement) [I].** Keep centered-CE + EMA; add SIGReg as an auxiliary on the **teacher-aligned pooled predictor output / clip embedding** with batch >=256 clips (accumulate with queue or use the dataset-level cache from 2610.20158, P2), small 64-128 projector, lambda 0.02-0.05, >=2 near/far views. Compare R@1 and RankMe on the backbone (P2: regulariser after projector leaves backbone low-rank).

### 4.4 EMA-free alternatives worth a pointer only
SALT frozen teacher (L1), Human-JEPA anchored (frozen initialisation) targets for continued training [V], SMWM inverse-dynamics regulariser (2606.20104) [V abstract], VJEPA variational (2601.14354) [V abstract]. For a **continuation run from the 160-ep checkpoint**, the frozen-copy teacher is the one with direct evidence (Human-JEPA Fig. 3): naive continuation collapses silently while the loss improves, anchor-only never collapses but plateaus; the full recipe (anchor + something that adds signal) rises. Our "+200 ep" experiment is exactly a naive continuation.

---

## 5. Q5 - Target design for sparse / low-SNR tokens; discrete and soft targets; teacher-input asymmetry

### 5.1 Distributional / discrete targets
- **S-JEPA (speech), soft GMM posteriors** 2606.19398 [V]: JEPA encoder-predictor trained with KL to GMM posteriors at masked frames; SUPERB (frozen) WER 12.10 vs DistilHuBERT 13.37 (<90M); ER 64.83; predictor entropy bimodal (confident <0.3 bit, a mode near 1 bit at two-way ties). No controlled hard-vs-soft ablation (authors say so).
- **Which clusters get the residual mass matters** - 2608.19084 [V]: counterfactual targets keep top-1 and all probability values but reassign the remaining mass. Three seeds, 20 h LibriSpeech, 51 speakers: REAL SOFT vs RANDOM-REASSIGN gains: DeepTailCE -0.0679/-0.0135/-0.0402, controlled dynamic R2 +0.0262/+0.0247/+0.0167, **phone acc +1.55/+0.12/+1.07 pp**; graded exposure slopes consistent in sign in all seeds. Effect sizes small, one config. Implication [I]: the *dark knowledge* beyond top-1 carries signal; our teacher temperature tau_t=0.06 (centred CE) sharpens it - a tau_t sweep is a cheap test.
- **DinoSR** 2305.10005 [V] (online clustering on EMA teacher layers, iterated implicitly): SUPERB PR PER 3.21 vs data2vec 4.69, ASR WER 4.71 vs 4.94; codebook size V from 64 to 2048 "only changed WER by a small margin"; choice of top-N layers to cluster matters more (best 6-10); codebook decay 0.5-0.99 fine, 0.999 worse (9.43 vs 8.40 WER). Online codebook has fewer active codes than size (217/256 active).
- **SC-JEPA** (3.4) [V]: removing the bounded code simplex -> collapse.
- **ContentVec teacher labels** [V]: k-means on HuBERT features of voice-converted audio; "the student can still preserve the content far better than the teacher".
- **No JEPA paper found with iterated cluster-ID prediction on skeleton/pose** (SHuBERT does it for sign but is reference-only here).
- **Noisy-target robustness.** Bootleg Table 33 [V]: z-scoring each layer's target separately beats joint standardisation; target statistics logged in Table 34. No paper I found handles per-token *missing/low-confidence keypoint* targets in JEPA; MotionJEPA's DISReg uses difference *embeddings*, not confidence weights. [I] Weighted-token CE (we already pass w) is the standard answer.

### 5.2 Teacher-input asymmetry ("privileged" teacher)
- **ContentVec** [V]: teacher input = utterance voice-converted to one speaker; student input = speaker-perturbed copies; No-Disentangle-Teachers costs ABX(w) 5.13 -> 5.67, ABX(a) 6.32 -> 6.92. The same label set is used for both student copies (Eq. 3).
- **Data2Vec-denoising** 2403.18560 [V] (keyword spotting): student sees noisy input, teacher clean. Mean accuracy over noise levels, KWT-1 seen noise: Data2Vec-clean 0.606; Data2Vec-clean+noisy 0.693; Data2Vec-noisy (both noisy) 0.692; **Data2Vec-denoising + noisy 0.708**; unseen noise 0.601/0.676/0.676/0.688. Only +1.5 pts over noisy-for-both; KWT task, downstream fine-tune.
- **DINO local-to-global** 2104.14294 [V text]: all crops through student, only global views through the teacher; the standard form of asymmetry (our near/far views follow this).
- **LeJEPA** [V]: "predictors in image JEPA (without asymmetric information) and teacher-student architectures serve primarily to prevent collapse" - i.e. asymmetry is usually justified as anti-collapse, not as nuisance control; **ContentVec and Data2Vec-denoising are the only places it is used to control a nuisance**, and neither is formalised as theory (I found no paper that does).
- **Human-JEPA** [V]: the teacher is a *frozen copy of the initialisation* for dense targets (anchoring), i.e. asymmetry in *time* rather than input.
- **Transfer [I].** Canonicalised teacher input = retarget the skeleton to a canonical body (rescale each bone to median length; normalise hand/face scale; mirror handedness) before the EMA teacher; student sees randomly de-canonicalised copies. Cheaper than voice conversion: no learned model, just a deterministic bone-length retarget in 2D (limitation: 2D ratios are view dependent; wrist/finger pose distortions). Pair with within-clip negatives for the student.

---

## 6. Q6 - 2026 JEPA on skeleton / motion / pose / hand / gesture / sign: nothing missed that matters

Searched (arXiv, project pages, ECCV/ICCV/NeurIPS-2026 listings) for combinations of JEPA x sign / skeleton / hand / gesture / EMG / pose. Result: **no JEPA paper on sign language, hand pose, or skeleton-for-sign**.

| Paper | What it is | Status vs. our setting |
|---|---|---|
| Action Motifs 2604.28173 [V abstract] | 3D pose, nested latent transformer, masked token prediction in latent spaces, variable-length segments as single tokens | Already in L1/P2 |
| Human-JEPA 2608.21160 [V] | RGB video, forecasting vs block masks, re-ID/pose/action frozen probes | **New**: identity control by mask family; anchored targets (Sec. 3.4, 4.4) |
| WiFi-JEPA 2607.11064 [S] | WiFi-CSI 3D pose, latent masked prediction, ECCV 2026 | In L3 |
| EgoJEPA 2610.20904 [V abstract] | JEPA-style visual prediction conditioned on hand-motion differences (latent actions) for robot manipulation | Hand-motion-conditioned predictor; manipulation target; not retrieval |
| Skeleton GFP 2509.03609 [V abstract] | high-level feature prediction from a lightweight target generator, 6.2x faster training | Already in P12/P14 docs |
| SignMAE 2605.02094 [V abstract] | segmentation-masked mask-and-reconstruct (video+keypoints), WLASL / NMFs-CSL / Slovo | Reconstruction, not JEPA; covered in A |
| SignReGraph (Multimedia Systems, Oct 2026) [S] | skeleton GCN with masked frame reconstruction | Covered in A |
| T-MOR 2606.21607 [V abstract] | skeleton motion contrastively aligned with video+text | Contrastive; not JEPA |
| SOfA 2609.07078 [V abstract] | cross-sensor skeleton foundation model with canonical joint slots | In P2 |
| EMG-GPT 2610.05235 [V abstract] | predictive pretraining on RVQ EMG tokens for hand pose | Discrete-token predictive, not JEPA |
| Adaptive masked reconstruction 2606.11450 [V abstract] | skeleton MAE, decoupled decoder, motion-informativeness guidance | In L2/L3 |

---

## 7. Candidate strategies ranked for our setting

Ranking criterion: expected effect on frozen cosine retrieval x evidence strength / cost; cost S (< 1 GPU-day or CPU only), M (1-3 GPU-days), L (> 3). The same ASLC300 / full-ASLC retrieval harness is assumed.

| # | Strategy | Mechanism (evidence) | Expected effect on frozen retrieval | Risk | Cost | Minimal test |
|---|---|---|---|---|---|---|
| 0 | **Diagnostics before training**: retention ratio + oracle headroom + readout choice | Sec. 1.7 (untrained baseline), CF-JEPA online-vs-EMA (+3.9), Bootleg/LeNEPA layer choice | Tells whether the signer shortcut is real and how much R@1 is on the table. Oracle: subtract per-signer centroid from query+gallery embeddings; if R@1 jumps by >=3 pts, nuisance removal is worth chasing. Evaluating teacher vs student and mid-layer costs nothing | None (read-only). Per-signer centroid uses test-gallery signer labels, so it is an upper bound, not a method | S, CPU/GPU inference | (a) signer-ID retention vs random-init + B0; (b) per-signer-mean-subtracted R@1; (c) R@1 for {student, EMA teacher} x {layer 3,4,5,6}; (d) per-factor probes vs epoch for the 40..200-ep checkpoints (Littwin test, Sec. 1.1) |
| 1 | **Multi-depth concatenated targets** (+ keep centered CE per head) | Bootleg (+15 Patch over pixel; stable to 1200 ep), EEG-JEPA (+3.66 frozen), V-JEPA 2.1 (SSv2 62.5 -> 72.1). Early-layer targets contain near-raw kinematics, which our final-layer centred CE lacks | Moderate positive on movement/handshape factors; may also stabilise long training (the plateau) | Masking must support it (Bootleg Sec. 3.2); 6-block encoder leaves few layers; z-score each level separately (Bootleg Table 33) | S | Targets = teacher blocks {1,3,6}: one extra linear head per level, loss weights equal; 4k-step pilot vs PLctx-4k; report R@1, per-factor R2, effective rank |
| 2 | **Clip-residual (within-clip-centred) targets** / within-clip negatives | Gulati-Nemenman WTN (error tracks supervised under fixed noise), ContentVec Eq. 2 (within-utterance negatives), Sobal theory | Removes the clip-constant component of the target: lower signer decodability; may raise movement R2. Could raise R@1 several pts if signer shortcut is real (0 oracle size tells) | Held handshape is also clip-constant (peak error at tau ~ T): use target = t - alpha * clip_mean(t), alpha in {0.25, 0.5, 1}; eval must show lexical info survives | S | 4k-step pilot with alpha in {0, 0.5, 1}; track signer-ID, R@1, handshape probe. Pre-check offline (Sec. 1.3 variance split) before any training |
| 3 | **Signer-conditioned predictor** (ContentVec) | Predictor gets a morphology descriptor; encoder released from carrying it. ContentVec: No-Cond ABX 5.76 vs 5.13 | Lower signer decodability; retrieval gain if shortcut matters; no inference cost (predictor discarded) | Descriptor leakage of lexical info; predictor shortcut (copy descriptor -> trivial); requires bone-length descriptor robust under 2D foreshortening | M | Descriptor = median-frame bone-length ratios (arm segments, shoulder width, hand/forearm ratio) -> AdaLN in predictor. Control: descriptor shuffled across clips |
| 4 | **Canonicalised teacher + signer-style student augmentation** | ContentVec (No-DT 5.67, No-DS 5.62 vs 5.13), Data2Vec-denoising (+1.5), Van Assel alignment condition (augmentations must hit the irrelevant factor) | Targets free of morphology; encoder learns invariance to it. Likely larger effect than 3 alone | 2D retarget artefacts on hand/face keypoints; may distort handshape; teacher/student target mismatch increases difficulty | M | Teacher input: bones rescaled to median lengths (deterministic); student: random limb-ratio jitter (+-20%) with within-clip negatives; 4k pilot |
| 5 | **Velocity / inverse-dynamics auxiliary head (DISReg-lite)** | MotionJEPA (planning mean 11.8 -> 70.3; without lambda_z 49.2), SMWM; Sobal's own suggestion (image differences) | Raises movement encoding, the weakest factor (R2<0) | Auxiliary may dominate; 2D RTMW jitter makes differences noisy; MotionJEPA evidence is synthetic games | M | Predict embedding of the movement/delta-channel token of bin t+1 from embeddings of bins (t, t+1); lambda 0.5 balanced; check movement probe R2 and R@1 |
| 6 | **Predictor capacity / conditioning sweep** (IWM) | Depth moves invariance <-> equivariance; invariant regime wins linear (74.5 vs 67.5) | Retrieval likes invariance; shallower predictor could add 0-2 pts; deeper adds equivariance (R2 up, R@1 down) | Effects confounded with schedule; IWM is photometric | S | pdepth in {1,2,4,8}, 4k steps; report R@1, signer-ID, per-factor R2 |
| 7 | **Anchored (frozen-copy) teacher continuation** + more signal | Human-JEPA Fig. 3 / Table 6 (naive continuation collapses silently; anchor alone plateaus) | Explains/avoids the 160 -> 200 ep stagnation; only helps if combined with 1-5 | Anchor limits movement away from the 160-ep solution | S | Continue 160 -> 200 with teacher frozen = 160-ep EMA copy and strategy 1 or 2 added; compare to the existing naive +200 ep (+0.2 / -0.05) |
| 8 | **SIGReg done per the paper, as an add-on** | LeJEPA recipe + LeWM/LpJEPA pitfalls (Sec. 4.3) | Small; main purpose is rank/isotropy of the *backbone* and a second opinion on the plateau | Previous pilot -13.8; token non-i.i.d.; low-dim data | M | Keep centred CE+EMA; SIGReg on pooled predictor/clip embedding (>=256 clips, 64-128 projector, lambda 0.02-0.05, >=2 views); RankMe of backbone |

Notes:
- Order of execution suggested: 0 -> (1, 2 offline pre-check) -> 2/1 pilots -> 3/4 -> 5/6 -> 7 -> 8.
- Strategies 2-4 are the same hypothesis (signer shortcut); run 0 first because if the oracle headroom is small, drop 2-4 and keep 1, 5, 6.
- data2vec-2.0 multi-mask (M=8-16 masked views per teacher pass, +4.6% at batch 64 for M=16 vs 2) is a free amplifier for any of the above; test in the 4k-step pilot if not already used.

---

## 8. Source index with verification status

| id | what | [V] basis |
|---|---|---|
| 2407.03475 | Littwin, How JEPA avoids noisy features (NeurIPS'24) | HTML text: eqs, Thm 4.4-4.7, Discussion |
| 2211.10831 | Sobal, JEPAs focus on slow features | HTML text, Sec. 3, App. A.6 |
| 2606.07770 | Gulati & Nemenman, contrast encodes inductive bias | HTML text, Sec. 2-4 |
| 2609.23881 | MotionJEPA / DISReg | HTML text, Fig. 1, Table 3 |
| 2606.20104 | SMWM sensorimotor world models | abstract only |
| 2505.12477 | Van Assel, joint embedding vs reconstruction (NeurIPS'25) | abstract only |
| 2511.09783 | Koopman invariants JEPA | abstract only |
| 2204.09224 | ContentVec | HTML text: Sec. 3, Table 4, Fig. 3 prose |
| 2403.18560 | Noise-robust KWS, Data2Vec-denoising | HTML text, Table 3 |
| 2403.00504 | IWM | HTML text, Tables 1, 2, 8 |
| 2505.03176 | seq-JEPA | HTML text, Tables 1, 4, 7 |
| 2211.01244 | EquiMod | HTML text (qualitative; table cells shifted) |
| 2111.09613 | AugSelf | abstract only |
| 2603.07246 | LEPA | HTML text |
| 2506.09985 | V-JEPA 2 / 2-AC | HTML text (frozen encoder statement) |
| 2608.07409 | UniJEPA | HTML text, not audited; [S] |
| 2609.34375, 2609.20800 | LRC-JEPA, JEPA-Anything | abstracts only |
| 2603.15553 | Bootleg | HTML tables 2, 27, prose |
| 2603.14482 | V-JEPA 2.1 | HTML Table 1 (column mapping from prose), Table 2 |
| 2608.00114 | EEG-JEPA | HTML Table 6 prose |
| 2509.23238 | WavJEPA | HTML ablation prose |
| 2606.07031 | CF-JEPA | PDF text, Tables 13 and ablation prose |
| 2602.04643 | MTS-JEPA (v1) / SC-JEPA | HTML Table 3 |
| 2212.07525 | data2vec 2.0 | HTML prose (Fig. 3 claim) |
| 2511.08544 | LeJEPA | HTML Tables 1a-e, Eq. 9 |
| 2605.26379 | When does LeJEPA learn a world model | abstract only |
| 2603.19312 | LeWorldModel | HTML limitations, App. G |
| 2602.01456 | Rectified LpJEPA | HTML Tables 2, 3 |
| 2305.10005 | DinoSR | HTML Table 3, 4, prose |
| 2606.19398, 2608.19084 | speech S-JEPA, soft-target counterfactuals | HTML text, Tables 1-2 and Table 1 |
| 2104.14294 | DINO | HTML prose (local-to-global) |
| 2608.21160 | Human-JEPA | HTML Tables 1, 2, 4 prose |
| 2610.06805, 2608.05806 | H-JEPA, HiLP | abstracts only |
| 2604.28173, 2610.20904, 2610.05235, 2509.03609, 2606.21607, 2609.07078, 2605.02094, 2606.11450, 2512.21064 | Q6 related | abstracts only |

Not verified / not found: any direct ablation of A-JEPA's masking curriculum; any paper on pooled-span targets in masked JEPA; any controlled identity-retention study of EMA masked-token JEPA vs MAE; any formal theory of teacher-input asymmetry; any 2026 JEPA on sign/hand skeletons. UniJEPA numbers and WiFi-JEPA are [S].
