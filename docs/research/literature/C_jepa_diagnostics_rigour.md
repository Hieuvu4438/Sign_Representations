# C. JEPA delta, representation diagnostics, proxy validity, and ablation rigour for a skeleton-JEPA sign-SSL programme

Compiled 2026-10-14. This file extends `docs/proposal12/literature/L1_jepa_paradigms.md`; it does not repeat L1. I-JEPA, V-JEPA 1/2/2.1, SALT, C-JEPA, LeJEPA/SIGReg and its 2026 users, Littwin, DirectPred, data2vec 1/2, DINOv2/v3, S-JEPA/MAMP and efficient probing are all in L1. They come up here only where a diagnostic depends on them.

**Tags.**
- **[V]** I grepped the number or quote from the raw arXiv HTML text in `scratchpad/lit/raw/txt/<id>.txt`. Table or figure captions are named where available.
- **[L1]** Already verified in L1. For SALT R² = 0.951 and the LeJEPA 99% loss correlation I re-grepped this session.
- **[S]** Comes from a search-engine summary only.
- **[K]** Background knowledge, not re-checked.

No WebFetch was used.

---

## 0. Ten-line summary

1. **No single label-free metric is reliable across methods.**
   - RankMe is a *necessary-condition* (collapse) detector and a within-family hyperparameter selector.
   - On I-JEPA and data2vec it ranks badly. Kendall τ is 0.58 overall and 0.29 on target-mask scale for I-JEPA, and 0.22 for data2vec. LiDAR gets 0.82, 0.79 and 0.55 [V 2312.04000 Tab. 1, 5].
2. **Across heterogeneous model pools, intrinsic dimension does best.** Lower ID means better downstream. The correlations are ρ ≈ −0.74 (IdEst) and ρ ∈ [−0.95, −0.26] in a 260-model study, against RankMe at 0.49 and [0.09, 0.46] [V 2606.03338, 2608.23182]. Minimising ID would also select a collapsed model, so ID must be gated by a rank check.
3. **Masked-latent-prediction (JEPA) runs are unusually noisy across seeds.** I-JEPA ViT-S at 300 epochs gives 51.5 ± 2.9 frozen accuracy, against MAE at 43.2 ± 0.2 [V 2603.15553 Tab. 17]. A single-seed JEPA comparison cannot resolve differences of 2–3 points.
4. **Multi-level teacher targets help on three fronts.** Concatenating standardised outputs of blocks {1, 4, 8, 12} (a) adds about 10 points over I-JEPA, (b) cuts seed std to 0.6, and (c) makes EMA τ nearly irrelevant: under 1.3 points of change across τ ∈ [0.988, 0.9996], against 20 points for last-layer targets [V 2603.15553].
5. **Training loss can be the cheapest proxy, but only in certain regimes.** It works with a frozen teacher (SALT R² = 0.951) or SIGReg (99% Spearman after λ-rescaling). It does not work for EMA-JEPA [L1].
6. **Small-scale rankings transfer imperfectly but usefully.**
   - About 80% pairwise decision accuracy from 150M to 1B [V 2504.11393].
   - Rank correlations of 0.71–0.90 between DataComp scales [V 2304.14108].
   - None of 8 scaling-law extrapolations beat single-scale ranking [V 2504.11393].
   - Benchmarks with a higher signal-to-noise ratio give more transferable decisions: R = 0.791 [V 2508.13144].
7. **The readout is a confound.**
   - Generative and masked objectives carry high "usability error" under linear probes (p = 3e-4 over 169 models) [V 2302.03068].
   - Attentive probes change rankings [L1].
   - Choosing the layer to probe can move accuracy by up to 30% [V 2206.13378] or 16% [V 2502.02013].
8. **Nuisance leakage must be measured with non-linear probes too.** Per-utterance standardisation cut linear speaker decodability from 93.4% to 14.8%, but a non-linear probe still reached 89.0% [V 2108.00917 Tab. 2]. Adversarial removal shows the same gap between training-time adversary and post-hoc attacker [V 1808.06640].
9. **ABX and minimal-pair tests carry over directly to sign.** A 2026 study probes ASL recognition models with minimal-pair cosine contrasts and finds pose models handshape-sensitive [V 2606.28667]. This is the cheapest test of phonological content.
10. **Fair comparisons shrink claimed gains.** Metric learning gains became "marginal… sometimes on par" once comparisons were fair [V 2003.08505]. GANs converge with equal tuning budget [V 1711.10337]. The biggest variance source in benchmarks is data sampling, not initialisation [V 2103.03098].

---

## 1. JEPA-family delta (2024–2026) beyond L1

Each card gives: what it is, ablation evidence, and the implication for a skeleton/sign JEPA.

### 1.1 Joint-embedding vs reconstruction: when latent prediction wins
**Van Assel, Ibrahim, Biancalani, Regev, Balestriero.** arXiv 2505.12477, NeurIPS 2025 [S venue].
- **Theory.** Closed-form linear SSL solutions are derived for both families. Both need augmentations aligned with the irrelevant features. "In scenarios with low-magnitude irrelevant noise features, reconstruction methods are preferable … with high-magnitude irrelevant noise features … joint-embedding methods should be preferred as they impose a strictly weaker alignment condition" [V].
- **Evidence.** ImageNet-C, linear probe, drop from severity 1 to 5 [V Tab. 1]:

  | Corruption | MAE drop | DINO drop | BYOL drop |
  |---|---|---|---|
  | Pixelate | 27.9% | 12.4% | 12.0% |
  | Gaussian noise | 27.3% | 12.7% | 16.1% |
  | Zoom blur | 20.0% | 6.5% | 9.0% |

  Averages in the text: MAE 25.1%, DINO 10.5%, BYOL 12.4%.
- **Implication.** In keypoint space, the large-magnitude "irrelevant" variance is signer body shape, framing and detector jitter on hands, not texture. JEPA should win over coordinate reconstruction only to the extent that this nuisance dominates. That is a testable prediction: the JEPA–MAE gap should *grow* as you skip skeleton normalisation or inject keypoint noise. It also fits L1's observation that S-JEPA ≈ MAMP on clean NTU.

### 1.2 Multi-level hidden-layer targets ("Bootleg")
arXiv 2603.15553, 2026.
- **Method.** An EMA teacher sees the full image. The predictor regresses the *concatenation* of z-scored outputs of teacher blocks {1, 4, 8, 12}.
- **Main gain.** "+10% vs. I-JEPA" frozen-probe on IN-1k [V].
- **Target ablation** (ViT-S, 300 ep, IN-1k Patch-probe acc) [V Tab. 2]:

  | Targets | Patch acc |
  |---|---|
  | Pixel (MAE-style) | 52.9 |
  | Block 12 only (late) | 60.2 |
  | Blocks 3–12, concatenated | 64.6 |
  | **Blocks {1, 4, 8, 12}, concatenated** | **67.8** |
  | Blocks 3–12, data2vec-style averaged | **44.1** |
  | {1, 4, 8, 12}, averaged | 57.2 |

  Averaging layers, as data2vec does, is clearly worse than concatenating them.
- **EMA sensitivity** [V Tab. 28].
  - Multi-level targets change by less than 1.3 points across τ ∈ [0.988, 0.9996].
  - Block-12-only targets move 42.1 → 62.2 in Patch accuracy and 1.8 → 17.2 in ADE kNN over the same range.
- **Seed variance** (3 seeds, ViT-S, 300 ep, IN-1k Patch / X-Blk) [V Tab. 17]:

  | Method | Patch | X-Blk |
  |---|---|---|
  | MAE | 43.2 ± 0.2 | 64.5 ± 0.0 |
  | I-JEPA | 51.5 ± 2.9 | 62.1 ± 2.3 |
  | Bootleg | 67.2 ± 0.6 | 74.0 ± 0.3 |

- **Hygiene.** The final numbers come from a fresh seed not used in the hyperparameter search [V].
- **Implication.**
  - A last-layer EMA latent target is the high-variance, τ-sensitive choice.
  - A spaced multi-level target is both a candidate *module* and a *variance-reduction device*.
  - It is a stronger default than data2vec's top-K average, which L1 R3 recommended. This revises L1.

### 1.3 Predictor as inductive bias: JEPA learns invariants
**Ruiz Morales et al.** "Koopman invariants…", arXiv 2511.09783.
- **Claim.** An idealised JEPA loss is minimised when the encoder represents regime indicator functions, i.e. Koopman eigenfunctions. "Constraining the JEPA's linear predictor to be a near-identity operator is the key inductive bias" [V abstract]. Evidence is synthetic only.
- **Implication.** This agrees with Sobal's "slow features" [L1]. A sequence JEPA with a weak or near-identity temporal predictor is pushed toward *time-invariant* clip properties, and signer identity is the most invariant property of a sign clip. A signer-leakage metric is therefore essential (§2.9).

### 1.4 Other domain JEPAs (2025)
- **TS-JEPA** (2509.25449): "outperforms both contrastive and autoregressive approaches in the majority of classification tasks, while exhibiting comparable performance to the Masked Auto-Encoder" (10 runs) [V]. As on skeletons, JEPA roughly equals MAE on low-dimensional signals.
- **Audio-JEPA** (2507.02915): "comparable performance to wav2vec 2.0 and data2vec while using less than one-fifth of their training data and with no hyper-parameter tuning" [V abstract].

### 1.5 Anti-collapse mechanics: EMA, centering, sharpening, isotropic losses
- **BYOL τ ablation** (2006.07733, Tab. 5a, 300 ep top-1) [V]:

  | τ | Top-1 |
  |---|---|
  | 1 (fixed random target) | 18.8 ± 0.7 |
  | 0.999 | 69.8 |
  | **0.99** | **72.5** |
  | 0.9 | 68.4 |
  | 0 (stop-grad of online) | 0.3 |

  The broad optimum is 0.99–0.999.
- **EMA scaling rule** (Busbridge et al., 2307.13813). When batch size changes B → κB, use momentum ρ̂ = ρ^κ [V]. Small-GPU pilots usually run batches 4–16× smaller than published recipes, so the per-step τ must move *closer to 1*. For example, 0.996 at B = 2048 becomes 0.996^(1/8) ≈ 0.9995 per step at B = 256. Otherwise the teacher averages over far fewer samples than the recipe intended and tracks the student too closely.
- **DINO centering and sharpening** (2104.14294) [V].
  - Centering prevents collapse to a dominant dimension but encourages a uniform output; sharpening does the opposite. The authors monitor both by splitting the cross-entropy into teacher entropy h(P_t) plus KL(P_t‖P_s) (Fig. 7).
  - With a momentum teacher, centering alone avoids collapse. Without momentum, centering "does not work".
  - **Usable diagnostic.** For any latent-target JEPA, log teacher-target entropy (or per-dimension std) together with the student–teacher gap.
- **IsoLoss** (Halvagal, Laborieux, Zenke, 2212.04858). In non-contrastive SSL the eigenvalues act as learning-rate multipliers per eigenmode. IsoLoss equalises them, "speeds up the initial learning dynamics… allowing us to dispense with the EMA" [V].
- **Stepwise learning** (Simon et al., 2303.15438). SSL learns embedding eigen-directions "one at a time" (or "a few at a time" for VICReg/Barlow). The same steps appear in hidden layers [V].
  - SIGReg follow-up (Lee & Lee, ICML 2026): an "Explosion" phase then a "Shrinking" phase, with stepwise delays caused by early saturation of top features [S].
  - **Implication for early stopping and proxies.** Rank-type curves plateau and jump, so short-run rank values can mis-rank configurations whose next "step" has not happened yet (§3.1).

### 1.6 Which layer and which head to read out
- **Guillotine regularisation** (2206.13378). The linear-probe gap between trunk and projector "can be as high as 30%". The best layer depends on optimisation, data and downstream task [V].
- **Layer by Layer** (Skean et al., 2502.02013). Intermediate layers "often surpass the final layer by up to 16%". The paper's metrics (prompt and dataset entropy, curvature, LiDAR, DiME) track layer quality [V].
- **Speech, layer-wise CCA** (Pasad et al., 2211.03929). Models trained to recover local features (wav2vec 2.0) show an "autoencoder-style pattern": final layers are close to the input and middle layers are best. Models that recover an intermediate layer (HuBERT, WavLM) show this less. "CCA trends provide reliable guidance to choose layers" [V].
- **Implication.** For skeleton MAE vs JEPA, compare *each at its validation-selected layer*. Final-layer-only comparisons favour the objective whose last layer happens to be most abstract.

### 1.7 A pre-registered spectral collapse audit (practical template)
**JEPA Policy** (2609.09630, Sep 2026) [V].
- Fixed 4,096-sample manifests, float64.
- Metrics: effective rank, top-eigenvalue share, and TE32 = 1 − Σ_{i≤32} λ_i / Σ_i λ_i on mean-centred features.
- A pre-specified low-rank boundary of TE32 = 0.05: 32 directions hold 95% of the energy.
- A 256-candidate retrieval test checks that predictor outputs depend on the input.
- 3 seeds × 7 tasks.

This is the right *shape* for a gate: a fixed manifest, a pre-specified threshold, and an input-dependence test.

### 1.8 Measured JEPA/MAE vs contrastive trade-offs for linear probing
- **Dubois et al.** (2302.03068, 169 models): "objectives that are generative (e.g. MAE or BEiT)… are less usable (GLA p-value = 3e-4)" [V]. Usability error is the part of the error that disappears with a better probe.
- **Park et al.** (2305.00729): contrastive learning beats masked image modelling under linear probing. CL captures global, low-frequency, shape information; MIM captures high-frequency, texture information and matters more in early layers [V].
- **Marks et al.** (2407.12210, 26 models): "differences in performance may be attributed to the differences in the backbone used by different SSL methods" rather than discriminative vs generative [V].
- **No skeleton or sign JEPA published since L1** (2026 searches) [S].

---

## 2. Diagnostics catalogue

**Notation.** Z ∈ R^{N×d} holds frozen clip embeddings (one row per held-out clip). σ_k are the singular values of centred Z, and λ_k = σ_k² are the covariance eigenvalues.

### 2.0 Master table

| Metric | Formula (short) | Predicts | Best published evidence | Failure modes | Cost (N ≈ 5–25k, d ≤ 1024) | Use as |
|---|---|---|---|---|---|---|
| **RankMe** | exp(−Σ p_k log p_k), with p_k = σ_k/‖σ‖₁ + ε (singular values, not eigenvalues) | Linear-probe accuracy within one JE family across hyperparameters | Oracle gap < 0.5 pt on average for VICReg/SimCLR/DINO [V 2210.02885]. Oracle vs RankMe: 68.2/67.8, 68.5/67.1, 72.3/72.2 [V 2304.12210 Tab. 3] | High rank is necessary, not sufficient [V]. Poor on I-JEPA (τ = 0.58) and data2vec (τ = 0.22) [V 2312.04000]. I-JEPA rank inflates early then falls [V]. Cross-model ρ only 0.09–0.46 [V 2608.23182] | One SVD; seconds | **Gate** (collapse); ranking only within a family |
| **Eff. rank on λ** / stable rank / NESum | exp H(λ/‖λ‖₁); Σλ_i/λ_1 | Degree of whitening | NESum equals stable rank for centred features [V 2608.23182] | The entropy of λ is systematically *lower* than RankMe on σ, so the two are not comparable (see §4.5) | Seconds | Descriptive |
| **α-ReQ** | Fit λ_i ∝ i^{−α}; "good" is α ≈ 1 | Linear generalisation | ρ_{>1} = −0.922 on STL-10 across layers [V 2202.05808] | Depends on the fitting range. Negative correlation in the α < 1 regime contradicts the hypothesis [V 2608.23182]. Power law fails on collapsed features [V 2210.02885] | Seconds | Do not use for decisions |
| **LiDAR** | Smooth rank of Σ_w^{−½} Σ_b Σ_w^{−½}, with clips as surrogate classes and SSL views/masks as samples; n clips × q views | Linear probe, including masked-prediction SSL | I-JEPA τ 0.82 vs RankMe 0.58; target-mask-scale τ 0.79 vs 0.29; data2vec τ 0.55 vs 0.22 [V Tab. 1, 5]. VICReg Spearman 0.92 vs 0.32 [V Tab. 9]. DINO 0.88 vs 0.82 [V Tab. 16]. Accuracy recovered (I-JEPA grid): 67.9 vs RankMe 56.8, oracle 70.6 [V Tab. 3] | Depends on the positive-pair sampler, so cross-method comparison is "a challenge" [V]. Needs matrix inversion. Assumes the view distribution is meaningful | n = 1k, q = 50 sufficed for I-JEPA/data2vec/DINO [V]; for a skeleton, ~1k clips × 10–20 masks, minutes | **Ranking** within a method family (masks, λ, τ) |
| **Intrinsic dimension** (TwoNN; MST/IdEst) | MLE from r₂/r₁ nearest-neighbour ratios; *lower is better* | Linear and kNN accuracy across architectures and objectives | IdEst ρ = −0.74, τ = −0.55 vs RankMe 0.49/0.39 [V 2606.03338]. 260 models: ρ ∈ [−0.95, −0.26], stable under OLS or kNN probe [V 2608.23182] | Collapse also gives low ID. Reliability is "moderated by architecture class and training objective" [V]. Estimator noise: median 4.4% within a batch, 7.6% across batches [V] | Seconds to minutes; IdEst 1.8 min vs 65.8 min linear probe for ViT-S [V Tab. 2] | Ranking, **only after** a rank gate |
| **CLID / cluster learnability** | k-means with K = √N, then 1-NN accuracy predicting the cluster labels, combined with ID | kNN/linear accuracy across SSL methods | 28 checkpoints: Kendall τ CLID 0.75, CL 0.44, ID 0.12, −L_align 0.26, −L_unif 0.03, rotation pretext 0.27 [V 2206.01251 Tab. 1]. OOD average τ 0.70 [V Tab. 2] | One pool of ImageNet checkpoints. Clustering metrics worked for SimCLR/MoCo but not SimSiam [V 2409.06612] | k-means + kNN; minutes | Secondary ranking |
| **Alignment / uniformity** | L_align = E‖f(x)−f(x⁺)‖^α (α = 2); L_unif = log E e^{−t‖f(x)−f(y)‖²} (t = 2) | Contrastive-family quality | Correlated within contrastive encoders [V 2005.10242]. Across SSL families, τ is only 0.26 and 0.03 [V 2206.01251] | Not meaningful for MAE/JEPA latents. Uniformity rewards spreading nuisance | Seconds | Descriptive |
| **Self-Cluster / DSE** | Deviation of ‖X̃X̃ᵀ‖_F from its expected value on the sphere | SSL models (not supervised) | Significant on all 6 datasets for SSL models; magnitude roughly triples under a kNN probe [V 2608.23182] | Uninformative for supervised models; architecture-dependent | Seconds | Descriptive |
| **kNN monitor** | Weighted k-NN (k = 20, cosine) on frozen features | Linear-probe ranking | kNN is "almost on par" with linear for DINO-ViT (74.5 vs 77.0) [V 2104.14294]. Linear and kNN "highly correlated when embedding normalization is applied" [V 2407.12210]. Cost 0.6–1.3 vs 36–70 GPU-h on IN-1k [V Tab. S.8] | Needs labels. Under-rates features that are not locally clustered (MAE-like) | One forward pass + k-NN | **Ranking gate** (cheap, label-using) |
| **Online linear probe** | Linear head trained on stop-grad features during SSL | Offline linear probe | "Online and offline linear probing are well correlated" [V 2303.01986]. "Extremely cheap… good indication" [V 2304.12210] | Moving features. Uses training-signer labels | ~0 | Trend monitor |
| **Linear vs attentive probe** | Mean-pool + linear vs cross-attention queries over tokens | Final quality, depending on readout | Attentive minus linear: V-JEPA +16 to +24, MAE-B +7.9, DINO +0.5 [L1] | Method × probe interaction flips rankings [L1] | Minutes to an hour | **Final arbiter** (report both) |
| **Training loss as proxy** | Loss value | Downstream accuracy | SALT (frozen teacher) R² = 0.951 [V]. LeJEPA loss Spearman ≈ 85% [L1], rising to 99% after λ^α rescaling with α = 0.4 [V] | EMA-JEPA loss is uninformative and non-monotone [L1] | 0 | Ranking *only* for frozen-teacher or SIGReg variants |
| **Spectral audit** (top-eigenvalue share, TE_k) | 1 − Σ_{i≤k}λ_i/Σλ_i | Collapse | Pre-registered boundary TE32 = 0.05 [V 2609.09630] | Threshold arbitrary; depends on d | Seconds | **Gate** |
| **Probe selectivity** (control tasks) | acc(task) − acc(control), where control labels are random per type (e.g. per gloss or signer) | Whether the probe reads the representation or memorises | MLP probe POS 97.3 vs control 92.8 gives selectivity 4.5. Linear 97.2 vs 71.2 gives 26.0. Dropout "ineffective" [V 1909.03368 Tab. 1] | The control-task design must mirror the real task's type structure | 2× probe cost | Validity check on phonology probes |
| **MDL (online codelength)** | Σ_blocks −log₂ p_{θ_t}(y_{t+1:t'} \| x), blocks at 0.1…100% of the data | Information plus extraction effort | Accuracy fails to separate pretrained from random features; MDL "more informative and stable" [V 2003.12298] | Implementation effort; sensitive to the block schedule | ~11 probe fits | Tie-breaker between close runs |
| **CKA / SVCCA / PWCCA / Procrustes** | Similarity of representations | Whether two runs or layers learned the same thing | CKA identifies corresponding layers across seeds where CCA fails [V 1905.00414]. PWCCA fails specificity and CKA fails sensitivity; orthogonal Procrustes "consistently performs well" [V 2108.01661]. CKA is sensitive to outliers and can be manipulated [V 2210.16156] | Not a quality metric | Seconds | Analysis only (layer choice, redundancy of modules) |
| **Nuisance (signer) leakage** | Signer-ID probe on held-out-gloss clips (linear and MLP); centroid gap ‖μ_s − μ‖ relative to within-signer spread | Spurious slow-feature content | Linear speaker accuracy 93.4 → 14.8 after per-utterance standardisation, but MLP 99.5 → 89.0 [V 2108.00917 Tab. 2]. Post-hoc attacker beats the training adversary [V 1808.06640]. Linear guardedness holds iff the class-conditional means are equal [V 2306.03819] | Signer identity is confounded with gloss when splits are not crossed. Linear probes alone understate leakage | Minutes | **Negative gate** plus a reported trade-off |
| **ABX / minimal-pair discrimination** | err = 1[d(A,X) ≥ d(B,X)] (ties count ½); A and X share a sign, B differs by one parameter; across-signer variant | Phonological content, invariant to signer | Speech: within- vs across-speaker ABX as speaker-invariance measure [V 2108.00917]. Sign: minimal-pair cosine t-tests; pose models handshape-sensitive; r ≈ 0.49 with human similarity (signer–non-signer agreement r = 0.89) [V 2606.28667] | Needs curated minimal pairs. Pooling choice matters | Seconds once pairs exist | **Ranking** and phonology gate |
| **SSL risk decomposition** | Approximation, usability, probe-generalisation and encoder-generalisation error | Why a model wins (full-shot vs few-shot) | Probe generalisation "is now the largest source of error"; a 2-parameter law gives test R² 0.94–0.96 [V 2302.03068] | Needs several probe fits and data sizes | ~4 probe fits | Explanatory for final reports |

### 2.1 RankMe in detail
- **Definition.** Exactly as in the table, with ε = 1e-7 and 25,600 samples [V]. 10,000 samples reach more than 95% of the final rank [V App. G]. The source-to-target rank transfer has Pearson > 0.99, but the StanfordCars distribution transfers less well [V].
- **Selection rule** (paper, Fig. 4). Choose the highest-rank configuration. On plateaus, choose the first value that reaches the maximum [V]. Rank is a necessary condition: "best performance is not achieved with the highest rank" for VICReg on StanfordCars [V].
- **Where to compute it.** The authors applied it to *projector-output embeddings* to predict *backbone representation* accuracy [V]. For I-JEPA and data2vec the predictor is not an expander, so LiDAR's authors computed it on the encoder instead [V 2312.04000 §4.0.2].
- **Using the SSL augmentations helps.** Computing RankMe on augmented inputs gave I-JEPA τ 0.75 vs 0.58 for vanilla [V Tab. 1]. For skeletons, compute RankMe on clips passed through the training-time augmentation and mask pipeline as well as on clean clips.

### 2.2 LiDAR in detail
- **Mechanics** [V].
  - Each clean clip x is a surrogate class. Its q views are its training-time transforms; for a JEPA, these are masked contexts encoded by the student, or teacher outputs.
  - μ_x is the class mean. Σ_b is the covariance of the class means; Σ_w is the within-class covariance plus δI.
  - LiDAR is the smooth rank of the eigenvalues of Σ_w^{−½} Σ_b Σ_w^{−½}.
  - It "ignores directions of variability… useless in solving the SSL task".
- **Why it matters for JEPA.** In I-JEPA "the rank of a representation inflates early on in training, only to rapidly diminish, with peak downstream performance occurring far from the point of maximal rank". LiDAR instead rises steadily [V §3].
- **Student vs teacher branch.** Both were evaluated. For I-JEPA the student was better (τ 0.82 vs 0.68). For DINO they were similar [V].
- **Limitations.** Matrix inversion cost. Dependence on the positive-sampling strategy "makes cross-method comparisons a challenge" [V §6].
- **Sign adaptation.** Use the *same* masking distribution for all arms when comparing mask strategies; otherwise LiDAR partly measures the mask. Alternatively use a neutral view family, such as small temporal crops plus keypoint jitter, applied identically to all models.

### 2.3 Intrinsic dimension vs rank: opposite directions
- **Opposite trends.** IdEst found that during DINO training "IdEst decreases while RankMe increases" [V]. Selection is f* = argmin_i ID_i [V Eq. 4].
- **Direction is consistent.** The 260-model study also reports *negative* ID–accuracy correlation [V].
- **Why both are needed.** Rank (spread) and ID (manifold compactness) measure different things. A collapsed model has low ID and low rank, so the decision rule is: (i) pass a rank or TE gate, then (ii) rank the survivors by LiDAR and ID.
- **Caveat.** Both ID papers evaluate *cross-model* pools of ImageNet-scale backbones. Within-family sensitivity to skeleton-JEPA hyperparameters is untested. Calibrate it on our own pilots (§4.4).

### 2.4 Probing methodology (validity of what we measure)
- **Selectivity.** Run a control task alongside every phonology probe, e.g. handshape. The control assigns a random label to each *gloss type*, keeping the label marginals of the real task. Report selectivity = real − control.
  - An expressive MLP reached 92.8% on a control task where a linear probe reached 71.2% [V 1909.03368]. High-capacity probes can manufacture "phonology".
- **MDL.** Report online codelength when two runs are within noise on accuracy. Accuracy does not separate pretrained from random encoders; MDL does [V 2003.12298].
- **Usability vs probe generalisation** [V 2302.03068].
  - Larger d improves usability but worsens probe generalisation.
  - Few-shot rankings follow probe generalisation.
  - Sign ISLR has few examples per gloss, so expect few-shot behaviour, and report a low-shot probe (e.g. 1–3 clips per gloss) in addition to the full one.

### 2.5 Representation similarity (CKA / SVCCA / Procrustes)
- **Use.** (a) Layer selection, i.e. which layer of the JEPA matches MAMP's best layer. (b) Module redundancy: does adding module X change the representation at all? A CKA ≈ 1 with the baseline plus no probe change suggests a no-op.
- **Do not use as a quality metric.**
  - Linear CKA "fails sensitivity" to principal-component deletion that does change accuracy [V 2108.01661].
  - CKA can be manipulated without functional change and is sensitive to outliers [V 2210.16156].
- **Prefer orthogonal Procrustes distance**, the robust baseline [V 2108.01661]. Always include a seed-vs-seed reference: two seeds of the *same* configuration give the floor.

### 2.6 Nuisance leakage and invariance (signer, recording setup)
- **Speech precedents.**
  - SUPERB [V 2105.01051 Tab. 2]:

    | Model | Phone error rate | Speaker-ID accuracy |
    |---|---|---|
    | HuBERT-Large | 3.53 | 90.33% |
    | wav2vec 2.0-Large | 4.75 | 86.14% |
    | HuBERT-Base | 5.41 | 81.42% |

    The phonetically best SSL models keep speaker identity *highly decodable*, so content quality does not imply invariance.
  - The per-utterance mean of CPC features alone gives speaker verification EER 6.7% and accuracy 95.8%, against 19.8% / 59.8% for MFCC means [V 2108.00917 Tab. 1].
  - Standardising per utterance makes the linear speaker probe drop to 14.8% while phone accuracy rises 75.7 → 77.0. The MLP speaker probe still gets 89.0% [V Tab. 2].
- **Sign translation.**
  - The clip-mean of skeleton-JEPA features is the analogue of the utterance mean. It carries body shape and framing.
  - Measure signer decodability on (a) raw clip features, (b) per-clip standardised features, and (c) features after LEACE-style linear erasure. LEACE guarantees that *no* linear classifier can recover signer identity [V 2306.03819].
  - Leakage that survives erasure is non-linear and needs an MLP probe.
- **Adversarial training is not a measurement.** The adversary reaches chance during training while a post-hoc attacker "still manages to reach substantially higher" accuracy [V 1808.06640]. Always use a *fresh* post-hoc probe.
- **Split hygiene.** Probe signer identity on glosses held out from the signer probe's training set, and probe gloss on signers held out from the gloss probe's training set. Otherwise signer and gloss are confounded.

### 2.7 ABX and minimal pairs: the sign analogue
- **Speech ABX** [V 2108.00917].
  - A and X are tokens of the same triphone, B differs in the middle phone. Error is 1[d(A,X) ≥ d(B,X)], aggregated over triphone pairs. d is the DTW-aligned average frame cosine.
  - Within-speaker: all three tokens from one speaker. Across-speaker: A and B from one speaker, X from another. This measures invariance.
- **Sign version** (recommended, building on Yin et al. 2606.28667).
  - Minimal pairs differ in exactly one of handshape, location or movement. Candidates come from ASL-LEX-coded vocabularies (ASL Citizen / Sem-Lex) [V].
  - Yin et al. compare the distribution of cosine similarities between two signers' productions of the *same* sign (D_same) with the distribution for the *minimal-pair* signs (D_diff). They run t-tests per pair and report "% wins vs randomly-initialised baseline" [V Tab. 1].
  - They retrain models on splits that *exclude all minimal-pair signs* to rule out lexical memorisation [V]. We must do the same: lexically held-out minimal pairs.
  - Findings: pose-based models are handshape-sensitive and pixel models location-sensitive [V].
- **ABX formulation for our models.** Given triplets (A, B, X) with A and X the same gloss from different signers, and B a minimal-pair gloss produced by A's signer:

  ABX_err(param) = mean over triplets of { 1[d(A,X) > d(B,X)] + ½·1[d(A,X) = d(B,X)] }

  - Report this per parameter (handshape, location, movement), on pooled clip embeddings *and* on part tokens (hand tokens for handshape).
  - Chance is 0.5. Signer invariance is ABX_across − ABX_within when repeated productions exist.
  - Existing repo code: `method/signrel_jepa/signrel/eval/diagnostics.py::minimal_pair_retrieval` already does same-signer-excluded minimal-pair retrieval. This is reuse, not new infrastructure.

### 2.8 Dimensional collapse diagnostics
- **Mechanisms** (Jing et al., 2110.09348) [V]:
  - strong augmentation along feature directions;
  - implicit regularisation in multi-layer linear maps pushing toward low rank.
  - A projector-free variant (DirectCLR) beat SimCLR with a linear projector.
- **Diagnostic.** Plot the full sorted log-spectrum of encoder features (and of the projector when SIGReg is used) at each checkpoint. Do not just log a scalar: plateaus and step-like dynamics (§1.5) are visible only in the spectrum.

---

## 3. Early signals and proxy validity

### 3.1 Within-run: do early checkpoints predict final quality?
- **LiDAR vs RankMe on VICReg as training progresses** [V 2312.04000 Tab. 7]. These are rank correlations across hyperparameter sets, computed at each checkpoint:

  | Epoch | Spearman, RankMe | Spearman, LiDAR | Kendall τ, RankMe | Kendall τ, LiDAR |
  |---|---|---|---|---|
  | 20 | 0.61 | 0.70 | 0.45 | 0.53 |
  | 40 | 0.43 | 0.87 | — | — |
  | 100 | 0.32 | 0.92 | 0.21 | 0.81 |

  **Proxy validity depends on the training phase. RankMe gets worse with training on this pool; LiDAR gets better.**
- **I-JEPA.** Rank peaks early and falls while accuracy keeps rising [V]. Early RankMe *anti-predicts* final quality there.
- **Stepwise dynamics** [V 2303.15438; S ICML-2026 SIGReg]. Short pilots can stop between "steps". Use at least 2 checkpoints late in the pilot and compare slopes, not levels.
- **Loss-based extrapolation.**
  - LC-PFN does Bayesian learning-curve extrapolation "over 10 000 times faster" than MCMC. Predictive early stopping gives "2–6× speed-ups on 45" datasets [V 2310.20447].
  - It needs a monotone, informative curve. That holds for SALT- and SIGReg-type losses [L1] and for kNN/probe curves, *not* for EMA-JEPA losses.
- **Instability signals.**
  - Large-batch joint-embedding ViT instability "appears as drops in kNN probe accuracy during training when the L∞-norm of the gradient spikes" [V 2304.12210].
  - Log gradient-norm spikes and kNN together. A kNN dip that recovers flags a run whose final number is luck-dependent.

### 3.2 Across scales: do small-pilot rankings transfer?
- **DataDecide** (2504.11393, LM pretraining, 25 corpora, 3 seeds) [V].
  - "The ranking of models at a single, small size (e.g., 150M parameters) is a strong baseline for predicting best models at our larger target scale (1B) (~80% of comparisons correct)".
  - "No scaling law methods among 8 baselines exceed the compute-decision frontier of single-scale predictions".
  - Continuous likelihood proxies make benchmarks ">80% predictable … with just 0.01% of the compute".
  - Final metrics average the last 10% of checkpoints.
- **Signal and noise** (2508.13144) [V].
  - Noise = relative std of the final n checkpoints: √(Σ(m_i − m̄)²/(n−1)) / m̄.
  - Signal = relative dispersion max_{j,k}|m_j − m_k| / m̄ across candidate models.
  - SNR vs decision accuracy: R = 0.791 (R² = 0.626).
  - Within the last 30 checkpoints of 1B models, ARC-Challenge varies over a 1.7% range.
  - Averaging checkpoints and filtering noisy sub-tasks raises SNR.
  - **Directly portable.** Compute SNR for each of our evaluation metrics on pilot runs and drop low-SNR metrics from gating.
- **DataComp** (2304.14108): small vs medium baselines, "rank correlations of performance is high, between 0.71 and 0.90 for different scale pairs" [V].
- **Reversals and non-transfer (SSL-specific).**
  - Kolesnikov et al. (1901.09005): "neither is the ranking of architectures consistent across different methods, nor is the ranking of methods consistent across architectures" [V]. Linear-evaluation conclusions did carry over to MLP evaluation in that study [V].
  - Ericsson et al. (2011.13377, 13 models × 40 tasks): ImageNet top-1 is "highly correlated with transfer to many-shot recognition, but increasingly less so for few-shot, object detection and dense prediction"; "little to no correlation" for ISIC [V].
  - Web-SSL (2504.01017): on ImageNet-1k, 1B–3B DINO models "do not exhibit a clear scaling trend", unlike MC-2B [V]. Small data can hide scaling benefits entirely.
  - Goyal et al. (1905.01235): SSL methods were "not 'hard' enough to take full advantage of large scale data" [V].
  - Marks et al. (2407.12210): in-domain protocols "can be robust proxies to estimate the ranking of SSL methods, but not their absolute performance". 10% fine-tuning best predicts OOD fine-tuning rankings [V].
- **Implication for skeleton pilots.** Short pilots on a subset of the pose corpus predict *rankings* of large effects reasonably well (on the order of 70–90% pairwise agreement in other fields). They do not reliably predict small effects, few-shot or out-of-distribution transfer (e.g. lexically held-out phonology), or scaling-dependent modules (e.g. a larger predictor that only pays off with more data).
  - Rule: any module whose claimed benefit is "it scales better" cannot be gated at pilot scale. Mark it "deferred", not "dropped".

### 3.3 Hyperparameter transfer
- **μP** (2203.03466). Hyperparameters from 13M parameters transfer to beat BERT-large (350M). From 40M they transfer to beat GPT-3 6.7B, "with tuning cost only 7% of total pretraining cost" [V].
- **Small-scale proxies for instabilities** (2309.14322). Large-model instabilities (attention-logit growth, output divergence) reproduce in small models at high LR. qk-layernorm and z-loss "reduce LR sensitivity and enable successful training across three orders of magnitude" [V]. LR sensitivity is the expected deviation from optimal loss over a 3-decade LR sweep, and is a cheap robustness metric for pilots.
- **EMA scaling** (2307.13813): ρ̂ = ρ^κ [V]. Re-derive τ whenever the batch size changes.
- **Batch-size myths** (Bordes et al., 2303.01986). A reported 7% small-vs-large batch gap for SimCLR is "largely countered by adapting the learning rate"; projector architecture "plays a crucial role", so compare methods "with the same projector" [V].

---

## 4. Recommended measurement stack for a single-GPU lab running many short pilots

**Design goals.** Keep evaluation overhead under 10% of pilot wall-clock. Use only frozen features and closed-form or cheap probes during pilots, and fit the attentive probe only at gates. Pre-register gate thresholds.

### 4.1 What to log

**Tier 0: every 200–500 steps, essentially free.**
1. Loss components, separately: latent prediction, auxiliary motion/coordinate head, SIGReg/VICReg. Also gradient-norm (max, mean) and EMA τ.
2. **Target health:**
   - per-dimension std of teacher targets *before* LayerNorm, averaged and minimum;
   - teacher-target entropy (DINO-style, if heads are discrete);
   - cosine(pred, target);
   - the fraction of dimensions with std below 1e-3.
3. **Spectral monitor** on a rolling buffer of about 4k clip embeddings (encoder output, mean-pooled; also projector output if SIGReg is used): RankMe on σ, top-eigenvalue share, and TE_k with k = d/16.
4. **Online linear probe** (stop-grad) on gloss labels from *training* signers only: a trend signal.

**Tier 1: every checkpoint (5–10 per pilot), a few minutes.** Fixed manifests, signer-disjoint, lexically held-out where noted.
1. **kNN retrieval.**
   - Cosine kNN with k = 20 (DINO found 20-NN "consistently working the best" [V 2104.14294]) and weighted voting with temperature 0.07 [K, the Wu et al. 2018 convention]: gloss top-1.
   - **MAP@R** with same-signer gallery items excluded. Musgrave et al. argue MAP@R is more informative than P@1 [V 2003.08505].
2. **Closed-form linear probe** (ridge/OLS on one-hot glosses, λ chosen on validation).
   - The 260-model study used OLS "to circumvent the long training time" [V 2608.23182].
   - Fit on 3 candidate layers (last, 2/3 depth, 1/2 depth) and keep the best by validation (§1.6).
3. **LiDAR.** 1k clips × 16 masked views, using the *common* view family across arms (§2.2).
4. **TwoNN ID** over 20 random 90% subsamples, reported as mean (this averaging follows the 260-model study [V]).
5. **Signer leakage.**
   - Ridge/logistic signer-ID accuracy on held-out glosses, raw and per-clip standardised.
   - A 1-hidden-layer MLP signer probe at the gate checkpoints only.
6. **Minimal-pair ABX** per phonological parameter, on lexically held-out pairs.

**Tier 2: at a gate, 1–2 checkpoints per arm.**
1. Attentive probe (1–4 queries) plus linear-on-mean-pool, ISLR top-1 and MAP@R on the signer-disjoint test set.
2. ASL-LEX phonology probes, with a control task and selectivity reported.
3. Low-shot probe (1–3 clips per gloss) to capture probe generalisation [V 2302.03068].
4. Optional: MDL for close calls; a dense per-frame probe if sign boundaries or fingerspelling are in scope.

### 4.2 The 8 metrics to carry into every decision table
| # | Metric | Role | Rationale |
|---|---|---|---|
| M1 | Per-dimension target std (min, mean) and TE_k | **Hard gate**: kill the run | Collapse is cheap to detect and fatal. TE-style boundary as in [V 2609.09630] |
| M2 | RankMe (σ-based) on encoder clips | **Hard gate**: necessary condition | Necessary, not sufficient [V 2210.02885]. Unreliable for ranking JEPA [V 2312.04000] |
| M3 | kNN MAP@R, gloss, signer-excluded | **Primary ranking proxy** in pilots | Cheapest label-using proxy; tracks linear probes [V 2407.12210] |
| M4 | LiDAR (common views) | Secondary ranking, label-free | Best published label-free ranking metric for masked-prediction SSL [V 2312.04000] |
| M5 | TwoNN ID (lower is better, after M1/M2 pass) | Tertiary ranking | Strongest cross-architecture label-free correlate [V 2606.03338, 2608.23182] |
| M6 | Signer-ID probe accuracy (linear and MLP) | **Negative metric / trade-off axis** | Slow-feature risk [V 2511.09783, L1 Sobal]. Non-linear leakage [V 2108.00917] |
| M7 | Minimal-pair ABX per parameter | Phonology ranking / gate | Direct test of lexical-contrast content [V 2606.28667] |
| M8 | Attentive and linear ISLR top-1 (gate only) | **Final arbiter** | Readout confound; report both [L1] |

### 4.3 Decision statistics (formulas)

Let a_i and b_i be per-item scores (e.g. per-query AP or correct/incorrect) for arms A and B on the *same* test items. Write Δ_i = a_i − b_i and Δ̄ = mean(Δ_i).

1. **Signer-cluster paired bootstrap** (default for retrieval and accuracy).
   - Test items are not i.i.d.: queries share signers and glosses. Resample *signers* with replacement and include all their queries, R = 10,000 times.
   - Compute Δ̄* each time. The 95% CI is the 2.5–97.5 percentiles. The one-sided p-value is #{Δ̄* ≤ 0}/R.
   - The design effect is deff = 1 + (m−1)·ICC (m = queries per signer). The naive SE underestimates the true SE by a factor √deff [K].
   - The 260-model study likewise used a cluster-robust bootstrap with BH correction [V 2608.23182].
2. **Paired sign-flip permutation test** (exact alternative).
   - Statistic T = Σ_c Δ̄_c over signer clusters c.
   - Permute by flipping the sign of each cluster's Δ̄_c independently.
   - p = (1 + #{|T_perm| ≥ |T_obs|}) / (1 + R).
   - In IR evaluation, randomisation, bootstrap and t-tests showed "little practical difference", but they disagree more with fewer topics [S, Smucker et al. CIKM 2007 / SIGIR 2009]. With few test signers, prefer the permutation test.
3. **Across seeds** (the training-noise component).
   - With k seeds per arm, paired by seed (same data order and init seed where possible), report d̄ ± t_{k−1}·s_d/√k.
   - Report the **probability of improvement** P(A>B) = (1/k²) Σ_i Σ_j 1[a_i > b_j] (Mann–Whitney form). Bouthillier et al. propose the threshold γ = 0.75 [V 2103.03098].
   - Noether sample size: N ≥ ((Φ⁻¹(1−α) − Φ⁻¹(β)) / (√6 (½ − γ)))² [V].
     - α = 0.05, power 0.95, γ = 0.75 → N ≈ 29.
     - Power 0.8, γ = 0.75 → N ≈ 17.
     - Power 0.8, γ = 0.8 → N ≈ 12 (my computation).
   - This is infeasible per arm on one GPU. **Use seed replicates to estimate σ_seed once per method family, then decide on single runs only when the effect exceeds the minimum detectable effect (MDE).**
4. **Minimum detectable effect for k seeds per arm** (two-sided α = 0.05, power 0.8):
   - MDE ≈ (z_{.975} + z_{.80})·σ_seed·√(2/k) = 2.80·σ_seed·√(2/k).
   - That is ≈ 2.29·σ_seed for k = 3 and ≈ 1.77·σ_seed for k = 5 (my computation).
   - With I-JEPA-like σ_seed ≈ 2.9 [V 2603.15553], k = 3 gives an MDE of about 6.6 points. Hence the emphasis on variance reduction (multi-level targets, checkpoint averaging).
5. **Total noise for a single-run comparison:**
   - σ²_Δ ≈ 2σ²_seed + 2σ²_ckpt + σ²_test,paired + 2σ²_probe.
   - σ_ckpt is the relative std of the last n = 5 checkpoints × the mean [V 2508.13144].
   - σ_test,paired comes from the signer bootstrap of Δ.
   - σ_probe is the std over 5 probe refits on fixed features. For linear-probe evaluation-only reruns, the std is 0.02–0.13 points, but 2.62 for 1% fine-tuning [V 2407.12210]. Probe noise is usually small *except* in low-shot regimes.
6. **Keep / drop rules** (pre-registered per module; δ is the smallest effect size of interest, e.g. 1.0 point top-1 or 0.01 MAP@R).
   - **KEEP:** lower 95% CI bound of Δ > 0, *and* Δ̄ ≥ δ, *and* P(A>B) ≥ 0.75 across the seeds available, *and* no gate failures (M1/M2), *and* signer leakage (M6) does not rise by more than its own MDE.
   - **DROP as equivalent:** TOST, i.e. the 90% CI of Δ lies within [−δ, +δ]. Prefer the simpler model.
   - **DROP as harmful:** upper 95% CI bound < 0.
   - **INCONCLUSIVE:** anything else. Add seeds (up to k = 5) or defer. *Never* convert "not significant" into "drop" without TOST.
7. **Multiple comparisons.**
   - Within one ablation table (m modules × primary metric), use Holm: order the p-values, reject p_(i) ≤ α/(m − i + 1) sequentially [K].
   - For exploratory metric × module grids, control FDR with Benjamini–Hochberg: reject the largest i with p_(i) ≤ iα/m [K]. This is the procedure used in [V 2608.23182].
   - Designate *one* primary metric per gate (M3 in pilots, M8 at gates). Secondary metrics are descriptive.
8. **Aggregating across tasks** (ISLR, retrieval, phonology, signer).
   - Normalise each metric (e.g. by the baseline's value or the oracle range).
   - Report the interquartile mean (IQM) with stratified bootstrap CIs. Bootstrap CIs with 3 runs "underestimate the true 95% CIs" [V 2108.13264], so widen the nominal level (e.g. 99%) when k = 3.

### 4.4 Noise floors and proxy calibration (do once, reuse)
1. **Baseline seed replicates.** Run 3, ideally 5, seeds of the reference configuration (e.g. a MAMP-style motion MAE and the default JEPA) at pilot budget. This gives σ_seed per metric (M3–M8).
   - Expect JEPA-type σ to be several times MAE-type σ [V 2603.15553]. Estimate it per family, not globally.
   - Randomise data order *and* initialisation, plus subset sampling if pilots use corpus subsets. Data sampling is the dominant variance source [V 2103.03098]. Randomising more sources makes a biased estimator approach the ideal one at "51× reduction in compute cost" [V].
2. **Checkpoint noise.** Compute the relative std over the last 5 checkpoints of every run, logged for free.
3. **Test-sampling noise.** Signer-cluster bootstrap SE of each metric, and of Δ, for one pair of runs.
4. **SNR per metric.** SNR = Rel.Dispersion(all pilot arms at the final checkpoint) / Rel.Std(last n checkpoints) [V 2508.13144]. Remove metrics with SNR < 2 from gating; keep them as descriptive.
5. **Proxy-validity check.** Across the first ~10–20 pilot runs (different modules or hyperparameters), compute Kendall τ between each proxy (M2–M7 at 25%, 50% and 100% of the pilot budget) and the Tier-2 arbiter (M8).
   - Bootstrap over runs for a CI.
   - Report the **top-1 regret**: the arbiter score of the oracle-best run minus that of the proxy-selected run. LiDAR and RankMe papers report this as "accuracy recovered" [V 2312.04000].
   - Promote a proxy to a ranking gate only if its τ lower CI bound is > 0.4 and its regret is below the MDE.
   - Literature priors to beat: LiDAR τ 0.55–0.82 on masked SSL; RankMe 0.22–0.58 [V].
6. **Successive halving with noise-awareness** [K, Hyperband/ASHA].
   - Promote the top 1/η arms at each rung by M3, but promote *all* arms within MDE of the rung leader.
   - Use rungs only where the §4.4.5 check shows early-vs-final τ > 0.6. Rank metrics can invert early in JEPA runs [V 2312.04000].

### 4.5 Repo-specific notes
- **Two "effective_rank" functions in the repo are not RankMe.**
  - `method/signrel_jepa/signrel/eval/diagnostics.py::embedding_health` and `method/pose_jepa/src/signrepr/evaluation/representation.py::representation_diagnostics` compute exp-entropy of **σ² (eigenvalues)**. RankMe uses **σ**.
  - The σ²-entropy is systematically lower and more top-heavy, so the numbers are not comparable to published RankMe values or thresholds.
  - Add a σ-based RankMe alongside, or rename the field (`eig_entropy_rank`).
  - `signrel_jepa` also L2-normalises the features before computing it, which RankMe does not do.
- `embedding_health`'s own note, "Descriptive geometry, not a stand-alone collapse verdict", is consistent with this review. Formalise it as an M1/M2 *gate*, not a ranking metric.
- `minimal_pair_retrieval` (same file) is the natural base for the M7 ABX implementation.

---

## 5. Ablation design patterns

### 5.1 Why one-factor-at-a-time (OFAT) misleads in JEPA design
JEPA modules interact strongly. The evidence below shows it.
- **Target × EMA.** τ barely matters with multi-level targets (under 1.3 points) but swings last-layer targets by 20 points [V 2603.15553]. An OFAT EMA sweep on a last-layer-target baseline would "find" τ critical. On the multi-level baseline it is irrelevant.
- **Context loss × deep supervision.** V-JEPA 2.1's context loss hurts global tasks unless multi-level supervision is added [L1].
- **Stop-grad × anti-collapse regime.** Stop-grad hurts in PointTube (79.4 vs 94.1) but helps in Laya [L1].
- **Mask × target.** Mask geometry is the most sensitive JEPA knob [L1 LiDAR]. The published optima come with different targets: MAMP's best 90% ratio is for motion targets, and I-JEPA's best (0.15, 0.2) target blocks are for latent targets [L1]. So it is plausible, though not directly tested, that the mask optimum moves with the target. Test the mask × target interaction rather than assume it.
- **Probe × method.** Attentive probing adds +17 for V-JEPA and +0.5 for DINO [L1]. "Ablating" a module that mainly improves token-level features looks harmful under mean-pool linear probes.

OFAT estimates each effect conditional on one arbitrary setting of all the others. Under interactions that estimate does not generalise and can change sign.

### 5.2 Factorial and fractional-factorial designs
- **2^k full factorial.** For binary modules, the main effect of A is mean(y | A = 1) − mean(y | A = 0), averaged over all other settings. The interaction is AB = ½[(effect of A | B = 1) − (effect of A | B = 0)] [K].
- **Hidden replication.** Each main effect uses *all* 2^k runs, half on each side. With one seed per cell, a 2^4 design estimates each main effect with the precision of 8 vs 8 replicates. That is far more efficient than OFAT, which needs separate replicates per factor [K].
- **Fractional factorial 2^{k−p}** [K].
  - Five binary modules in 16 runs (2^{5−1}, resolution V) estimate all main effects and all two-factor interactions unconfounded.
  - Six modules in 16 runs (2^{6−2}, resolution IV) keep main effects clear of two-factor interactions, which are aliased with each other.
  - **Candidate factors for the skeleton JEPA:**
    1. mask family (part-time tubes vs random);
    2. target (multi-level vs last-layer);
    3. auxiliary motion head (on/off);
    4. anti-collapse (EMA vs SIGReg);
    5. predictor (narrow-deep vs wide-shallow);
    6. context loss (on/off).
- **Analysis.**
  - Fit y = β₀ + Σβ_j x_j + Σβ_jk x_j x_k + ε with x ∈ {−1, +1}.
  - Estimate ε from seed replicates of 2–4 centre or anchor cells, *not* from the residual, which is contaminated by unmodelled interactions.
  - Apply Holm across the tested coefficients.
- **Split-plot when a factor is expensive to change** [K]. Example: the frozen-teacher training stage in SALT-style variants. Train 2 teachers (whole plots) and cross each with all student factors (subplots). Analyse with teacher as a random block.

### 5.3 Leave-one-out from full vs add-one to baseline
- **Leave-one-out (LOO) from the full model** measures *necessity given the others*. Redundant modules each look useless, e.g. two partially overlapping anti-collapse terms. LOO underestimates them.
- **Add-one to a baseline** measures *sufficiency alone*. It overestimates modules that duplicate what another module would also provide.
- **When they disagree, there is an interaction.** A factorial lets you compute a **Shapley-style attribution**: the module's average marginal effect over all subsets, available from the 2^k table [K].
- **Practical rule.** For each module report both LOO-from-full and add-to-base, from the same factorial. Keep a module if it is *necessary* (LOO Δ passes the §4.3 KEEP rule) *or* if it provides a unique capability probed elsewhere (e.g. a dense or phonology metric) and costs less than 10% compute.

### 5.4 Compute and tuning-budget matching
- **Report three cost axes per arm:** (i) samples or steps seen, (ii) training FLOPs, (iii) wall-clock on the same GPU. They can disagree; for example, JEPA pays a teacher forward pass on the full input plus a predictor, while MAE encodes only visible tokens. "Incomplete reporting across the spectrum of cost indicators may lead to an incomplete picture" [V 2110.12894].
  - I-JEPA claims about 5× fewer iterations to converge [L1].
  - S-JEPA used 1200 vs 400 epochs and about 4.5× MAMP's wall-clock [L1].
- **Primary comparison at matched wall-clock (or FLOPs); secondary at matched epochs.** A module that wins only at matched epochs but loses at matched compute is a "compute-for-accuracy trade". Mark it as such.
- **Tuning-budget parity.**
  - Equal numbers of hyperparameter trials per arm, the same search space breadth, the same validation split, and test-set isolation.
  - Metric-learning SOTA gains became "marginal… sometimes on par" with classic methods under fair tuning [V 2003.08505].
  - GANs "can reach similar scores with enough hyperparameter optimization and random restarts" [V 1711.10337].
  - Report the best-of-n *and* the expected max given n trials (Dodge et al.'s expected validation performance [K 1909.03004]), so a method that needs more tuning is penalised visibly.
- **Fixed components.** Use the same projector or probe head across methods ("compare JE-SSL methods with the same projector network", [V 2303.01986]). Use the same skeleton normalisation and the same data subset and order seeds.

### 5.5 The probe / readout confound (protocol)
1. Treat the readout as a *crossed* factor. Every pretrained arm is evaluated with {linear on mean-pool, attentive, kNN}, each at its validation-selected layer.
2. Test for a method × readout interaction (two-way ANOVA on probe scores, or an interaction contrast with the §4.3 bootstrap). If it is significant, report conclusions per readout. Do not average.
3. Run phonology probes with control tasks (selectivity), and report the low-shot probe separately (probe-generalisation regime) [V 1909.03368, 2302.03068].
4. Keep the probe budget fixed (epochs, LR search grid, augmentation off) and record probe-seed variance as σ_probe.

### 5.6 Sequencing pilots on one GPU (putting it together)
1. **Phase 0: noise floor.** 3–5 seeds × 2 reference configurations, then σ_seed, σ_ckpt and σ_test per metric (§4.4). Also run the SNR check on metrics.
2. **Phase 1: screening.** A resolution IV/V fractional factorial over 5–6 binary modules, one seed per cell plus 2–4 replicated anchor cells. Gate on M1/M2 and rank on M3. Every run logs M1–M7 so proxy calibration (§4.4.5) comes for free.
3. **Phase 2: confirmation.** For the top modules and any significant interactions, 3–5 seeds × {full, LOO per module}, at matched wall-clock. Apply the Tier-2 arbiter (M8) and the §4.3 keep/drop rules with Holm correction.
4. **Phase 3: held-out confirmation.** A fresh seed and fresh test signers for the final configuration vs the strongest baseline (Bootleg's "fresh seed" hygiene, [V 2603.15553]). Report the IQM across tasks with CIs.

---

## 6. Source index (all fetched this session into `raw/txt/`)

- **Diagnostics:**
  - 2210.02885 RankMe
  - 2312.04000 LiDAR
  - 2202.05808 α-ReQ (arXiv version, "Investigating power laws…"; NeurIPS 2022 [S])
  - 2608.23182 label-free metric comparative study
  - 2606.03338 IdEst
  - 2206.01251 CLID
  - 2409.06612 label-free monitoring
  - 2005.10242 alignment/uniformity
  - 2110.09348 dimensional collapse
  - 1909.03368 control tasks
  - 2003.12298 MDL probing
  - 1905.00414 CKA
  - 2108.01661 similarity statistical testing
  - 2210.16156 CKA reliability
  - 2302.03068 risk decomposition
  - 2407.12210 benchmarking SSL protocols
  - 2609.09630 JEPA Policy spectral audit
- **Speech and nuisance:**
  - 2105.01051 SUPERB
  - 2108.00917 speaker info and ABX
  - 2211.03929 layer-wise CCA
  - 1808.06640 adversarial removal
  - 2306.03819 LEACE
- **Sign:**
  - 2606.28667 Phonological perception of sign language models
- **JEPA delta:**
  - 2505.12477 JE vs reconstruction
  - 2603.15553 Bootleg (multi-level targets, seed variance)
  - 2511.09783 Koopman-JEPA
  - 2509.25449 TS-JEPA
  - 2507.02915 Audio-JEPA
  - 2307.13813 EMA scaling
  - 2006.07733 BYOL
  - 2104.14294 DINO
  - 2212.04858 IsoLoss
  - 2303.15438 stepwise SSL
  - 2206.13378 Guillotine
  - 2502.02013 Layer by Layer
  - 2305.00729 CL vs MIM
  - 2303.01986 democratising JE-SSL
  - 2304.12210 Cookbook of SSL
- **Proxy validity and scaling:**
  - 2504.11393 DataDecide
  - 2508.13144 Signal and Noise
  - 2304.14108 DataComp
  - 2011.13377 transfer of SSL
  - 1901.09005 revisiting SSL
  - 1905.01235 scaling/benchmarking SSL
  - 2504.01017 Web-SSL
  - 2203.03466 μP
  - 2309.14322 small-scale proxies
  - 2310.20447 LC-PFN
- **Rigour:**
  - 2103.03098 variance in benchmarks
  - 2108.13264 rliable
  - 2109.08203 seed 3407
  - 2003.08505 metric learning reality check
  - 1711.10337 GANs equal
  - 1909.03004 Show your work
  - 2110.12894 Efficiency misnomer
- **Search-only [S]:**
  - Smucker, Allan & Carterette, CIKM 2007 / SIGIR 2009 (IR significance tests)
  - Lee & Lee, ICML 2026 (SIGReg BHEP dynamics)
- **Not found:** a 2026 skeleton or sign-language JEPA paper; a dedicated multi-seed SSL pretraining-variance study beyond 3-seed tables (2603.15553 is the best data point).
