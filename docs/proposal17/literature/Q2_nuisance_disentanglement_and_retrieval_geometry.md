# Q2. Nuisance (signer) disentanglement and retrieval geometry for frozen skeleton-SSL dictionary retrieval

Date: 2026-10-17. Scope: GAPS only relative to `docs/research/literature/{A,C}_*.md` and `docs/proposal12/literature/L4_sign_ssl.md`. Path-JEPA/signatures excluded.
Not repeated here: SignRep ablation tables and frozen benchmark tables (A §1.5, §3.2), SUPERB speaker-ID numbers, per-utterance standardisation (van Niekerk, C §2.6), AUTSL anonymisation basics (L4 §7), RankMe/LiDAR/dimensional-collapse diagnostics (C §1.5, §2).

**Tags.** [V] number/quote grepped in raw arXiv HTML (this session, `scratchpad/q2/<id>.txt`, built with `curl` + tag stripping; table caption checked). [S] secondary, seen only in a search-engine summary or a source I could not open (Cloudflare block etc.). [I] my inference, not stated by any source. [L] local in-progress diagnostics found in `docs/proposal17/diagnostics/results/` (written by another process while I worked; I did not verify model, split, or code; use only to calibrate magnitudes, not as evidence).

**Verification protocol.** Every [V] below was located in the raw text of the cited arXiv id; table row order was re-read from the flattened HTML (cells joined with `|`). The arXiv search API rate-limited me (HTTP 429), so ids were found by web search and then confirmed by fetching `arxiv.org/html/<id>` and reading the title. No id below is invented. Two items could not be confirmed on arXiv: DN2A (CVPR 2023 only, no arXiv version found) and JOKR (CGF 2022; no arXiv id found); both are marked [S].

---

## 0. Ten-line summary

1. **Closest precedent is speech.** Every successful speaker-disentanglement recipe (ContentVec, Spin, R-Spin, Komatsu BYOL) uses the same triad: (a) an *analytic nuisance perturbation* of the input (formant/F0 scaling + EQ), (b) an *asymmetric* teacher/student or swapped-prediction objective so the target does not change under the perturbation, (c) a *discrete or clustered target* that bottlenecks continuous speaker detail. None needs speaker labels in the main loop.
2. **Effect sizes are modest and metric-specific.** ContentVec vs same-teacher HuBERT-iter: ABX across-speaker 7.20 -> 6.32 (-12% rel.) [V]. Spin on HuBERT: ABX avg 4.22 -> 3.00 (-29% rel.) at 45 GPU-minutes [V]. Removing speaker perturbation from R-Spin costs only +1.9/+1.4 abs. WER [V], less than removing noise or the auxiliary loss.
3. **Pooled aggregators learn the nuisance.** A DINO-style sentence token had speaker-ID NMI 0.61 with its pseudo-labels [V 2409.10103]; mean-pooled linear speaker-ID (VoxCeleb1) was 67.2 (HuBERT), 47.6 (SD-HuBERT), 26.6 (BYOL with speaker-perturbed student and clean teacher) [V]. Our clip-mean embedding is exactly this situation.
4. **Post-hoc linear removal works for held-out speakers, but gains are small and capped.** Collapsing a PCA speaker subspace (30-57 dims) estimated on 251 training speakers lowers across-speaker ABX by 5-15% rel. on unseen speakers; beyond ~50 dims it overfits [V 2305.12464]. With only O(40) signers we will have much noisier subspaces [I].
5. **Linear guarantees do not protect neighbourhoods.** LEACE/INLP guard against *linear* classifiers only [V]; k-means still recovers gender 92.5% after hard-debiasing [V 1903.03862]; INLP leaves MLP recovery at 85.0% [V]; adversaries hit chance while post-hoc attackers reach 56.0% vs 49.0% [V 1808.06640].
6. **kNN is the weak spot of masked/predictive objectives, and a short cross-instance tuning stage repairs it.** MAE ViT-L kNN 60.6 vs linear 76.0; 20 epochs of NN-contrastive tuning -> 77.4/80.2 [V 2304.10520]. Skeleton MAMP linear 84.9 vs kNN(K=1) 63.1 on NTU-60 xsub; STARS 87.1/79.9 [V 2407.10935]. The NN lookup itself and head initialisation are essential (kNN 40.5 and 68.1 when ablated) [V].
7. **The same mechanism fails if neighbours share the nuisance.** Our own SignRep finding (same-signer NN 48.7% > same-gloss 43.8%) means unrestricted NN mining will mostly pick same-signer, different-gloss pairs [I]. Fixes with evidence: camera-aware cross-camera proxies in unsupervised re-ID (MSMT17 R@1 34.0 -> 67.4) [V 2012.10674]; latent positive sampling across recordings in speaker verification (SimCLR EER 6.30 -> 2.57) [V 2505.14561].
8. **Sign-specific:** body proportions alone carry a large part of signer identity in pose: AUTSL signer-ID 80.18 -> 65.34 (average-body transfer) -> 52.20 (random-body transfer), hands untouched [V 2410.13675]. But naive anonymisation costs sign accuracy (80.97 -> 64.48 when trained and tested anonymised) [V]. So morphology augmentation must preserve body-relative location.
9. **Isolated-sign clips differ from speech utterances in one crucial way [I]:** per-utterance mean/variance normalisation works in speech because an utterance averages many phones; a 2-3 s isolated sign has 1-3 "phones", so clip-level centring erases content. Signer statistics must come from rest frames, from other clips (pseudo-signer clusters), or from the pretraining corpus.
10. **Ranked plan (end of file):** (1) retrieval-time geometry stack, (2) label-free pseudo-signer subspace erasure, (3) asymmetric morphology perturbation + style-conditioned predictor, (4) cross-signer NN tuning only after (1)-(3) lift cross-signer NN precision, plus four lower-priority items.

---

## 1. Speaker/identity disentanglement in self-supervised speech and audio

### 1.1 ContentVec (Qian et al., ICML 2022, arXiv 2204.09224) [V]

**Mechanism (three components).**
- *Disentanglement in teachers (DT).* All training utterances are converted to **one** target speaker with an unsupervised voice converter (Polyak et al. 2021, retrained on 200 LibriSpeech speakers); converted audio -> pretrained HuBERT layer 7 -> k-means (100 clusters) -> teacher labels. Layer 7 is chosen over the usual 6 because it has lower speaker classification accuracy at comparable teacher quality [V §4.1].
- *Disentanglement in students (DS).* Two random speaker perturbations (formant scale rho1, F0 scale rho2, both U[1,1.4] then reciprocal with p=0.5, plus random equaliser; algorithm from NANSY, Choi et al. 2021) are fed to the student; a frame-level SimCLR loss makes the two perturbed views agree, **with negatives drawn from other frames of the same utterance**. The same perturbations are applied inside the masked-prediction loss. Loss weight lambda ramps linearly to 10 over 100k steps (1e-5 x step). It is imposed at the "last-but-five" layer, not the output.
- *Speaker conditioning (Cond).* A frozen GE2E speaker embedding is fed to the 3-layer predictor via conditional LayerNorm, so the encoder does not need to carry the speaker information that remains in the teacher labels. ContentVec's own training needs no speaker labels; GE2E was pre-trained with labels elsewhere.

**Headline vs controls [V Tab. 1, 2].** HuBERT-iter is the key control (same teacher network, no voice conversion).

| Model | ABX(w) | ABX(a) | PPX | SUPERB PER | ASR WER | QbE MTWV | IC acc | SF CER |
|---|---|---|---|---|---|---|---|---|
| ContentVec | 5.13 | 6.32 | 650.0 | 0.049 | 5.7 | 0.0590 | 0.991 | 0.236 |
| HuBERT-iter | 6.01 | 7.20 | 739.1 | 0.052 | 6.5 | 0.0891 | 0.983 | 0.259 |
| HuBERT | 6.06 | 7.37 | 790.2 | 0.054 | 6.4 | 0.0736 | 0.983 | 0.256 |

ABX(w)/(a) = within/across-speaker phone ABX error (%); lower is better. Note QbE (query-by-example) is **worse** for ContentVec (0.0590 vs 0.0891), so the benefit is not uniform across retrieval-flavoured tasks. SID: "reduction is as high as 36% compared to the HuBERT-iter" (text, Fig. 3, exact definition not stated) [V].

**Ablation (Tab. 4, best layer per metric) [V].**

| Variant | ABX(w) | ABX(a) | PNMI |
|---|---|---|---|
| full ContentVec | 5.13 | 6.32 | 0.590 |
| No-DT (no voice converter) | 5.67 | 6.92 | 0.576 |
| No-DS (no student invariance) | 5.62 | 6.93 | 0.577 |
| No-Cond (no speaker-conditioned predictor) | 5.76 | 7.08 | 0.593 |
| contrastive at last layer (N1) | 5.76 | 7.03 | 0.545 |
| contrastive at last-but-three (N4) | 5.19 | 6.43 | 0.587 |
| contrastive at last-but-seven (N8) | 5.14 | 6.40 | 0.584 |
| lambda 1e-6 / 5e-6 / 2e-5 / 5e-5 | 5.18/5.10/5.18/5.22 | 6.36/6.39/6.51/6.58 | 0.586/0.586/0.584/0.582 |
| weaker voice converter (10k vs 40k steps, appendix) | 6.05 | 7.78 | 0.5616 |

**Which component matters most?** On across-speaker ABX: Cond (+0.76) > DS (+0.61) ~ DT (+0.60) ~ contrastive at the last layer (+0.71). On PNMI: DT and DS (about -0.014 each) and last-layer placement (-0.045) matter; Cond does not (+0.003). All three are needed together (authors' conclusion). Two caveats are directly relevant to us: (i) a *poor* voice converter is worse than none (ABX(a) 7.78 vs 6.92 without DT), i.e. an unrealistic nuisance transform damages content; (ii) the invariance loss should sit at an intermediate layer, with later layers free to re-use style (N4/N8 ~ standard, N1 clearly worse). Information-flow probe [V Fig. 4a, text]: speaker-ID accuracy decreases monotonically up to layer ~9 with a significant drop near layer 7 where the contrastive loss sits.

**Setting caveats.** 94M-parameter base model, LibriSpeech 960h, one converter per run, ABX evaluated at the best layer per metric (layer selection is itself a tuned degree of freedom), single seeds.

### 1.2 Spin (Chang, Liu, Glass, Interspeech 2023, arXiv 2305.11072) [V]

**Mechanism.** Self-supervised *fine-tuning* of only the top 2 layers of a pretrained HuBERT/WavLM on 100 h of LibriSpeech. Original and speaker-perturbed (ContentVec's perturbation) utterances each produce frame probabilities over a learnable L2-normalised codebook (D=256, tau=0.1); targets are Sinkhorn-Knopp-balanced assignments (eps=0.02, 3 iterations) of the *other* view (swapped prediction, SwAV style). Without the entropy smoothing the trivial solution is one codeword [V]. No masking. 45 min on one RTX A5000 [V], "<1% of ContentVec" compute [V].

**Numbers [V].** SUPERB (Tab. 1): PER HuBERT 5.41 -> Spin256 4.39; WavLM 4.84 -> 4.18; ASR WER HuBERT 6.42 -> 6.34 (small), WavLM 6.31 -> 5.88; QbE MTWV HuBERT 0.0736 -> 0.0912. ZeroSpeech ABX (Tab. 2; W/C = within/across speaker, C/O = clean/other; Spin2048):

| Model (layer) | W-C | W-O | C-C | C-O | Avg | gap C-C minus W-C | gap C-O minus W-O |
|---|---|---|---|---|---|---|---|
| HuBERT (11) | 3.07 | 3.90 | 3.71 | 6.19 | 4.22 | 0.64 | 2.29 |
| HuBERT + Spin2048 (12) | 2.44 | 3.00 | 2.81 | 3.76 | 3.00 | 0.37 | 0.76 |
| WavLM (11) | 2.73 | 3.41 | 3.21 | 4.95 | 3.58 | 0.48 | 1.54 |
| WavLM + Spin2048 (12) | 2.75 | 3.33 | 3.24 | 4.17 | 3.37 | 0.49 | 0.84 |
| ContentVec100 (12) | 2.98 | 3.70 | 3.44 | 5.17 | 3.82 | 0.46 | 1.47 |
| data2vec (4) | 4.03 | 5.09 | 4.72 | 6.97 | 5.20 | 0.69 | 1.88 |

(gap columns computed by me from the table.) Spin (K=256) lowers last-layer speaker-ID accuracy to ~10% [V text]. Note the *within-to-across gap* is the speech analogue of our "gallery-signer-disjoint" penalty and is a cleaner nuisance-reliance metric than absolute ABX: HuBERT and data2vec have gaps 0.64/0.69 (clean) and 2.29/1.88 (other), Spin shrinks the "other" gap threefold.

**Caveats.** Spin only works on top of a pretrained model ("only positional information is learned if trained from scratch" [V]); K and top-layer count were tuned on the same dev set; WavLM gain is small (3.58 -> 3.37) because WavLM already has denoising/mixing.

### 1.3 R-Spin (Chang & Glass, NAACL 2024, arXiv 2311.09117) [V]

Adds additive-noise invariance and an auxiliary pseudo-label loss on "acoustic pieces". Ablation on CHiME-4 WER (real/sim, fine-tuned WavLM) [V Tab. 2]:

| Variant | Real | Sim |
|---|---|---|
| R-Spin (full) | 26.4 | 26.6 |
| no auxiliary loss | 47.8 | 45.6 |
| no Spin loss | 31.9 | 32.4 |
| no speaker perturbation | 28.3 | 28.0 |
| no additive noise | 49.4 | 46.8 |
| plain Spin2048 | 52.1 | 46.6 |

**Reading.** Speaker perturbation is the *smallest* of the four components (+1.9/+1.4 abs.). The invariance loss itself (+5.5/+5.8) and the auxiliary anti-collapse loss (+21.4/+19.0) matter more. For a JEPA, the analogue of the auxiliary loss is the anti-collapse/variance regulariser, which we already have; the analogue of the perturbation is optional per this evidence [I].

### 1.4 Frame-level BYOL with perturbed student (Komatsu & Shinozaki, arXiv 2409.10103) [V]

- Observation: SD-HuBERT's pooled CLS aggregator has **speaker-normalised mutual information 0.61** with its predicted sentence-level categories (LibriSpeech test) [V]. A DINO-style pooled objective "learned to discriminate speaker identity rather than linguistic content".
- Fix: BYOL (student with predictor, EMA teacher, MSE between L2-normalised outputs) **at frame level, no CLS token**; student input = speaker-perturbed waveform (restricted to realistic male<->female conversion by an utterance-mean-pitch threshold, no gender labels), teacher input = clean. Targets = projection of the *last* teacher Transformer layer; using teacher layer 6 gave "very poor results" for syllabic organisation.
- Mean-pooled linear speaker-ID on VoxCeleb1 [V Tab. 2]: HuBERT 67.2, VG-HuBERT 37.4, SD-HuBERT 47.6, **ours 26.6**.

This is the most direct template for our situation: an EMA-teacher latent-prediction model whose pooled output leaks identity, fixed by (clean teacher, perturbed student). The perturbation was tuned to be *realistic*; the authors note unrealistic perturbations (doubling female pitch) degrade results [V].

### 1.5 WavLM utterance mixing, DinoSR, data2vec, HuBERT iterations

- **WavLM (arXiv 2110.13900) [V Tab. I].** Utterance mixing (a secondary speaker overlaid on up to 50% of the clip) is *not* a speaker-removal device: Base SID 84.51, ASV EER 4.69, SD DER 4.55; without the denoising/mixing task SID 84.39, EER 4.91, DER 6.03. It improves *diarisation* (22.6% rel. over HuBERT Base), i.e. teaches the model to separate speakers, and leaves SID unchanged. WavLM Large adds +6.07 abs. SID by scale. **Transfer verdict:** mixing is the wrong direction for invariance; it is useful only if a task needs signer separation (not ours).
- **DinoSR (arXiv 2305.10005) [V Tab. 1].** EMA teacher + online clustering of teacher layer outputs, no speaker mechanism, yet ABX (same/cross speaker; clean, other): DinoSR 3.08/3.43/3.42/4.42 (avg 3.59; layer 5), vs HuBERT 3.07/3.90/3.71/6.19, data2vec 4.03/5.09/4.72/6.97, ContentVec 2.98/3.70/3.44/5.17, wav2vec 2.0 4.15/5.22/4.82/7.38. DinoSR has the smallest cross-vs-same speaker gap among non-Spin models (clean 0.34, other 0.99). The limitation section states that "other contents (e.g., speaker/paralinguistic information) might be neglected". **[I]** a discrete codebook over EMA-teacher features acts as a nuisance bottleneck; the effect is not isolated by an ablation, so treat as hypothesis.
- **data2vec (arXiv 2202.03555).** Regresses continuous contextualised EMA targets; I found no speaker-leakage analysis in the paper. Its ABX gap (0.69 clean, 1.88 other, from DinoSR's table) is as large as HuBERT's. **[I]** continuous latent targets (our JEPA case) do not bottleneck nuisance; discretisation or explicit invariance is needed.
- **HuBERT iterations.** ContentVec reports that HuBERT-iter has "slightly lower SID accuracy than HuBERT" [V text of 4.5]; the HuBERT paper gives no speaker analysis. Iteration alone is a weak nuisance filter; ContentVec's 36% SID reduction is relative to that.

### 1.6 Voice-conversion-based and perturbation-based approaches (summary)

NANSY (arXiv 2110.14513) is the origin of the formant/pitch/EQ perturbation reused by ContentVec/Spin [V via their text]; it uses it to teach a *synthesis* network to rely on a stable speaker embedding, not to remove speaker from an encoder. Voice-converted *teachers* (ContentVec) need a trained VC model; perturbed *students* (Spin, Komatsu) need only an analytic transform. The sign analogue of an analytic voice transform exists (analytic 2D retargeting, §2.4); the analogue of a trained VC model is skeleton retargeting networks (Aberman et al., arXiv 2005.05732 [V abs]: unpaired retargeting between homeomorphic skeletons through a shared "primal" latent), which are 3D, motion-capture-trained and not suited to noisy 2D WholeBody.

### 1.7 Transfer to skeleton sign retrieval

**Verdict: HIGH relevance for mechanism (b) and (c), MEDIUM for (a), LOW for utterance mixing and trained VC.**
- Transferable and cheap: asymmetric clean-teacher / perturbed-student JEPA; style-conditioned predictor; invariance regulariser at an *intermediate* layer; discrete/clustered teacher target; avoid pooled-token objectives for the readout feature.
- Not transferable: utterance-level mean/variance normalisation (§0 item 9), speaker labels to train a speaker-embedding net (label rule), WavLM mixing.
- Expected magnitude [I]: speech gains were 5-30% relative on invariance-sensitive metrics; our nuisance is larger (signer decodable 94%, same-signer NN dominant) but part of it is *behavioural* (handedness, speed, signing space, dialect) which analytic morphology perturbations cannot touch.

**Minimal tests.**
1. *Morphology-sensitivity probe (CPU/GPU inference only).* For 2k val clips, apply an analytic 2D retarget (per-bone scale s_b ~ U[0.85,1.15], clip-constant, forward-kinematics re-integration, aspect/shear jitter). Compare cos(emb(x), emb(morph(x))) with cos between same-gloss different-signer pairs. If the former is *lower* than the latter, the encoder is more sensitive to morphology than to signer-swap-with-same-gloss, and perturbation training has headroom.
2. *Within/across-signer gap* (speech ABX analogue): R@1 with gallery restricted to same-signer vs different-signer queries; track the gap, not just R@1.
3. *Short JEPA run* (one seed pair): teacher sees median-skeleton canonicalised input, student sees morph(x); report signer-ID linear accuracy, same-signer NN rate, R@1 with signer-clustered bootstrap CI.

---

## 2. Identity/style disentanglement in motion, skeleton and gesture

### 2.1 Identity is decodable from skeletons, including 2D sign poses

- **Sign pose.** [V 2410.13675] On AUTSL, a frozen sign-classifier network re-trained with a signer head gets signer accuracy **80.18%** (chance 3.23%) from original poses; **65.34%** after transferring an *average-proportioned body* (appearance of frame 0, Eq. 1: P - P0_alpha + P0_beta, shoulder-width-normalised, hands unchanged); **52.20%** after ensemble over 10 random-body transfers. Sign accuracy (Tab. 2, rows = train, cols = test: original / anonymised / transferred): train original 80.97 / 65.82 / 71.46; train anonymised 63.26 / 64.48 / 51.50; train transferred 67.08 / 66.54 / 57.32; train combined 79.96 / 60.88 / 76.78. The authors conclude it "does not eliminate" identity and "hurts sign language recognition".
  - Reading [I]: (a) body proportions are a *substantial but not dominant* part of pose identity (about 15-28 points of 80, with hands untouched and the rest from hand geometry, prosody, movement style, sign choice); (b) even *training* on anonymised poses loses ~16 points, so removing body-relative information removes content (sign location relative to body), which cautions against aggressive retargeting.
  - Caveat: the signer head sits on a frozen *supervised* sign network, not on an SSL embedding; the 80% is a probe on a representation tuned for signs.
- **Generic skeleton.** [V 2111.15129, AAAI 2023] On NTU-60, supervised re-identification from skeleton trajectories reaches **80% top-1 / 97% top-5** (Shift-GCN 79.62, MS-G3D 82.23, 2s-AGCN 76.89) and gender 87% on ETRI-Activity3D (85.99/87.90/86.43). Their adversarial anonymiser takes identification from 0.8095 to 0.0420 with action accuracy 0.9510 -> 0.9175 (Tab. 1; their own Fig. 5 notes a train/validation gap on the privacy classifier, i.e. overfitting of the attacker). Caveat: the anonymiser is trained against labelled subjects.
- **Gait.** GaitGraph (arXiv 2101.11228 [V title/abs]) and SGELA/SM-SGE (2008.09435, 2107.01903 [S abs only]) show that 2D/3D skeleton sequences identify people; I did not extract numbers, they do not change the conclusion.

### 2.2 Skeleton SSL and nuisance: what the augmentation literature actually tests

- Standard skeleton-SSL augmentations are *view* augmentations (shear, rotation, flip, axis mask, joint mask, temporal crop), none targets limb length. STARS Tab. 7 [V 2407.10935, kNN K=10, NTU-60 xsub/xview]: spatial flip 85.0/90.6, rotation 84.8/90.1, axis mask 81.2/89.0, shear 83.2/90.4, none 84.5/89.6; the authors drop augmentation because the effect is "minor" and sometimes negative.
- Supervised sign recognition does benefit from generic keypoint augmentation: SAM-SLR on AUTSL (signer-independent test), "w/o keypoints augmentation" 95.02 -> 90.16 [V 2110.06161 Tab. III; augmentations are random sampling, mirroring, rotation, scaling, jitter, shift; none is bone-length-specific].
- A bone-length-scale augmentation reported for 3D MediaPipe hand keypoints (BISINDO alphabet, cross-domain F1 67.20 -> 89.55) [S, journal page blocked by Cloudflare; value seen only in a search summary]. It is the only *direct* evidence of bone-length perturbation helping cross-subject generalisation; 3D, hand-only, alphabet-level.
- **No label-free skeleton-SSL paper that I could find removes body shape from the *representation*.** The only sign precedent is SignRep's **targets**: angles converted back to 3D keypoints with the dataset-median bone length "avoiding the leakage of person-specific features" [V 2503.08529 appendix]. Its gain is bundled with other priors.

### 2.3 Retargeting and style transfer (analytic vs learned)

| Approach | Mechanism | Needs | Relevance |
|---|---|---|---|
| Skeleton-aware retargeting (Aberman et al., 2005.05732) [V abs] | learned unpaired retargeting through a shared primal skeleton latent (skeletal pooling) | 3D mocap, adversarial cycle | shows that identity (skeleton morphology) can be factored from motion in a latent; not usable on noisy 2D |
| Everybody Dance Now (1808.07371) [V §3.1/§8.5] | per-frame scale+translation from heights between nearest/farthest ankle positions; limb proportions *not* retargeted | 2D pose, video pairs | confirms that 2D pipelines normalise scale/position only |
| Pose appearance transfer (2410.13675) [V] | analytic: subtract own frame-0 pose, add target frame-0 pose; body+face only | rest first frame | cheap; costs ~16 points of AUTSL accuracy (see 2.1) |
| JOKR (CGF 2022) [S] | joint 2D keypoint representation for unsupervised video retargeting, different limb lengths | video pairs | 2D precedent; no arXiv id found |
| DisUnknown (2109.08090) [V title] | disentanglement of labelled factors (identity, view) from an unknown one (motion) on Mixamo 2D projections | factor labels | needs identity labels |

### 2.4 What does 2D allow? [I]

- True bone-length retargeting needs 3D: a 2D bone length is L cos(theta) with limb pitch theta, so a clip-constant per-bone scale cannot be estimated from one frame. Practical 2D invariants: (i) clip-level *robust maximum* (e.g., 90-95th percentile over frames) of each normalised segment length approximates L when the limb is near the image plane; (ii) shoulder-width and torso-height normalisation (already standard, EDN); (iii) unit 2D direction vectors/angles are invariant to *length* scaling by construction but depend on view; (iv) hand segment lengths are heavily foreshortened and noisy (hand AP 58-66 on COCO-WholeBody per L4), so retarget hands only by global hand-scale ratio, not per-finger.
- Analytic 2D morphology augmentation that is self-consistent: sample clip-constant factors for (a) shoulder width / torso height ratio, (b) upper-arm and forearm length, (c) hand scale relative to forearm, (d) face scale; recompute joint positions by forward kinematics along the kinematic chain from the unit 2D bone vectors, then re-attach translation. This equals SignRep's "angles -> median bone length" in 2D.
- What 2D cannot remove: handedness, speed/rhythm, signing-space size (partly morphology), resting posture, dialect/lexical variants, camera tilt interacting with posture. The AUTSL result (hands untouched, 15-28 point reduction) suggests morphology explains only part of identity.

### 2.5 Transfer verdict and minimal test

**Verdict: MEDIUM.** Morphology invariance is a necessary, cheap, partial fix; the literature gives no evidence on SSL-embedding retrieval, only on supervised classifiers and anonymisation.
**Minimal CPU test:** from ASL Citizen keypoints compute ~20 clip-level static descriptors (robust-max normalised bone lengths, rest-frame offsets). (a) Linear signer-ID from descriptors alone (how much of the 94% is morphology). (b) Regress the descriptors out of the embeddings (ridge, fit on pretraining signers), recompute R@1 and same-signer-NN rate. If (b) moves R@1 by < 1 point, skip the morph-augmentation training run (strategy 3 below).

---

## 3. Linear nuisance removal and its limits

### 3.1 Methods and guarantees

- **LEACE (Belrose et al., arXiv 2306.03819) [V].** Linear guardedness (no linear classifier beats a constant predictor) holds iff the class-conditional means equal the global mean, iff cross-covariance Sigma_XZ is zero (Thm 3.4); an affine eraser r(x)=Px+b guards iff colsp(Sigma_XZ) is in ker(P) (Thm 4.1). LEACE is the least-squares-optimal oblique projection (whiten, project off col(W Sigma_XZ), unwhiten). For k signers the removed subspace has rank <= k-1. Stated limits: only *linear* adversaries; kernel-based erasers "do not generalize well to other kernels"; the authors "conjecture that it is intractable to nondestructively edit X so as to prevent a general nonlinear adversary". The guarantee is on the fitted distribution; leakage on new data is not guaranteed.
- **INLP (Ravfogel et al., arXiv 2004.07667) [V].** Iterate: train linear classifier -> project to its nullspace; 35 iterations in the word-embedding experiment. Gender: linear 100% -> 49.3%, but a 1-layer MLP still recovers **85.0%**; k-means V-measure with gender 83.88% -> 0.44%; and *SimLex-999 correlation improves 0.373 -> 0.489* after projection (removing a dominant nuisance improved similarity quality).
- **RLACE (Ravfogel et al., arXiv 2201.12091) [V].** Minimax over rank-constrained orthogonal projections; a rank-1 projection takes frozen-BERT gender accuracy 99.84 -> 52.16 +- 0.13 with profession 79.91 -> 79.21 (binary attribute).
- **Hard-debiasing does not change neighbourhoods (Gonen & Goldberg, arXiv 1903.03862) [V].** After projecting out the gender direction, k-means (k=2) on the 1000 most biased words still aligns with gender 92.5% (Hard-Debiased; 99.9% before) and 85.6% (GN-GloVe; 100% before); correlation between original bias and number of male nearest neighbours stays 0.606/0.792 vs 0.747/0.820. **This is the closest published analogue of "signer is erased linearly yet neighbours are still same-signer".**

### 3.2 CMN/CVN and speaker normalisation in SSL speech (the one setting with held-out evidence)

- van Niekerk et al. (arXiv 2108.00917, in C §2.6): per-utterance standardisation drops the linear speaker probe to 14.8%, MLP stays 89.0%.
- **Liu, Tang, Goldwater (arXiv 2305.12464, Interspeech 2023) [V]** is the key paper for us. Speaker subspace = PCA of per-speaker mean vectors (M_spk) of frame features; collapse = project off the top components. Subspace fitted on LibriSpeech train-clean-100 (251 speakers, 25 min each), applied to *unseen* speakers; number of dims chosen on dev across-speaker ABX (98%/95%/95% of training variance = 57/36/30 dims for CPC-big/CPC-small/APC; "after about 50 dimensions ... ABX error also begins to rise" i.e. overfitting). Results (Tab. 3; ABX error %, within/across):

| Model, split | Original within/across | Utt.-centred | Collapsed |
|---|---|---|---|
| CPC-big, dev | 3.38 / 4.11 | 3.44 / 4.03 | 3.14 / 3.80 |
| CPC-small, dev | 6.22 / 8.10 | 5.86 / 7.26 | 5.32 / 6.91 |
| APC, dev | 6.77 / 9.83 | 6.42 / 9.01 | 6.45 / 9.24 |
| CPC-big, test | 3.29 / 4.22 | 3.27 / 4.11 | 3.10 / 4.01 |
| CPC-small, test | 5.86 / 7.48 | 5.54 / 6.91 | 4.85 / 6.37 |
| APC, test | 6.62 / 9.47 | 6.13 / 8.72 | 6.18 / 8.98 |

  Linear speaker-probe error rises to 65.5/65.1/69.5% (dev; chance-level would be higher) so speaker info is largely but not fully removed. Phone probe error barely changes. Centring alone beat centring+rescaling (rescaling raised both speaker and ABX errors) [V]. Also: speaker and phone subspaces are nearly orthogonal (mean max cosine 0.13 for top-20 speaker directions) [V].
  - Setting caveats: predictive-coding models (CPC, APC), not masked prediction; frame-level features with thousands of frames per speaker; 251 fitting speakers; read speech. Gains are 5-15% relative and **absent for APC test vs centring** (8.98 vs 8.72).
  - Orthogonality of speaker and content in the learned space is an *empirical property of these models*; no one has shown it for masked/JEPA skeleton embeddings. [I]
- **Isotropy is not the right target.** Mohamed et al. (arXiv 2406.09200) [V]: isotropy of all frames correlates with phone accuracy in only 2 of 6 models, but isotropy of *phone class centroids* correlates strongly and consistently. For retrieval the relevant isotropy is that of gloss centroids, which whitening of all embeddings does not directly target. [I]

### 3.3 Adversarial removal and its failure modes

- **Elazar & Goldberg (arXiv 1808.06640) [V].** At epoch 50 the adversary is at 49.0% (chance 50%) while a freshly trained attacker on the same encoder reaches **56.0%**; "in all cases, the adversarial's success rate is around 50%, while the attacker's rate is substantially higher". Protected-attribute leakage in an encoder trained on an unrelated task: race 64.5% (balanced), up to 73.8% (unbalanced).
- **Subject-aware contrastive learning for biosignals (Cheng et al., arXiv 2007.04871) [V Tab. 2].** EEG, 16 held-out subjects: subject-ID accuracy of frozen encoder Base SSL **88.6** vs subject-invariant (adversarial) **73.0** vs subject-specific loss 68.4 (chance ~6%): adversarial invariance leaves ~67 points above chance. Task accuracy 2-class inter-subject 78.8 -> 79.0, 4-class 49.4 -> 49.8: essentially no downstream gain unless few training subjects. Gains appear only "with fewer subjects".
- **Signer removal with gradient reversal (Zuo, Mak et al., arXiv 2212.13023) [V Tab. 7, 8].** PHOENIX-2014-SI WER 34.4 -> 32.7 (statistics pooling + GR; multi-task without GR 33.5); seen/unseen WER 22.7/34.4 (relative gap 51.5%) -> 23.0/32.7 (42.2%). Signer embeddings computed at sentence level (statistics pooling) worked better than frame-level; lambda=0.75 best, lambda=0 degrades to baseline.
- **SignRep style adversary** [V 2503.08529 Tab. 5]: +0.8 DCG (49.9 -> 50.7, single run, WLASL DCG, stride 8); no signer-ID probe reported. Adversary applied only when E_M[D] > E_U[D] (A §1.3).
- **Takeaway [I].** Always evaluate with a fresh post-hoc probe and with neighbourhood statistics (same-signer NN rate), not with the training-time adversary. Expect adversarial gains of 1-2 task points at best.

### 3.4 Does post-hoc removal improve retrieval/kNN in SSL features? Evidence summary

| Intervention | Domain | Metric | Effect | Tag |
|---|---|---|---|---|
| speaker-subspace collapse (30-57 dims, unseen speakers) | CPC/APC speech | across-speaker ABX | -0.31/-1.19/-0.59 abs. (dev), -0.21/-1.11/-0.49 (test) | V |
| remove 10 / 200 highest-variance dims | MAE patch features | few-shot segmentation kNN (patch-level) | 0.058 -> 0.170 -> 0.295 (m=10 -> m=200); linear unchanged until m>200 | V 2401.00463 |
| All-but-the-top (mean + D~d/100 top PCs) | word vectors | STS (21 sets) / word similarity (7 sets) | avg +4% / +2.3% | V 1702.01417 |
| INLP (35 iters) | GloVe | SimLex-999 | 0.373 -> 0.489 | V |
| hubness reduction (f-norm + mutual proximity) | Sentence-BERT | kNN error | -7% to -13% (own models), -7% to -9% (pretrained), hubness -69% to -83% | V 2311.18364 |
| CSLS vs NN | cross-lingual lexicon | P@1 | up to +7.2 (en-eo 22.1 -> 29.3) | V 1710.04087 |
| signer-subspace collapse / whitening / LEACE | our embeddings | val R@1 | 22.7 -> 29.4 (whiten k=512) / 29.7 (LEACE signer) / 31.7 (both); oracle per-signer offset 28.4 | L, unverified |

The literature pattern is consistent: **removing the dominant shared or nuisance directions raises neighbour quality by single-digit relative percentages in published settings; larger gains appear when the baseline is badly anisotropic (MAE patch kNN).** Our [L] numbers (+30-40% rel.) are at the extreme end; they could be real because our embeddings are an extreme case (R@1 16.7-22.7 vs ~49 for SignRep), but they should be re-derived with a signer-disjoint fit before being cited.

### 3.5 Transfer verdict and minimal test

**Verdict: HIGH as a first, cheap, falsifiable step; LOW as a final solution.** Linear removal plus whitening is the baseline that any training-time disentanglement must beat; otherwise the training change buys nothing over post-processing.
**Minimal CPU test (embeddings only).** On the frozen JEPA embeddings: (a) centre + ZCA whiten with shrinkage, fit on *pretraining-corpus* embeddings (label-free), k chosen on val; (b) PCA speaker-style collapse with signer means from *training* signers only (declared upper bound), cap dims at <= 20 (we have O(40) signers, not 251) and pick by signer-grouped CV; (c) drop top-D PCs (D in {d/100, 10, 50, 200}). Report R@1, same-signer NN rate, signer-ID linear and MLP accuracy, signer-clustered bootstrap CI. Falsified if signer-ID MLP stays > 80% and same-signer NN rate unchanged while R@1 gain < 1 point.

---

## 4. Retrieval geometry of frozen SSL embeddings

### 4.1 The kNN-vs-linear gap is large for masked/predictive objectives and shrinks with a short NN-contrastive stage

Vision, ImageNet, same protocol, linear probe / kNN [V 2304.10520 Tab. 1, Tab. 14]:

| Model | Linear | kNN | Gap |
|---|---|---|---|
| MAE ViT-B/16 | 68.0 | 51.1 | 16.9 |
| MAE ViT-L/16 | 76.0 | 60.6 | 15.4 |
| MAE ViT-H/16 | 78.0 | 61.1 | 16.9 |
| MAE ViT-H/14 | 77.2 | 58.9 | 18.3 |
| I-JEPA ViT-H/14 | 79.3 | 71.6 | 7.7 |
| MoCo v3 B / MSN B / iBOT B | 76.7 / 77.7 / 79.5 | 72.6 / 76.3 / 77.1 | 4.1 / 1.4 / 2.4 |
| MAE-CTmin B / L / H16 | 73.5 / 80.2 / 81.5 | 64.1 / 77.4 / 79.4 | 9.4 / 2.8 / 2.1 |

Latent prediction (I-JEPA) roughly halves the MAE gap but does not close it; contrastive/clustering objectives have gaps of 1-4 points. **Our observation (JEPA R@1 16.7 vs SignRep 49) is the same phenomenon in the frozen-dictionary protocol** [I; the kNN here is class-of-nearest cosine, equal in spirit to our protocol].

Skeleton, NTU, linear (Tab. 1) vs kNN K=1 cosine over the training set (Tab. 2) [V 2407.10935]:

| Model | NTU-60 xsub lin/kNN | xview | NTU-120 xsub | xset |
|---|---|---|---|---|
| MAMP (masked motion prediction) | 84.9 / 63.1 | 89.1 / 80.3 | 78.6 / 51.8 | 79.1 / 56.1 |
| STARS (MAMP + ~20 epochs NNCLR tuning) | 87.1 / 79.9 | 90.9 / 88.6 | 79.9 / 67.6 | 80.8 / 67.7 |

Few-shot (pretrained on NTU-60, 60 novel NTU-120 actions; Tab. 5): MAMP 47.6/44.4/48.4 (1/2/5-shot) vs STARS 63.5/62.2/65.7. Naive MAE as first stage: kNN 44.1/43.7 -> 53.5/55.2 after STARS tuning (Tab. 14). Tuning strategy (Tab. 6, kNN K=10, xsub/xview): DINO 77.6/86.3, MoCo 72.2/86.7, **NNCLR 81.9/89.6**. NNCLR trained from scratch on the skeleton transformer for 300 epochs: kNN K=1 only 37.6/30.5 (Tab. 12): it needs a pretrained encoder.

### 4.2 Why (evidence and hypotheses)

1. **High-variance nuisance dimensions dominate distance but not linear probes.** [V 2401.00463] MAE/SimMIM patch embeddings have a handful of very high-variance features ("roughly 200 dataset-agnostic features ... contain no useful information ... while having the highest variances"); removing the top 10 / 200 raised kNN few-shot segmentation from 0.058 to 0.170 / 0.295 without hurting the linear probe, then both declined. A linear classifier can down-weight those axes; cosine kNN cannot. Setting caveat: patch-level segmentation, ViT-B, not global retrieval. Analogous to signer/framing directions in our embeddings [I].
2. **Reconstruction/prediction objectives impose no metric structure on class-irrelevant variation.** MAE-CT's oracle (neighbour restricted to same class) reaches 83.9% linear [V], and "NNCLR without NN lookup" collapses kNN to 40.5 [V Tab. 3, ViT-L], i.e., the *cross-instance* positive is what creates class-level metric structure, not the contrastive form.
3. **Pooling choice matters for kNN.** In iBOT supplementary tables kNN is 71.5 with [CLS] but 49.0 with global average pooling (search summary of AttMask supp. [S]); our protocol is mean-pooling.
4. **Anisotropy/collapse.** Covered by C §1.5/§2 (RankMe, LiDAR); the new point from Mohamed et al. [V] is that *centroid* isotropy, not frame isotropy, tracks class accuracy.
5. **Hubness.** SBERT embeddings are hubby and hubness reduction lowers kNN error 7-13% [V 2311.18364]; "training on more data does not solve the hubness problem" [V]. CSLS (arXiv 1710.04087) = 2cos(x,y) - r_T(x) - r_S(y), with r the mean cosine to the k nearest neighbours in the *other* domain; +7.2 P@1 on en-eo [V]. In sign retrieval the "hub" gallery clips are probably clips from low-variance/neutral signers [I].

### 4.3 Scoring rules for the gallery: class-max vs alternatives [I]

- Class = max over gallery clips (our protocol) is a *max over ~15 draws per class*: a single hub clip (or a signer cluster close to the query's signer) can decide the class. It is the most hub-vulnerable aggregation and the one most affected by same-signer proximity. Mean over the class (centroid) is robust to hubs but blurs lexical variants and signer styles. Intermediate: top-m mean (m=3), log-sum-exp with temperature tau (soft-max pooling), or CSLS-corrected cosine before the max. DINO evaluates a weighted-vote kNN and states that the momentum teacher and multi-crop are "crucial for good k-NN performance" [V 2104.14294; I did not verify its kNN temperature].
- Local-vs-global scaling: local scaling / mutual proximity / CSLS fix *density* heterogeneity; they do not remove a nuisance that is shared between query and neighbour (same signer), which is a *similarity* problem, not a hub problem. Combine with signer-aware correction.
- **Minimal test (CPU):** on fixed embeddings, grid over {max, top-3 mean, LSE tau in [0.05, 0.2], class centroid} x {raw, centred, whitened, CSLS with k=10 using gallery-gallery or pretraining-bank neighbours}. Report R@1/R@5/DCG, hubness skewness of gallery clips (k=10 occurrence N_k), and the fraction of top-1 errors that are same-signer.

### 4.4 Transfer verdict

**Verdict: HIGH.** The kNN-vs-linear gap is the best-documented explanation for "JEPA features look fine for probing but poor for dictionary retrieval", and it has two published repairs: (i) post-hoc removal of high-variance/dominant directions and hubness correction (cheap, partial), (ii) a short NN-contrastive tuning stage (MAE-CT, STARS; expensive, large). Treat "kNN vs linear probe gap" as a mandatory diagnostic for every checkpoint. Caution: STARS and MAE-CT started from encoders with 51-63% kNN accuracy; the technique is a *refinement*, not a rescue (see §5).

---

## 5. Cross-instance positives without labels

### 5.1 Mechanisms and numbers

- **NNCLR (Dwibedi et al., arXiv 2104.14548, ICCV 2021) [V].** Positive = top-1 nearest neighbour of view 1 in a support queue (32,768-98,304 embeddings). ImageNet linear: SimCLR-aug positive 71.4 -> NN positive 74.5 (74.9 with momentum encoder). Hard top-1 NN 74.9 vs soft NN 71.4; k=1 best, k=32 73.2; support size 8,192 -> 98,304 gives 73.6 -> 75.4. **With crop-only augmentation:** SimCLR 67.9 -> 40.3 (-27.6), BYOL 72.5 -> 59.4, **NNCLR 72.9 -> 68.2 (-4.7)**: NN positives substitute for hand-designed augmentations. Oracle same-class NN: 75.8 vs 72.9 (300 epochs). NNCLR needs ~120 epochs to overtake SimCLR (harder task).
- **MAE-CT (arXiv 2304.10520) [V].** After MAE, 20 epochs, top half unfrozen (layer-wise lr decay 0.65), crop+flip only, top-k NN lookup. ViT-L ablation (probe/kNN): full 80.2/77.4; masking during CT 79.4/75.3; skip NNCLR-head initialisation 78.5/68.1; **no NN lookup 70.6/40.5**. Combined (joint) MAE+NNCLR pre-training: 77.8/66.1 vs sequential 80.2/77.4. The NN lookup's data-driven augmentation "is required to make the NNCLR task difficult enough".
- **STARS (arXiv 2407.10935; WACV 2026 per a search result [S]) [V].** Same recipe on skeleton MAMP (see 4.1). kNN K=1 +16.8 on NTU-60 xsub; queue size has little impact; tuning-strength alpha 1.0 -> 0.2: 74.3 -> 83.7 (K=10).
- **MSN / SwAV / DINO prototypes.** MSN B/16 linear 77.7 / kNN 76.3 [V via 2304.10520 Tab. 1], i.e. prototype-assignment objectives are kNN-friendly by construction. Spin (§1.2) is the speech instance: Sinkhorn-balanced prototype assignment *across perturbed views*.
- **MASA (arXiv 2405.20666) [V], the only label-free skeleton method with frozen ASL Citizen retrieval (27.96).** Masked motion prediction + MoCo-style instance contrast between a clip and a randomly temporally subsampled copy, memory bank negatives, EMA key encoder, mu=0.996, loss weight 0.05. The semantic-alignment branch adds +1.42 fine-tuned WLASL2000 per-instance top-1 (47.64 -> 49.06; SA alone 43.61) [V Tab. VIII]. **Both positives share the same clip, hence the same signer/session; [I] this is why its frozen retrieval, while the best label-free pose result, stays at 27.96.**
- **Gloss-supervised ceiling for the same idea.** SignSeek's GAC term (positives = same gloss across clips and signers, [V 2609.03695 Eq. 7]) alone gives frozen ASL Citizen R@1 48.40 / DCG 70.02, rising to 57.66 / 77.21 with articulator masking (Tab. 7: GAC 48.40; +MAC 54.15; +MAP 55.93; MAC+MAP without GAC 55.12; all 57.66). So cross-signer positives with labels move a pose encoder from about 28 (same-clip positives, MASA) to 48-58 [V; different corpora, backbones and data, so not a controlled comparison].

### 5.2 The known failure mode: neighbours that share the nuisance

- **No ablation exists in the NNCLR/MAE-CT/STARS papers for nuisance-correlated neighbours** (ImageNet and NTU classes are not confounded with the subject in the same way) [V negative: grepped]. Speech and re-ID literature does address it:
  - **Speaker verification (SSPS, Lepage & Dehak, arXiv 2505.14561) [V].** Standard same-utterance positives "primarily encode channel information from the recording condition, shared by the anchor and positive". Latent-space positive sampling (cluster assignments + queue; positive from the anchor's cluster, M=0, or a neighbouring cluster, M=1) takes SimCLR (ECAPA-TDNN) from EER 6.30 to **2.57** and DINO 3.07 -> 2.53; with labelled cross-recording positives 1.72/2.36. Fast-ResNet-34 SimCLR: baseline 9.41; K=6,000 M=0 6.63; K=25,000 M=1 5.80; supervised 3.93. (Here the *target* is the speaker, so the logic is mirrored: positives must differ in the nuisance (channel) and share the target.)
  - **Camera-aware proxies (CAP, arXiv 2012.10674, AAAI 2021) [V Tab. 1].** Unsupervised person re-ID: clusters are split into per-camera proxies; intra-camera loss + inter-camera loss over positive and hard-negative proxies. Baseline (cluster = identity) -> full CAP6: Market-1501 R1 79.7 -> 91.4 (mAP 62.9 -> 79.2), DukeMTMC 74.3 -> 81.1 (57.5 -> 67.3), **MSMT17 R1 34.0 -> 67.4 (mAP 13.7 -> 36.9)**; inter-camera term alone already gives MSMT17 66.3; intra-camera alone is worse than baseline on Market (78.7). Camera id is free metadata in re-ID, i.e. identical status to our signer id (diagnostic/upper bound only).
  - **DN2A (Liu et al., CVPR 2023) [S].** "Dual nearest neighbors" for unsupervised domain generalisation: strong augmentation to suppress intra-domain connectivity plus a cross-domain NN search; abstract claims +12.01% PACS, +13.11% DomainNet (1% labels, linear). No arXiv version found; numbers from the CVPR page summary only.
  - **DUDE (arXiv 2509.04193, 2025) [V title]** builds positives in stages: instance -> in-domain mutual NN -> cross-domain mutual NN (diffusion-based unsupervised cross-domain retrieval). I did not extract its ablation.
  - **Subject-aware CL (2007.04871) [V]**: negatives confined to one subject ("subject-specific loss") reduces subject-ID decodability from 88.6 to 68.4 with essentially no downstream change at 90 training subjects. Principle: *same-nuisance negatives* remove the shortcut.
  - CPC within-speaker negatives: Liu et al. state within-speaker sampling "gives better phone classification results" than unrestricted sampling [S: citing another paper; I did not locate the ablation].
- **False-neighbour literature.** MNN (arXiv 2311.00562 [S: id and abstract from a search result, not fetched]) names "false neighbours" and proposes weighting/mixing; not nuisance-specific.
- **Mechanism in our setting [I].** (a) Precision: with val R@1 16.7 the probability that a *cross-signer* nearest neighbour has the same gloss is of the order of R@1 itself (<~20%), against 51-61% kNN accuracy at which MAE-CT/STARS started and 37.6 (STARS from scratch) where NNCLR already failed; (b) the *most confident* neighbours are the most same-signer (82% in the top 10% per L4/A, project measurement), so confidence filtering selects the wrong pairs; (c) adding same-signer different-gloss neighbours as positives tells the model that signer identity is a positive-defining feature.
- **2025-2026 follow-ups.** I found no JEPA-with-NN-positive paper and no sign-language NN-mining paper in two targeted searches (extended mode); DMT-JEPA (semantically similar neighbouring *patches* within an image) and Point-JEPA (kNN patch grouping) are the nearest hits and are not cross-sample. **Gap confirmed for [I]: cross-signer NN positives in a JEPA/skeleton setting is novel as far as this search shows.**

### 5.3 Fixes available [I, from the evidence above]

1. Exclude same-(pseudo-)signer candidates from the NN lookup; pseudo-signer = k-means (k=50-200) on clip-level *static* descriptors (rest-frame pose, robust-max bone ratios) or on the embedding after top-PC removal, never on gloss info.
2. Mutual-NN + cross-cluster requirement (DUDE/CAP style), top-1 hard NN (NNCLR k=1 best), queue >= 32k (here the corpus supplies it).
3. Bootstrap order: raise cross-signer NN precision by Strategies 1-3 *first*; run NN tuning only when cross-signer val kNN precision (not R@1) exceeds ~40-50% [I threshold from MAE-CT start 51-61 and STARS-scratch 37.6 failure].
4. Head initialisation on frozen features before tuning (MAE-CT: skipping costs 9.3 kNN points).
5. Never use the signer label for training in the declared label-free method; use it for (a) oracle upper bound and (b) diagnostic precision of the pseudo-signer clustering.

### 5.4 Transfer verdict and minimal test

**Verdict: MEDIUM-HIGH potential, HIGH risk if applied before signer nuisance is reduced.**
**Minimal CPU test (before any training).** On fixed embeddings, for val queries mine nearest *cross-pseudo-signer* gallery neighbours; measure the same-gloss rate of (a) top-1 cross-signer neighbours, (b) mutual cross-signer NN pairs, (c) pairs after whitening/erasure. This is the positive-precision curve; run NN tuning only if (b) >= 40%. Add an oracle arm using true signer IDs (diagnostic) to separate pseudo-signer error from the intrinsic difficulty.

---

## 6. Sign-specific evidence

### 6.1 ASL Citizen paper (Desai et al., arXiv 2304.05934, Apr 2023; venue not verified) [V]

- 83,399 videos, 2,731 signs, 52 signers (49 Deaf, 3 hard of hearing; 32 F/20 M); splits are **signer-disjoint** and the seed signer (a trained, high-resolution-camera model) is placed in train. "we felt it critical to evaluate on unseen users to align with our dictionary retrieval problem"; earlier work "randomly split videos so users in the test set may be seen in training".
- Supervised baselines (test, unseen signers): I3D R@1 63.10, DCG 0.7913, MRR 0.7332, R@5 86.09; ST-GCN (pose) R@1 59.52, DCG 0.7637, R@5 82.68. Random guessing 0.04%.
- **Frozen-feature kNN of a supervised model** (cosine NN against ASL Citizen training features): ASL Citizen-trained I3D 58.03 R@1 vs WLASL-2000-trained I3D 9.85 (full test), 71.07 vs 16.49 on the WLASL-overlap subset; Kinetics-trained I3D "close to random". So a supervised in-domain encoder keeps ~92% of its head accuracy under frozen kNN (58.03/63.10), while a cross-dataset encoder collapses; signer/domain distribution, not vocabulary, is the barrier.
- Training-size sensitivity (Tab. 9, Fig. 4): "the scale of our training set is critical".
- Statement on prior work: "Models trained on a small number of dataset contributors ... may not generalize well to diverse signers".
- Implication [I]: ~35 train signers (our gallery, in [L]) is far fewer than any speech SSL setting where speaker subspaces were validated (251); signer-count scaling of R@1 in [L] (4 -> 35 gallery signers: 9.8 -> 22.7) points the same way.

### 6.2 SignRep's treatment of style [V 2503.08529, A §1.3]

- "Style" = background and appearance, computed as the Gram matrix of token embeddings averaged over columns; positives = two random crops of the *same video*; negatives = another video in the batch; spectral-norm MLP discriminator; hinge on EMA expected outputs applied only when E_M[D] > E_U[D]. Median-bone-length normalisation applies to *targets* (appendix), not inputs.
- Ablation: only Table 5 (angle/kpt/dist/mask/var+cov/adv, WLASL DCG at stride 8): adv adds 49.9 -> 50.7; no ablation of bone normalisation, no signer-ID probe, no ASL Citizen ablation, no seeds (A §1.5). **Signer invariance is therefore asserted, not measured.** The project's finding (same-signer NN 48.7% > same-gloss 43.8%) shows the invariance is incomplete.
- **Portable idea [I].** The "same video => same style" pairing is a label-free signer/session proxy (clip-level for a skeleton model: segments from the same clip share morphology; style descriptor = static pose statistics). Its weakness: negatives from other videos may be the same signer, so the discriminator is partly trained on false negatives.

### 6.3 Signer-independent sign recognition: methods and measured effects

| Method | Mechanism | Effect [V] |
|---|---|---|
| Signer removal module (2212.13023) | sentence-level signer embedding via statistics pooling + gradient reversal | PHOENIX-2014-SI WER 34.4 -> 32.7; relative unseen/seen gap 51.5% -> 42.2% |
| Keypoint augmentation (2110.06161) | random sampling, mirror, rotate, scale, jitter, shift | AUTSL 95.02 -> 90.16 when removed |
| Appearance transfer (2410.13675) | frame-0 offset, average or random body | signer-ID -15/-28 points, sign accuracy -16 points when anonymised in train and test |
| Skeleton anonymiser (2111.15129) | adversarial U-Net/ResNet, labelled subjects | NTU-60 re-ID 80.95 -> 4.20 with action 95.10 -> 91.75 |
| SignSeek cross-signer contrast (2609.03695) | gloss-label positives across signers | frozen ASLC R@1 48.40 (GAC alone) -> 57.66 |
| SSPS-like / CAP-like | cross-nuisance positives | see §5 (not sign) |

Gradient reversal gives ~1.7 WER points; label-supervised cross-signer contrast gives ~20-30 retrieval points. [I] The label-free method should therefore target cross-signer *positives*, not only adversarial removal.

### 6.4 Skeleton dictionary retrieval / metric-learning results (exact numbers, protocols) [V unless noted]

All frozen, train-gallery cosine retrieval, ASL Citizen test queries, from SignSeek Tab. 2 and SignRep Tab. 4 (A §3.2 gives the full table; reproduced only where needed for the gap argument):
- Pooled pose features, no learning: hand joint angles 8.81 R@1, hand keypoints 7.96, all keypoints 0.98.
- SignCLIP (text-aligned pose, 456K Spreadthesign) 4.00 / DCG 21.80.
- MASA (146K ISLR clips, label-free) 27.96 / DCG 49.31; WLASL 14.38 (WLASL seen in pretraining, transductive).
- SignRep 49.95 / DCG 71.21 (RGB, label-free + 3D-pose teacher).
- SignSeek 57.66 / DCG 77.21 (pose, labelled ISLR corpora from 5 sources; ASL Citizen unseen).
- Zero-shot BSL SignBank (SignSeek Tab. 3): SignRep 4.30 R@1, MASA 6.80, SignCLIP 1.02, I3D 0.55.
- CiCo (arXiv 2303.12793) is text-supervised and evaluated on WLASL/MSASL, not on ASL Citizen frozen retrieval; I found no ASL Citizen frozen retrieval number for it (not listed in SignSeek Tab. 2).
- No published number exists for within-signer vs across-signer retrieval on ASL Citizen for any model; the speech ABX gap (Table in §1.2) is the template.

### 6.5 Local calibration [L, not literature, unverified]

`docs/proposal17/diagnostics/results/nuisance_decomp.json`: within-gloss variance of embeddings explained by signer 17.5% (eta^2), duration decile 2.9%, fps 0.5%, resolution 0.03%; signer is the dominant *measured* nuisance (technical metadata are negligible). `geo_headroom.log`: raw R@1 22.7; centred 23.1; whitened 29.4; LEACE(signer) 29.7; whitened + LEACE 31.7; signer offset oracle 28.4; whiten + oracle offset 31.7; supervised LDA 47.6; R@1 grows with number of gallery signers (4: 9.8, 16: 18.9, 35: 22.7). If these hold, a *label-free* whitening alone recovers about the same as the signer-oracle offset, i.e. much of the "signer problem" in this model is global anisotropy/shared-variance structure, and the residual to 47.6 is *gloss-discriminative reweighting* that no unsupervised linear step supplies. [I]

---

## 7. Cross-cutting lessons

1. **Where invariance is imposed matters** (ContentVec N1 vs N4/N8; Komatsu removing the pooled token): impose at an intermediate layer and read out from the invariant side; do not give the pooled readout an objective that rewards nuisance.
2. **Clean teacher, perturbed student** is the lowest-cost transplantable idea (Komatsu: pooled linear SID 67.2 -> 26.6; no VC network).
3. **Perturbations must stay realistic** (ContentVec: bad VC worse than none; Komatsu: restricted to plausible conversions; AUTSL: crude anonymisation costs 16 accuracy points).
4. **Linear guarantees != neighbourhood guarantees** (Gonen & Goldberg; INLP MLP 85%; adversary vs attacker 49.0/56.0).
5. **Held-out nuisance subspaces need many nuisance instances** (250 speakers, overfit beyond ~50 dims); we have O(40) signers.
6. **A short, initialised, cross-instance tuning stage fixes kNN more than any objective change** (MAE-CT, STARS) provided the starting kNN is decent; it is the last step, not the first.
7. **Report the nuisance gap, not only R@1:** within-vs-across-signer retrieval, signer-ID MLP accuracy, same-signer NN rate, kNN-vs-linear gap, hubness skewness.

---

## 8. Candidate strategies ranked for our setting (max 8)

Ranking = expected value per unit cost, discounted by risk; effect sizes are my estimates [I] anchored to the cited numbers, not predictions from the literature.

**1. Retrieval-time geometry stack (centre, shrinkage whitening, top-PC removal, CSLS/hubness correction, top-m or LSE class scoring).**
- Mechanism: removes dominant shared directions and gallery-density heterogeneity (ABTT, MAE-200, hubness papers, INLP-SimLex).
- Expected effect on frozen R@1: +3 to +8 absolute on the current JEPA embeddings [I; [L] shows +6.7 for whitening]; effect will shrink for any better model.
- Risk: low; whitening can amplify noise dims (Liu: rescaling hurt; use shrinkage and k cap); fit statistics must be label-free (pretraining corpus, not the gallery labels) and the transductive/inductive choice must be declared.
- Cost: S (CPU, minutes).
- Minimal test: §4.3 grid on saved val embeddings; signer-clustered bootstrap CI; report same-signer-NN rate and hubness skewness before/after.

**2. Label-free pseudo-signer subspace erasure (PCA/LEACE on pseudo-signer or clip-static-descriptor clusters; signer-ID LEACE/collapse as declared upper bound).**
- Mechanism: Liu et al. speaker-subspace collapse generalising to unseen speakers; INLP/LEACE for the linear part.
- Expected effect: +2 to +6 R@1 beyond Strategy 1 [I]; upper bound with true signer IDs about +2 over whitening in [L] (31.7 vs 29.4).
- Risk: medium; pseudo-signer clusters may absorb gloss variance (erasing content); with O(40) signers the subspace overfits after ~10-20 dims; neighbourhoods may stay same-signer (Gonen & Goldberg).
- Cost: S.
- Minimal test: §3.5(b)/(c) with signer-grouped CV; falsified if signer-ID MLP stays > 80% and same-signer NN rate unchanged.

**3. Asymmetric morphology perturbation + style-conditioned predictor (clean/canonicalised EMA teacher, randomly retargeted student, intermediate-layer invariance term, style code from rest-frame/static pose statistics).**
- Mechanism: ContentVec DS+Cond, Spin, Komatsu BYOL; analytic 2D retarget of per-bone, hand-scale, aspect factors.
- Expected effect: signer-ID linear accuracy down 10-25 points, R@1 +2 to +5 [I]; AUTSL suggests body proportions are ~15-28 of 80 points of pose identity, ContentVec gives 12% rel. ABX gain.
- Risk: medium-high; 2D foreshortening makes bone retargeting approximate; AUTSL shows content loss from careless retargeting (-16 points); behavioural identity untouched; gain may be inside noise.
- Cost: M (new augmentation + one pretraining run per arm, multi-seed).
- Minimal test: §1.7 morphology-sensitivity probe and §2.5 descriptor regression first (CPU); train only if the probe shows morph-sensitivity larger than cross-signer same-gloss distance.

**4. Short cross-signer NN-contrastive tuning (MAE-CT/STARS recipe with same-pseudo-signer exclusion and head initialisation).**
- Mechanism: cross-instance positives create class-level metric structure (kNN 60.6 -> 77.4; 63.1 -> 79.9); signer exclusion = CAP/SSPS principle.
- Expected effect: largest potential (+5 to +15 R@1) *conditional* on cross-signer NN precision >= 40% after Strategies 1-3; otherwise negative.
- Risk: high (false positives; STARS-from-scratch 37.6 kNN; confident neighbours are 82% same-signer).
- Cost: M-L (20-epoch tuning of top half of the encoder plus mining queue).
- Minimal test: §5.4 positive-precision measurement on fixed embeddings; go/no-go at mutual cross-signer NN same-gloss rate >= 40%.

**5. Discrete/clustered teacher targets and teacher-layer choice by signer decodability.**
- Mechanism: ContentVec picks teacher layer 7 over 6 for lower speaker info and quantises to 100 clusters; DinoSR codebook has the smallest cross-vs-same-speaker ABX gap among non-Spin models; Komatsu: target layer matters.
- Expected effect: signer-ID down, R@1 +1 to +4 [I, weakest evidence: no isolated ablation].
- Risk: medium (codebook collapse; JEPA loses continuous targets).
- Cost: M.
- Minimal test: for existing EMA-teacher layers/pooled latents, compute signer-ID MLP accuracy vs gloss-probe accuracy per layer; pick the layer minimising the ratio; then a k-means (K=100-500) target variant for one pilot.

**6. Pseudo-signer-aware sampling (signer-matched hard negatives / cross-signer positives inside any contrastive or variance term).**
- Mechanism: CPC within-speaker negatives [S], subject-specific negatives (EEG subject-ID 88.6 -> 68.4), SSPS positives.
- Expected effect: +1 to +3 [I].
- Risk: pseudo-signer errors; JEPA has no negatives, so applies only to auxiliary VICReg/contrastive terms.
- Cost: M.
- Minimal test: batch composition ablation with pseudo-signer clusters on a short pilot; measure signer-ID MLP accuracy and R@1.

**7. Rest-frame/anchor-pose subtraction at the input (frame-0/rest-pose offset, AUTSL-style, applied to body+face only).**
- Mechanism: 2410.13675 appearance removal; cheap and label-free.
- Expected effect: signer-ID down 15-28 points on AUTSL; R@1 effect uncertain, risk of -5 to -15 content loss (AUTSL accuracy 80.97 -> 64.48).
- Risk: high (content is body-relative; assumes rest first frame).
- Cost: S.
- Minimal test: replace input by (P - P_rest + P_median_rest) on 2k val clips at inference only and compare embeddings/R@1; abandon if R@1 drops.

**8. Adversarial signer removal (GRL/discriminator) with pseudo-signers; also used as diagnostic upper bound with true IDs.**
- Mechanism: SRM (-1.7 WER), SignRep adv (+0.8 DCG), subject-invariant EEG.
- Expected effect: +0 to +1.5 R@1 [I]; high variance.
- Risk: high (post-hoc attackers recover; unstable; metadata signer IDs are not allowed in the method).
- Cost: M.
- Minimal test: train a gradient-reversal head with *true* signer IDs (diagnostic only) for one pilot; fresh MLP probe afterwards; if gain < 1 R@1 and same-signer NN unchanged, drop; never include in the label-free method.

**Sequencing.** 1 -> 2 (CPU, same day) establish the post-hoc ceiling; 3's probes (CPU) decide whether to spend GPU on it; 5's diagnostic is also CPU; 4 only after the precision gate; 6-8 are optional. Every training arm must beat the "embedding + Strategy 1 + 2" baseline, not the raw baseline.

---

## 9. Source index (arXiv ids verified by fetching the HTML title; text in `scratchpad/q2/<id>.txt`)

| id | short | used for | tag |
|---|---|---|---|
| 2204.09224 | ContentVec | §1.1 tables | V |
| 2305.11072 | Spin | §1.2 | V |
| 2311.09117 | R-Spin | §1.3 | V |
| 2409.10103 | Komatsu BYOL speaker-disentangled HuBERT | §1.4 | V |
| 2310.10803 | SD-HuBERT | context (via 2409.10103) | V title |
| 2110.13900 | WavLM | §1.5 | V |
| 2305.10005 | DinoSR | §1.5 | V |
| 2202.03555 / 2106.07447 | data2vec / HuBERT | §1.5 (no speaker analysis found) | V |
| 2110.14513 | NANSY | §1.6 | V title |
| 2005.05732 / 1808.07371 | skeleton-aware retargeting / EDN | §2.3 | V |
| 2410.13675 | pose appearance transfer | §2.1 | V |
| 2111.15129 | skeleton anonymisation | §2.1 | V |
| 2110.06161 | SAM-SLR | §2.2, 6.3 | V |
| 2101.11228 / 2008.09435 / 2107.01903 | gait/skeleton re-ID | §2.1 | V title / S |
| 2109.08090 | DisUnknown | §2.3 | V title |
| 2306.03819 / 2004.07667 / 2201.12091 | LEACE / INLP / RLACE | §3.1 | V |
| 1903.03862 | Lipstick on a pig | §3.1 | V |
| 2305.12464 | speaker-subspace collapse | §3.2 | V |
| 2406.09200 | orthogonality and isotropy | §3.2, 4.2 | V |
| 2108.00917 | CPC speaker information | §3.2 (numbers in C) | V |
| 1808.06640 | adversarial removal | §3.3 | V |
| 2007.04871 | subject-aware CL | §3.3, 5.2 | V |
| 2212.13023 | signer removal module | §3.3, 6.3 | V |
| 1702.01417 / 2103.15316 | all-but-the-top / BERT-whitening | §3.4 | V (whitening: qualitative) |
| 2311.18364 / 1710.04087 | hubness SBERT / CSLS | §3.4, 4.2 | V |
| 2401.00463 | local representations, high-variance features | §3.4, 4.2 | V |
| 2304.10520 | MAE-CT | §4.1, 5.1 | V |
| 2407.10935 | STARS | §4.1, 5.1 | V |
| 2104.14548 / 2204.07141 / 2104.14294 | NNCLR / MSN / DINO | §5.1, 4.3 | V |
| 2505.14561 | SSPS | §5.2 | V |
| 2012.10674 | CAP | §5.2 | V |
| 2509.04193 / 2311.00562 | DUDE / MNN | §5.2 | V title / S |
| DN2A (CVPR 2023), JOKR (CGF 2022), BISINDO bone-length (Jurnal Sist. Inf.) | | §2.3, 2.2, 5.2 | S |
| 2304.05934 | ASL Citizen | §6.1 | V |
| 2503.08529 | SignRep | §6.2 | V |
| 2405.20666 / 2609.03695 / 2609.01886 | MASA / SignSeek / SignMatch | §5.1, 6.4 | V (SignMatch not used for numbers) |

**Known gaps in this review.** (1) No isolated ablation of continuous-vs-discrete targets on nuisance (Strategy 5 evidence is indirect). (2) No published within-vs-across-signer retrieval on ASL Citizen. (3) The kernel/nonlinear erasure literature (kernelised concept erasure) was not fetched; I only have LEACE's and INLP's statements about it. (4) I did not verify NANSY's own numbers (only its role). (5) [L] numbers were produced by another process and not reproduced.
