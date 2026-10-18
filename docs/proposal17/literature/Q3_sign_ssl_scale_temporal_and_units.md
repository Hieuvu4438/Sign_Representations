# Q3. Sign SSL: data scale, temporal-predictive objectives, discrete units, segment structure, phonological structure, frozen pose features

Date: 2026-10-17. Scope: gaps left by `docs/research/literature/A_signrep_and_sign_ssl.md` (A), `docs/proposal12/literature/L4_sign_ssl.md` (L4), `docs/proposal16/literature/P3_sign_domain_signals.md` (P3), `D1_server_dataset_audit.md` (D1). Path-signature / Path-JEPA excluded. Target model: skeleton (RTMW-x COCO-WholeBody 2D) JEPA, PLctx (7 phonological channels x 32 bins, 40k ASL Citizen train clips, frozen-retrieval R@1 test 16.7 / val 22.7, plateau 160->200 ep).

**Tags.** [V] number grepped in raw arXiv HTML fetched this session (`scratchpad/q3/raw/<id>.txt`, table caption checked). [V-A] verified in A/L4/P3/D1 and not re-fetched. [S] search snippet / metadata only. [I] my inference (flagged). [P] project-internal measurement quoted from the task brief or A. "Label-free" means no gloss/text/phonology labels anywhere in training.

Every arXiv id below was fetched and the title matched. Two ids I first recalled from memory were wrong (2409.07336 is a quantum-computing paper, 2310.05861 a VQA paper); they were discarded. S-JEPA has no arXiv preprint that I could find (ECCV 2024, DOI 10.1007/978-3-031-73411-3_21 [S]).

---

## 0. Bottom line (read this first)

1. **No paper has measured a data-scale curve for FROZEN sign-retrieval quality.** All published scale curves are fine-tuned or translation metrics (sec 1.1). SignRep, the only strong frozen result, has no data ablation [V: no subset/scale text in the arXiv v1 HTML].
2. **At fixed data, the objective moves frozen retrieval by two orders of magnitude; at fixed objective, 10x data moves downstream metrics by single digits.** SignRep Table 4, same YouTube-SL-25 pretraining: Hiera pixel-MAE ASL Citizen R@1 **0.39**, SignRep **37.47** (avg) / **49.95** (activity-weighted); raw mean-pooled hand joint angles with no learning **8.81** [V]. 10x data: SHuBERT BLEURT 29.1->31.4, V-JEPA 2 +1.0 average accuracy for 11x videos [V].
3. **Target/input changes with direct evidence for movement**: velocity/motion-residual targets give +10.1 linear on NTU-60 (MAMP, 84.9 vs 74.8) and +1.8 to +4.2 on WLASL2000 fine-tune (MASA; +5.4 combined with motion-aware masking) [V]. SignRep's wrist-relative angle/distance targets beat keypoint targets by 12-14 DCG [V-A]. Multi-span latent targets (5/10/30 frames + global) lift **training-free retrieval** on NTU-60 x-sub from 62.0 (MAMP) to 70.9 [V].
4. **Masked infilling on pose is a copy/interpolation task.** OpenHands reports that pose MAE loss falls by interpolation and gives no fine-tune gain (91.3 vs 91.2 no-pretrain) while dense predictive coding gives 94.7; Human-JEPA (Aug 2026) reaches the same diagnosis for video ("temporally local copy task", block masks cost 4.8-5 points vs pure past-to-future split) [V]. But forward-only masking from scratch is *worse* in V-JEPA (SSv2 49.8 / 63.6 vs 67.4 bidirectional multi-block) [V]. Net: forecasting is a credible *auxiliary*, not a proven replacement.
5. **Readout is a first-class lever and the cheapest to test.** SignRep activity weighting: 37.47 -> 49.95 = **+12.48** R@1 [V, verified]. V-JEPA: average pooling vs attentive pooling on frozen features **50.1 vs 66.2** (SSv2) and 56.7 vs 73.7 (K400) [V]. SHuBERT: learned weighted sum over layers 29.3 vs last layer 21.4 BLEU-1 [V]. HuBERT: last layers degrade when targets are poor [V].
6. **Discrete units**: raw-feature k-means is a weak teacher (HuBERT: MFCC PNMI 0.276 -> WER 18.40; iteration-1 layer-6 PNMI 0.637 -> 11.91; iteration-2 layer-9 0.704 -> 10.75) [V]. SHuBERT performs one k-means round on DINOv2 features, no refinement [V]. No sign paper reports unit purity against ASL-LEX for unsupervised units.
7. **Label-free structure does not help recognition; weak phonological labels do** (Kezar VQ-ASL: parameter disentanglement .381 -> .372 OOV MRR, phonology-supervised .435) [V]. Aux phoneme prediction gives +8.7 top-1 (SL-GCN, WLASL) [V]. PhonSSM (Apr 2026) reports large gains for explicit factorisation but with claims I flag as unreliable (sec 5.4).
8. **Single most valuable investment for frozen retrieval, per the evidence**: input/target/readout changes first (S-M cost, strongest controlled effects), a measured 2.6x data step (105 h, already on server) second as a slope test, new corpus acquisition (YouTube-ASL / SignVerse-2M) only if the slope is positive. See sec 7.

---

## 1. Q1. Data scale and domain for sign SSL

### 1.1 Scale curves that exist (none for frozen retrieval)

| Paper | Eval protocol | Pretraining data axis | Numbers | Objective / label use | Tag |
|---|---|---|---|---|---|
| BEST (2302.05075) | MSASL1000 P-I, fine-tune | % of downstream train sets (pose, ~1e4-1e5 clips) | 0%: 51.96; 25%: 53.67; 50%: 54.69; 75%: 56.06; 100%: 58.82 | masked unit modelling, label-free pretrain | [V] Table 8 |
| STC (2406.10501) | MSASL1000 P-I, fine-tune | same | 0%: 51.37; 25%: 59.06; 50%: 62.82; 75%: 64.57; 100%: 65.22 | contrastive, label-free | [V] Table VIII |
| MASA (2405.20666) | WLASL2000 | % of 4 ISLR train sets | curve only as a figure ("gradually increases") | motion-residual MAE + momentum contrast | [V] Fig. 2 text |
| Sigma (2509.21223) | WLASL2000 P-I fine-tune; CSL-Daily WER | % of YouTube-ASL (530k) + CSL-News (723k) | P-I 0%: 23.00; 25%: 50.68; 50%: 58.83; 75%: 62.46; 100%: 64.54. WER 72.34, 32.18, 29.78, 27.92, 25.26 | **text-supervised** | [V] Table 10a |
| Uni-Sign (2501.15187) | CSL-Daily dev | % of CSL-News (1,985 h) | CSLR WER 74.7 / 31.5 / 31.5 / 28.8 / 28.2 (0/25/50/75/100%); SLT BLEU4 3.75 / 20.68 / 21.85 / 24.74 / 25.27 | text-supervised generative | [V] Table 8 |
| SHuBERT (2411.16765) | How2Sign SLT, frozen encoder, 100K steps | 98 h vs 984 h | BLEU 0.7 -> 2.2; BLEURT 29.1 -> 31.4 | cluster prediction, label-free | [V] Table 5 |
| SignDino (2609.06296) | How2Sign SLT, frozen | 98/246/492/738/984 h | BLEU 13.5/14.8/16.0/16.5/16.8; BLEURT 46.1->50.4 | temporal DINO on frozen DINOv3 crops, label-free (single-group preprint, self-reported, smooth curve: treat with care) | [V] Table 14 |
| V-JEPA 2 (2506.09985) | 6-task frozen average | 2M -> 22M videos | **+1.0** average; model 300M->1B +1.5; curated vs uncurated YT-1B **+1.4** | latent prediction | [V] sec 2.2 |
| DPC (1909.04656) | UCF101 | UCF101 -> K400 | self-sup acc 53.6 -> 61.1; supervised UCF 60.6 -> 65.9 | future latent prediction | [V] Table 2 |
| SignRep (2503.08529) | -- | **none** | no subset/scale ablation | -- | [V] grep |

Observations.
- **Concave, mostly front-loaded.** BEST/STC: the first 25% gives 1.7 / 7.7 of the total 6.9 / 13.9 gain. Uni-Sign: 0->25% buys 43 WER points, 25->100% buys 3.3. V-JEPA 2: 11x data is +1.0. [V; shape is my reading]
- **The downstream-training-set pretraining in BEST/STC is tiny** (tens of thousands of clips), i.e. the same regime as our 40k clips. Their curves say the gain from 25% -> 100% of ~1e4-1e5 clips is +4-6 fine-tuned points. Whether that survives frozen retrieval is untested.
- **Frozen sign-quality curves from label-free pose models do not exist.** The only frozen scale-like comparison is pose SSL at ~146k isolated clips (MASA, ASLC R@1 27.96) vs our 40k clips (16.7). That pair is confounded by target (motion residual), architecture (GCN), objective (adds momentum contrast), and extractor. [I]

### 1.2 What the data/domain evidence says, item by item

**SignRep / YouTube-SL-25** [V Table 6, V-A]: pixel MAE init Kinetics 51.5 vs YT-SL 57.2 on WLASL2000 fine-tune (+5.7). That is domain in-language data, not frozen. No frozen data ablation. Confound: same Table 4 shows MAE on the same YT-SL data is useless frozen (R@1 0.39).

**SHuBERT** (ACL 2025, 2411.16765) [V]: 984 h ASL (YouTube-ASL with OpenASL clips removed, back-filled from YT-SL-25), 86M-parameter transformer, per-channel k-means (256 clusters, fitted on 10% of data). Scale ablation is on translation only (above). ISLR with rank-1 LoRA (0.17M trainable, so close to frozen): ASL Citizen R@1/5/10 **.65/.87/.91**, Sem-Lex .54/.74/.80, WLASL2000 P-I 60.90 / P-C 58.01. Layer matters: weighted sum over layers 29.3 BLEU-1 vs 21.4 last-layer vs 15.3 no-SHuBERT (Table 6).

**Uni-Sign** (2501.15187) [V]: pretraining on CSL-News (1,985 h CSL) for CSL, YouTube-ASL for ASL. Pose-only encoder, MSASL1000 P-I: 43.45 from scratch (Table 13, all sub-poses) vs 56.92 with only the pose temporal encoder features after pretraining and mean-pool+CE head vs 77.88 with the unified paradigm (Table 7). WLASL2000 63.13 pose-only (Table 3). **Contamination caveat** [V-A D1, V SHuBERT fn.]: 56% of the local MS-ASL test clips come from videos listed in YouTube-ASL; SHuBERT explicitly omits MSASL for this reason. The MSASL gains are therefore inflated and WLASL (0 overlap per D1) is the cleaner number.

**SSVP-SLT** (2402.09611) [V]: on How2Sign, pretraining on YouTube-ASL only vs YouTube-ASL+How2Sign gives 14.7 vs 14.6 BLEU, i.e. in-domain studio data adds nothing once the in-the-wild set is large and diverse. Clip length matters a lot: 16-frame Hiera vs 128-frame SignHiera at 800 epochs, YT+H2S: 10.4 vs 14.3 BLEU; H2S only: 5.0 vs 7.0 (Table 4). Cost: ~2 weeks on 64 A100s for the largest run.

**OpenHands** (2110.05877) [V]: 1,129 h unlabelled Indian SL as pose (MediaPipe), ST-GCN. DPC pretraining: INCLUDE 91.2 -> 94.7; at 3 samples/class 15.2 -> 35.42; **cross-lingual to WLASL2000: 21.4 -> 27.4**, at 3 samples/class 1.6 -> 2.78. So 1,100 h of *another language* helped ASL isolated recognition. No scale curve.

**SignCLIP** (2407.01264) [V]: pretraining on Spreadthesign, ~456k isolated dictionary clips in up to 44 languages, **one example per concept per language**. The paper itself warns of "overfitting the exact signs by particular signers". Frozen pose-embedding zero-shot on ASL Citizen **R@1 0.00**, 10-shot 0.05, frozen + linear 0.23, from scratch 0.39, fine-tune 0.46 (Table 5). Interpretation [I]: isolated, single-signer-per-concept data has no cross-signer variation to teach signer-invariance; scale of that kind of data is not the same as scale of continuous multi-signer data.

**Isolated vs continuous pretraining, transfer to isolated tasks** (the question as posed):
- Continuous -> isolated works for RGB frozen/near-frozen (SignRep 49.95; SHuBERT .65 LoRA) [V]. For pose: Uni-Sign and OpenHands fine-tune gains [V].
- Isolated-only pose SSL pretrained on the downstream train sets (SignBERT/BEST/SignBERT+/MASA/STC) gives strong fine-tune but weak frozen/linear: WLASL2000 linear P-I 18.24 / 21.19 / 24.50 / 32.35, MSASL1000 31.54 / 30.89 / 34.44 / 48.30 (STC Table V) [V]; SSL-SLR linear 20.46 on ASL Citizen-500 [V].
- Nobody has compared the two on the same encoder. [I]

**Is there any pose SSL at YT-SL-25 scale?** A claimed "nobody". Correction: OpenHands pretrained a pose ST-GCN on 1,129 h (fine-tune eval only, no frozen retrieval) [V]. SignVerse-2M (2605.01720) is a DWPose (COCO-WholeBody family) conversion of the full YouTube-SL-25: 3,207 h, ~2M clips, 55+ languages, 24 fps, per-video `poses.npz` with 133-keypoint scores [V paper; actual download availability and size unverified, S]. The extractor differs from RTMW-x (DWPose is RTMPose-distilled), so keypoint noise statistics will differ. [I]

### 1.3 Domain and extraction facts from D1 that bear on scale decisions [V-A D1]
- Local pose-ready unlabeled ASL, train splits only: ASLC 32 h + WLASL 9.5 h + MS-ASL 10.8 h + How2Sign 53 h = **~105 h**; FP16 ~80-joint cache ~4.5 GB.
- Hand quality: ASLC worst (left hand score <3 in 49.8% of frames), How2Sign cleanest (0.5%). The SSL training set (ASLC train) is therefore the noisiest hand data in the pool; How2Sign adds clean hands but a different domain (long studio sentences, p95 15 s, no signer IDs).
- YouTube-ASL: only 11,095 IDs, no video; 94 GB free disk; 3,000 h pose extraction ~76 GPU-days at 41 fps aggregate. 111 videos overlap MS-ASL and must be blacklisted.

### 1.4 "Scale is the main lever" vs "objective is the lever": evidence ledger

| Claim | Controlled comparison | Number | Setting |
|---|---|---|---|
| Objective matters at fixed data | SignRep T4: HieraMAE-YTSL vs SignRep (same 3,000 h) | R@1 0.39 vs 37.47 / 49.95 | frozen, ASLC [V] |
| Learning can be worse than no learning | raw hand joint angles, mean-pooled, vs MAE | 8.81 vs 0.39 | frozen [V] |
| Target matters at fixed data (pose) | MAMP T7: joint->joint vs joint->motion | NTU-60 74.8 -> 84.9; NTU-120 72.5 -> 78.6 | linear [V] |
| Target matters (sign pose) | MASA T6: random+joint vs motion mask+motion target | WLASL2000 P-I 43.62 -> 49.06 | fine-tune [V] |
| Target matters (SignRep) | T5: kpt 32.9 vs dist 47.2 vs angle 45.1 | WLASL DCG | frozen retrieval [V-A] |
| Masking region matters | BEST T7: body-only masking 47.72 < no pretraining 51.96 < hand-only 54.60 < hand+body 58.82 | MSASL1000 P-I | fine-tune [V] |
| 10x data, same objective | SHuBERT 98 h -> 984 h | BLEU 0.7 -> 2.2, BLEURT +2.3 | translation, frozen [V] |
| 11x data, same objective | V-JEPA 2 | +1.0 avg | frozen probe [V] |
| Data axis dominates when labels are text | Sigma 0% -> 25% data | 23.00 -> 50.68 | text-supervised fine-tune [V] |

Reading [I]: scale dominates only when a *semantic* signal (text) is present, as in Sigma and Uni-Sign (0 -> 25% is a 27-point jump, 25 -> 100% is 14 more). For label-free objectives the published scale effect is much smaller than the objective/target effect. The evidence does not say scale is useless: BEST/STC show +7 to +14 over 0-100% on 1e4-1e5 clips, and no frozen curve exists to rule out a large effect.

**Transfer verdict for our setting.** (a) Our plateau (160->200 epochs: +0.2 val, -0.05 test) is an optimisation plateau on 40k clips; it does not discriminate data-limited from objective-limited. (b) Scale is worth buying only after readout/target diagnostics because the diagnostic cost is 1-3 GPU-days and the scale cost is weeks. (c) If scaling, diversity beats domain match (SSVP-SLT YT-only = YT+H2S), and cross-lingual pose is acceptable (OpenHands, Uni-Sign).

**Minimal test (M, ~3 pretraining runs).** Same PLctx recipe at 10 / 25 / 50 / 100% of the 105 h corpus (ASLC train subsampled by signer to keep it matched), same epochs-by-steps budget, frozen ASLC val R@1 with 3 readouts (mean, motion-weighted, bins2). Pre-registered readout: slope per doubling. Decision rule: slope >= +1.5 R@1/doubling at 25 -> 100% => acquire more; slope < +0.5 => stop scaling, go to targets. Exclude WLASL/MS-ASL test videos and the 112 WLASL-test clips whose source video is in MS-ASL.

---

## 2. Q2. Temporal-predictive / world-model objectives for motion

### 2.1 Direct evidence in sign

**OpenHands DPC** (2110.05877) [V Table 5, Fig. 4 text].
- ST-GCN on INCLUDE (263 ISL signs): no pretraining 91.2, masked regression (40% frames) 91.3, contrastive (shear/scale/rotation augment) 90.8, **predictive coding 94.7**.
- Mechanism claimed by the authors: masked pretraining "was able to approximate interpolation for the masked frames"; error vs true keypoints diverges as the masked span grows (Figure 4), so no longer-range structure is learned. Masking 20/30/40% and spans 2-10 give the same null.
- DPC predicts non-overlapping window embeddings with an InfoNCE loss against distractors; "predict coarse windows, not frames".
- Setting caveats: fine-tune, ST-GCN, MediaPipe, ISL; no frozen protocol; the 3.5-point gain is on a saturated benchmark. Few-shot gains are much larger (3/class: 15.2 -> 35.42).
- Direction-quadrant classification (a "movement" pretext) failed to converge because per-frame jitter makes quadrant labels near-random and >50% of motion vectors are static. [V] This is relevant to us: **per-frame velocity targets from 2D keypoint jitter are noisy; use temporal differences at intervals** (MASA k=3, below).

**MASA** (2405.20666) [V Tables VI, VII, VIII]. WLASL2000 P-I fine-tune:

| mask | target | P-I |
|---|---|---|
| random | joint | 43.62 |
| motion-aware | joint | 47.22 |
| random | motion | 47.78 |
| motion-aware | motion | **49.06** |

Interval k for motion residual M_i = x_{i+k} - x_i: k=1 47.64, **k=3 49.06**, k=5 47.71, k=7 45.03. Components: MA only 47.64, semantic-alignment (momentum contrast) only 43.61, both 49.06. Mask ratio 90% best. Interpretation: the motion target adds +4.2 with random masking (43.62 -> 47.78) and +1.8 with motion-aware masking (47.22 -> 49.06); the mask rule adds +3.6 with a joint target and +1.3 with a motion target; the interval is a tuned scale, too small = jitter, too large = hard. MASA input is 49 joints (7 body + hands) from MMPose 2D, confidence-thresholded [V]; the best interval is 3 frames.

**SignSeek** (2609.03695, Sep 2026) [V Table 7, ASL Citizen frozen R@1]: GAC (gloss contrastive) alone 48.40; +MAC 54.15; GAC+MAP 55.93; MAC+MAP 55.12; all three **57.66**. MAP is latent masked prediction of the most salient articulator (EMA-style cosine loss). The latent-prediction head adds **+7.5 R@1 over GAC alone** on frozen retrieval, but all three terms use gloss labels across signers (MAC aligns same-gloss references). [V; interpretation: label-supervised, so cannot be read as label-free evidence]

### 2.2 Skeleton action SSL (general motion): targets and retrieval

NTU-60 x-sub / x-view, linear, from GFP (2509.03609) Table 1 [V]:

| Method | Target | x-sub | x-view | Training cost |
|---|---|---|---|---|
| SkeletonMAE | joint coordinates | 74.8 | 77.7 | 20h27m |
| MAMP (2308.07092) | motion (temporal difference) | 84.9 | 89.1 | 20h27m |
| S-JEPA (ECCV 2024) | patch-level latent | 85.3 | 89.8 | 90h57m |
| GFP | hierarchical latent (5/10/30-frame spans + global) | 85.9 | 92.0 | 3h14m |

MAMP T7 [V]: input/target joint->joint 74.8 / 72.5 (NTU-60 / NTU-120), joint->motion **84.9 / 78.6**, motion->joint 76.5 / 71.0, motion->motion 75.9 / 70.5. Motion-aware vs random masking adds +1.2 / +1.3 (T8). **Predicting dynamics from static input is the effective direction; feeding motion as input while predicting joints or motion is not.** GFP T4/T6: the target generator should take *temporal differences* of the unmasked input (motion features beat raw joints and masked joints) [V].

**Training-free retrieval** (GFP Table 3, NTU-60 x-sub / x-view) [V]: LongT GAN 39.1/48.1, P&C (motion-prediction seq2seq autoencoder; Predict & Cluster, 1911.12409) 50.7/76.3, HiCo 68.3/84.8, UmURL 71.3/88.3, **MAMP 62.0/70.0**, **GFP 70.9/87.1**. Linear (84.9) vs retrieval (62.0) for MAMP is a 23-point gap: motion-target MAE is linearly strong but metrically weak, the same pattern as our frozen retrieval. Early forecasting AEs (LongT GAN 39.1) are poor, so "predict the future skeleton" as such is not the effective part.

### 2.3 Predictive coding family (non-sign): horizon, negatives, identity leakage

**CPC** (1807.03748) [V Tables 1-2], LibriSpeech phones, linear on 256-d context:
- Number of predicted steps: 2: **28.5**, 4: 57.6, 8: 63.6, **12: 64.6**, 16: 63.8. Multi-step prediction is the whole effect.
- Negative sampling: mixed speakers 64.6, same speaker 65.5, excluding current sequence 57.3 (mixed) / 64.6 (same-speaker), current sequence only 65.2.
- Speaker ID (251) by the same linear probe: **97.4%** (MFCC 17.6, supervised 98.5; phones: CPC 64.6, MFCC 39.7, supervised 74.6; Table 1). A predictive objective learns speaker identity as readily as content, i.e. 94% signer decodability in our probes is not a pathology of JEPA.
- Hidden-layer MLP on the same features: 64.6 -> 72.5, "not all information is linearly accessible".

**DPC** (1909.04656) [V Tables 1, 3]: dense (spatial map) prediction essential: removing the map (CPC-like vector) could not be trained at all; removing sequential prediction 60.6 -> 56.9 (UCF supervised eval). Predicting further (5pred3 -> 4pred4 with curriculum) 65.9 -> 68.2 supervised acc, though SSL accuracy of the harder task falls (61.1 -> 48.3).

**TCLR** (2101.07974) [V Table 4] (UCF101 pretrain, R3D-18): instance-contrastive only linear 54.58 / retrieval R@1 40.76; +local-local temporal contrast 62.70 / 51.10; +global-local 64.55 / 47.32; all three **69.91 / 56.17**. Purely temporal losses without instance discrimination: linear 20.61-23.39, R@1 12.83-14.17 (random init 17.15 / 8.21). Temporal objectives need an instance-level anchor to give frozen retrieval quality.

**SignDino** (2609.06296) [V Table 4, 18]: temporal-axis DINO on frozen crop features; shrinking the teacher to short windows only (U{10..32}) costs -2.1 BLEU and -0.05 ASLC R@1 (0.704 -> 0.65): long global context as supervisory signal dominates. Randomised crop lengths beat fixed pairs by 0.4-0.6 BLEU at iso-FLOP. (Fine-tuned R@1, not frozen.)

### 2.4 Causal vs bidirectional masking in V-JEPA-family models

- **V-JEPA** (2404.08471) [V Table 4]: ViT-L, frozen attentive probe, K400 / SSv2 / IN1K: random-tube[0.9] 51.5 / 46.4 / 55.6; causal multi-block[6 context frames] 61.3 / **49.8** / 66.9; causal multi-block[12] 71.9 / **63.6** / 72.2; **multi-block (bidirectional)** 72.9 / **67.4** / 72.8. Forward-only context hurts the motion benchmark most.
- Feature vs pixel targets [V Table 1]: SSv2 **66.0 (pixel) vs 66.2 (features)**, K400 68.6 vs 73.7, IN1K 73.3 vs 74.8. Latent targets gave no motion gain by themselves; the motion advantage over pixel baselines (VideoMAE SSv2 attentive 61.2 vs V-JEPA 69.5, linear 41.3 vs 50.1, Table 12) comes with the multi-block mask geometry. (Different recipes/data, so not a clean ablation.)
- **Human-JEPA** (2608.21160, Aug 2026) [V Tables 1, 2, 4]: from a V-JEPA 2.1-L init, replacing block masks by a pure past-to-future split recovers frozen NTU-120 action (70.4 vs base 70.8) while anchored block-mask adaptation pays -5.2 (65.6) and gains/loses ReID +2.7 mAP vs -17. Pilot-horizon swap: +8.6 NTU points. Authors' diagnosis: block inpainting "spends the adaptation budget on a temporally local copy task". Another finding: "any pose input makes it worse... the skeleton supplies most of the answer" for person-level masked latent targets, i.e. a latent target that is a function of the visible skeleton leaks. **Regime is continued pretraining of RGB JEPA, not from-scratch skeleton.**
- LeVJEPA (2608.27395) [V Table 2]: block-causal vs bidirectional encoder, ImageNet frozen attentive probe 51.2 vs 50.7: causality is free for appearance accuracy. P-JEPA (2606.23256) [V Table 2], Assembly101 linear probe: token-causal 35.6, bidirectional 37.6, clip-causal (bidirectional within a clip, causal across clips) **40.1**. V-JEPA 2-AC conditions a block-causal predictor on a *frozen* encoder [V]; it does not train the encoder causally.
- Verdict [I]: no from-scratch evidence that causal masking encodes motion better; strict token-causal is the weakest in the one place it was ablated. Within-sign bidirectional context plus a future-block component is the safe design.

### 2.5 Slow-feature / temporal-coherence

No sign or pose paper reports slow-feature or temporal-coherence SSL with frozen eval. TCLR is the nearest (above). [I] Slowness favours quantities that change little over a sign (handshape in a hold, signer anatomy, camera pose) and suppresses movement by construction; it is more likely to improve handshape and signer-identity encoding than movement. Not evidence-supported for our movement failure.

### 2.6 Transfer verdict and minimal tests (Q2)

- Strongest transferable evidence is on **targets**: motion-difference targets at an interval (MASA/MAMP) and multi-span latent targets (GFP). These are cheap and aimed at the failing probe (movement R2 < 0). Verdict: transfer likely.
- Forecasting/DPC-style: sign evidence = one fine-tune ablation on ISL (OpenHands); video evidence is regime-dependent. Verdict: transfer plausible as an *added* future-block prediction head with multi-horizon (CPC: 2 steps 28.5 vs 12 steps 64.6), not as replacement of masked prediction.
- Minimal test A (S): add motion-difference channel targets (k in {2,4,8} bins of the 32) to the teacher target; report movement R2 / ASL-LEX path-movement linear probe, frozen R@1. Minimal test B (S-M): mask mixture 70% block-mask + 30% "last 25% of bins masked" at fixed compute; report the same. Success if movement probe moves from R2 < 0 to > 0.2 and R@1 val >= +2.

---

## 3. Q3. Discrete sub-sign units (HuBERT-style targets, tokenizers)

### 3.1 HuBERT mechanics that matter (speech, the reference result) [V, 2106.07447]
- Iteration 1: 100-cluster k-means on 39-d MFCC; iteration 2: 500 clusters on iteration-1 layer-6 features; large models re-cluster layer 9 of iteration 2.
- **Purity against phones is measured explicitly**: phone purity, cluster purity, phone-normalised mutual information (PNMI), using forced-aligned phones. MFCC at K=100: PNMI 0.255; K=500: 0.287 (Fig. 2 text).
- Table V (K=500, dev-other WER at loss weight alpha=1.0 on masked frames, alpha=0.5, alpha=0): MFCC PNMI 0.276, WER 18.40 / 33.42 / 97.66; iteration-1 layer-6 PNMI **0.637**, WER 11.91 / 13.47 / 23.29; iteration-2 layer-9 **0.704**, 10.75 / 11.59 / 13.79; supervised chenone targets 0.809, 10.38. Better teacher => lower WER monotonically, and **predicting masked frames only (alpha=1) is critical** (alpha=0 WER 94-98 with MFCC targets).
- Layer choice matters: iteration-1 features are best around layer 6 and "the last few layers degrade dramatically", because they learn to mimic poor labels (Fig. 2).
- Ensembles of k-means with different K, and product k-means, help (WER 17.81 -> 16.73).
- K-means stability: fit data 1/10/100 h changes PNMI by at most 0.012 (Table IV text).

### 3.2 Sign units

**SHuBERT** (2411.16765) [V]. Four streams (face, left hand, right hand, 14-d body pose), DINOv2 features for crops, **one round** of k-means (256 clusters, 10% of data, per channel), cross-entropy on masked positions. Unit analysis is qualitative only (Figs. 5-8: pointing configurations, closed fist with thumb on top, transitional chest motion; some clusters mixed). No purity or NMI against ASL-LEX. **No iterative refinement**. Phonological feature recognition after full fine-tuning with 16 heads (Table 9, R@1 Sem-Lex / ASL Citizen): Major Location .8477/.9022, Minor Location .7130/.8000, Repeated Movement .8265/.8993, **Path Movement .7275/.7942**, Flexion .7264/.7773, Selected Fingers .7953/.8344, **Handshape .6293/.7080**, Wrist Twist .9058/.9300, average .7938/.8502. These are supervised-head numbers, not frozen. Masking ablation (translation): channel masking BLEU 2.6 / BLEURT 29.9, time 2.3 / 31.2, random 2.2 / 31.4 (Table 4): choice of masking type is a second-order effect.

**BEST** (2302.05075) [V Tables 6, 7, 9]. MSASL1000 P-I, fine-tune after pretraining on downstream train sets:
- Tokenizer (same K as codebook): k-means on raw pose 54.29, separate hand/body VQ 55.87, coupled hand+body VQ **58.82**.
- Objective: SignBERT-style random mask + regression 53.60; random mask + token prediction 53.95; hand-body unit mask + regression 56.90; unit mask + token prediction **58.82**. Token prediction adds +1.9 over regression at equal masking; the masking geometry adds +3.3.
- Hand-only masking 54.60, body-only 47.72 (below no-pretraining 51.96), both 58.82.
- Tokens are never audited against ASL-LEX parameters.

**VQ-ASL** (2509.04745) [V Table 3]. Sem-Lex unseen-gloss split. OOV ISR MRR: baseline VQ-VAE .381, parameter disentanglement (PD, label-free) **.372**, phonological-label supervision (PSS) .435, full .452. Phonological-feature probe MRR .488 / .475 / .565 / .583. Reconstruction MSE .058 -> .046 (PD) -> .041. **Label-free structure improves reconstruction but not discrimination.** (Probe on frozen quantised codes Zq, trained on the OOV split; train/eval vocab protocol not fully specified, as noted in A.)

**3D-LEX** (2409.01901) [V Table 4]. k-means (k=50, ~ASL-LEX handshape count) on handshape from a *sensor glove* (3D, clean), used as an auxiliary label for SL-GCN on WLASL overlap with 1,000 classes: top-1 none 0.44, expert labels 0.48, **automatic cluster labels 0.49**. Input cleanliness determines whether raw-feature clusters work. With 2D MediaPipe landmarks the handshape task itself is hard: a dual-GNN supervised handshape classifier reaches 46% on 37 PopSign-ASL-LEX classes (MLP on raw landmarks 25%) (2509.18309) [V]. Consistent with our handshape R2 ~0.3. [I]

**Q-BridgeNet** (2607.11215) [V abstract/method]: iterative alternation of segmentation and residual VQ (shared base codebook 1,024 + language-specific 512) for text-supervised SLT; no sign-level purity analysis. Low relevance. **SignVQNet / SOKE (2309.12179, 2411.17799)** are generation tokenizers [S].

### 3.3 What discovered sign units correlate with and how it depends on input
- No paper measures unit-to-parameter purity (handshape/location/movement) for label-free sign units. SHuBERT's own clusters are described as mixing signer-independent content, pose, face tilt, and "transitional hand movements" (cluster 98) [V]. [I] Movement is likely under-represented in frame-wise clusters: a unit assignment per frame cannot carry path movement; only sequences of units can.
- Input cleanliness dominates: glove 3D clusters match expert labels (3D-LEX); DINOv2 crop embeddings give usable hand clusters (SHuBERT figs); raw 2D keypoint clusters are the BEST k-means baseline (-4.5 vs coupled VQ).
- Clustering quality improves with iteration and with using an intermediate layer, not with more k-means data (HuBERT).

### 3.4 Transfer verdict and minimal test (Q3)
Verdict [I]: a HuBERT-style round-2 on PLctx intermediate features is the only form of "units" with a track record of improving the teacher, and PNMI-style purity can be measured against ASL-LEX parameters on ASL Citizen. Expected benefit on frozen retrieval is uncertain (no frozen evidence in sign); cost M; the specific risk is that clusters will encode signer anatomy (our signer probe 94%) and handshape units on 2D keypoints will be impure.

Minimal test (M, 1 extra pretraining run): (1) k-means K in {256, 1024} per channel on layer-l features of the existing PLctx checkpoint, l in {mid, last}; compute **phone-purity analogue** against ASL-LEX handshape/location/path-movement (ASLC videos map directly to ASL-LEX entries), cluster purity, PNMI, and **signer PNMI** (should be low); (2) one round of masked-cluster prediction using the best layer as teacher. Gate: unit PNMI(handshape) >= 0.2 and signer PNMI <= 0.1 before spending the second run; report frozen R@1 for the new encoder.

---

## 4. Q4. Boundary / segment structure, activity weighting, augmentations

### 4.1 SignRep activity weighting: verified
Table 4 (stride 2, class-max cosine against train gallery) [V]: ASL Citizen **avg R@1 37.47 / DCG 61.40 / R@5 68.77** -> **weighted 49.95 / 71.21 / 80.09**: **+12.48 R@1** (+9.81 DCG). WLASL 26.16 -> 29.92 (+3.76); NMFs-CSL 58.48 -> 63.04 (+4.56). The weight is gamma_n = max(P_LH, P_RH) from an activity head (Eq. 11-12); the activity prior is defined by "hand below the middle of the stomach", with jitter control (appendix) [V, line 99/864]. So the +12.5 is a *readout* gain from down-weighting rest/hold-out-of-signing-space windows. Activity head ablation (training without it) is not reported [V-A]. Confound: weighting also down-weights unreliable windows.

### 4.2 Evidence on rest frames, boundaries, phases
- **SSL-SLR** (2509.05188) [V Fig. 4 text, Table 1]: linear accuracy is unchanged when permuting the first ~N/3 and last ~N/4 frames; beyond that it collapses. Positive pairs built by permuting only those regions beat classical rotation/blur/flip/translation by "over 6%" (500 classes) and "over 8%" (310 classes) in linear eval; ASL Citizen-500 linear 20.46 +/- 1.21. Datasets are GSL/LSFB/LSA/ASL, MediaPipe. So **the start and end thirds of isolated clips are largely non-discriminative preparation/retraction**.
- **Training Strategies for ISLR** (2412.11553, RGB MViTv2/I3D, Slovo/WLASL/AUTSL) [V]: random boundary shift (moving sign boundaries inward) is a standard component; slow-down speed aug removal -0.66 top-1; auxiliary boundary regression +0.52 vs IoU-balanced CE +0.05.
- **Linguistically Motivated Sign Language Segmentation** (2310.13960): BIO tagging of signs/phrases on pose; used for continuous signing [S]; pose segmentation for stroke onset/offset is also used by IlluSign (2504.10822) with manual correction for short isolated signs [S].
- No controlled "exclude rest bins at train time" ablation exists for pose SSL or frozen retrieval. [V by search; I did not find one in arXiv full-text queries]

### 4.3 Augmentations: what preserves lexical identity and what does not
| Aug | Evidence | Verdict |
|---|---|---|
| Horizontal flip (blind) | WLASL -1.3 top-1 with flip (RGB, 2412.11553) [V]; AUTSL +0.11. SignCLIP in-domain Spreadthesign R@1 0.37 -> 0.36 with p=0.2 flip [V Table 4] | Destroys identity for non-mirror-invariant signs. Do not flip blindly. |
| Handedness canonicalisation (flip to right-handed at test time / dominant-hand mirror) | SignCLIP E8 test-time flipping "helpful for unseen examples signed by left-handed signers" (R@1 unchanged 0.39 in-domain, zero-shot gains) [V]; Bootstrapping paper +5 R@1 (supervised, split caveat) [V-A]; SignSeek mirrors LH into shared hand encoder [V-A] | Canonicalise, do not randomly flip. |
| Temporal speed/stretch | SignCLIP temporal aug sigma=0.2 **0.37 -> 0.39** [V]; slow-down -0.66 if removed [V] | Preserves identity (within limits); best-supported. |
| Spatial 2D aug (scale/shift/rotate) | SignCLIP spatial sigma=0.2 **0.37 -> 0.35**; Gaussian noise 0.001: 0.37 [V] | Mild harm at that strength; location parameter is absolute. |
| Keypoint standardisation | SignCLIP E6.1 in-domain R@1 0.37 -> 0.40, but "hurts out-of-domain" (their text) [V] | Helps in-domain, may hurt transfer. |
| Start/end frame permutation | SSL-SLR as above [V] | Identity-preserving by construction (non-discriminative frames). |

### 4.4 Phonological-parameter-aware objectives
- Hand-region reconstruction: SignMAE (2605.02094) ST mask + hand-region reconstruction 47.08 vs whole-frame 40.31 (WLASL2000 heatmap encoder) [V-A].
- BEST body-only masking harmful (47.72 < 51.96 no pretraining) [V].
- Motion-aware masking (select high-motion tokens): MAMP +1.2/+1.3 linear, MASA +1.3 (47.78 -> 49.06) [V].
- SignSeek saliency-guided articulator masking: ASGM vs random selection WLASL R@1 34.75 vs 29.66; ASLC 57.66 vs 52.22 (Table 8) [V].

### 4.5 Transfer verdict and minimal test (Q4)
Verdict: rest/activity handling is high-value and cheap because (a) the SignRep gain is a readout gain, (b) SSL-SLR gives direct evidence that ~half of each isolated clip is non-discriminative, (c) ASLC clips start and end at rest. Mirror flips and spatial jitter should be removed or replaced by dominant-hand canonicalisation; speed jitter kept. Test (S): measure frozen R@1 with (i) mean, (ii) motion-energy-weighted, (iii) drop first/last 25% bins, (iv) SignRep-style hand-height gating, all on the same checkpoint. A training-time variant (M): exclude rest bins from the loss; compare.

---

## 5. Q5. Minimal-pair / phonological structure

### 5.1 Evaluation protocols (usable on frozen SSL)
- **Yin et al. 2026** (2606.28667) [V]: ASL Citizen 259 Deaf-curated minimal pairs (137 handshape, 31 location, 73 movement, 18 orientation/NMM; 4,427 videos, 11 signers); intra-sign vs inter-sign cosine, t-test, % wins vs random-init. I3D-ASL 69.50 total (handshape 69.34, location **80.65**, movement 73.97); STGCN-ASL **81.08** (handshape **86.13**, location 67.74, movement 80.82); I3D-Kinetics 1.93. Sem-Lex out-of-domain roughly halves. Only supervised models tested; "SSL is future work". Code/data public (github.com/kayoyin/sign-phonology) [V].
- **SHuBERT / SignDino phonology benchmark**: 16 ASL-LEX heads, supervised fine-tune [V], not frozen.
- **Kezar VQ-ASL** frozen-code MLP probes [V].
- **Vandendriessche et al. 2026** kNN-vs-K multi-hot probe (A) [V-A].

### 5.2 Weak supervision from ASL-LEX (not SSL)
**Improving Sign Recognition with Phonology** (Kezar et al., EACL 2023, 2302.05759) [V Tables 2, 3]: WLASL, 48% of videos have ASL-LEX labels, aux heads on SL-GCN / transformer (OpenHands implementations), four seeds.
- WLASL_all test top-1: SL-GCN 29.4 -> **38.1 (+8.7)**, top-3 50.2 -> 61.0; Transformer 20.5 -> 23.4 (+2.8). Gains also on signs with no phonology labels in training (SL-GCN +8.3).
- Not all phonemes help: on the labelled val subset, baseline 50.2; handshape + minor location **55.7**; +non-dominant handshape/path/repeated 51.5; 9 types 52.5; all 16 types 54.3.
- Handshape, minor location, path movement are the least accurately predicted features.

### 5.3 Factorised / compositional heads
- **Kezar VQ-ASL**: structure without labels did not help (sec 3.2).
- **Vandendriessche et al.** (A) [V-A]: phoneme-supervised dual-branch network; gloss-trained embeddings are phonologically coherent only at K <= 4.
- **Zero-shot cross-lingual handshape recognition** (2609.18772, Sep 2026) [V abstract/Table 4]: five ASL-LEX handshape features (selected fingers, flexion, spread, thumb position, thumb contact) predicted, composite phonological distance decodes 37 LSC handshapes zero-shot: 80.0% feature accuracy, **54.5% expected handshape accuracy**, for a shallow MLP on dominant-hand features; SHuBERT 65.6 / 22.1; SL-GCN 70.4 / 29.5; cropping harmonisation +11 points. Supervised on ASL labels; single signer, 2 videos per class. The shallow MLP beating SHuBERT is attributed to overfitting to dataset/pose artifacts by high-capacity models. [V; interpretation is theirs]
- **PhonSSM** (2604.08761, Apr 2026) [V text]: four orthogonal phonological subspaces (handshape/location/movement/orientation) + prototypes, MediaPipe, 3.2M parameters. WLASL100 ablation: full 88.37, w/o factorisation 76.49 (-11.9), AGAN->MLP 79.84, BiSSM->LSTM 82.17. WLASL2000 72.1 claimed (+18.4 over their skeleton SOTA of 53.7, which ignores Uni-Sign's pose 63.13). Linear probes show diagonal 78.4/71.8/68.2/74.6 on branch-vs-parameter. **Flags**: single-source, supervised, "zero-shot to ASL Citizen 64.1" is measured with a merged training set that *contains ASL Citizen* ("held-out ASL Citizen samples") so it is not zero-shot; baselines are weak. Treat the factorisation effect size as unverified; the one robust statement is their own tradeoff finding that compositional models do better on phonologically distinct signs and worse on dense minimal pairs (Table 10). [V text; verdict [I]]
- Hard-negative mining: **SAN** (2607.09263) [V]: hard negatives from visual confusability in embedding space for text-sign retrieval on PHOENIX-2014T; the authors list phonological minimal pairs as future work. **Selective contrastive** gloss-free SLT (2604.22374) uses curriculum hard negatives, not phonological [S]. **No published method uses ASL-LEX minimal pairs as hard negatives.** SignSeek's GAC uses same-gloss positives (balanced k=40 glosses x n=2) [V].
- ASL-LEX identifiability: ~70% of signs are uniquely identified by their coded phonemes (Kezar 2023) [V]; P3 gives a 35% ceiling-loss figure with a different feature set [V-A]. Minimal-pair pools are small. [S]

### 5.4 Transfer verdict and minimal test (Q5)
Verdict: (a) a label-free route to compositional structure has negative evidence (VQ-ASL PD); (b) weak ASL-LEX supervision on ASLC train labels is the one robust gain (+8.7) but is a different track from SSL; (c) use Yin's protocol and ASL-LEX linear probes as *evaluation* of our frozen SSL, because no SSL has been tested on it; this is also a cheap diagnostic of where PLctx loses information.

Minimal test (S, evaluation only): run Yin's 259-pair protocol on PLctx vs SignRep (released checkpoint) vs raw hand-angle mean features. Prediction [I]: PLctx wins on handshape relative to location (position-normalised channels), and loses on movement. If wins on movement pairs are near the random-init level, the movement-probe finding (R2 < 0) is confirmed independent of the lexical task.

---

## 6. Q6. Frozen-feature quality for pose (and RGB references), with label status

ASL Citizen (F-R = training-free dictionary retrieval, test queries vs train gallery, class-max cosine):

| Method | Input | Training signal | Protocol | R@1 | Label-free? | Tag |
|---|---|---|---|---|---|---|
| Hand joint angles, mean-pooled | 3D pose | none | F-R | 8.81 | yes | [V] SignRep T4 |
| All joint angles / all keypoints | 3D pose | none | F-R | 0.57 / 0.98 | yes | [V] T4 |
| **MASA** | 2D pose, 49 joints from MMPose | 4 ISLR train sets (146k clips); includes WLASL/NMFs-CSL (transductive) but not ASLC | F-R | **27.96** (DCG 49.31, R@5 49.78) | **yes** | [V] SignSeek T2 |
| SignCLIP | MediaPipe pose | Spreadthesign text (456k) | F-R | 4.00 | no (text) | [V] SignSeek T2; its own paper: zero-shot 0.00, linear 0.23 |
| **SignRep** | RGB (+3D pose teacher) | YT-SL-25 | F-R weighted | **49.95** (avg 37.47) | yes | [V] T4 |
| SignSeek | 2D pose, 15.7M | 266k gloss-labelled clips (Sem-Lex, AUTSL, SLR500, MSASL, MeinDGS) | F-R | 57.66 | **no** | [V] T2 |
| Same, no MAC/GAC variants | | | | GAC only 48.40; GAC+MAP 55.93 | no | [V] T7 |
| SignMatch pose / RGB (2609.01886) | pose / RGB | BSL dictionary+corpus labels, pseudo-labels | F-R | 57.32 / 67.12 | no | [V-A] |

Other frozen pose protocols:

| Method | Dataset / protocol | Score | Label-free? | Tag |
|---|---|---|---|---|
| SignBERT / BEST / SignBERT+ / STC | WLASL2000 linear P-I | 18.24 / 21.19 / 24.50 / **32.35** | yes (pretrain on downstream train sets) | [V] STC Table V |
| same | MSASL1000 linear P-I | 31.54 / 30.89 / 34.44 / **48.30** | yes | [V] |
| same | SLR500 linear | 69.89 / 70.84 / 71.65 / 94.3 | yes | [V] |
| SSL-SLR | ASL Citizen-500 linear | 20.46 +/- 1.21 | yes | [V] |
| MASA | WLASL2000 F-R † | 14.38 | yes (transductive) | [V] SignSeek T2 |
| MASA vs SignRep vs SignSeek | zero-shot BSL SignBank (726 queries) R@1 | **6.80** vs 4.30 vs 12.76 (Video-Swin 9.15) | MASA/SignRep yes | [V] SignSeek T3 |
| SignCLIP | ASL Citizen: zero-shot 0.00; 10-shot 0.05; frozen+linear 0.23; scratch 0.39; fine-tune 0.46 | | no (text) | [V] Table 5 |
| Uni-Sign pose branch | ISLR fine-tune only; no frozen protocol | WLASL2000 63.13 P-I | no (text) | [V] |
| SHuBERT (LoRA rank 1) | ASL Citizen R@1/5/10 | .65/.87/.91; Sem-Lex .54/.74/.80 | yes pretrain | [V] |
| SignDino (LoRA / full fine-tune) | ASL Citizen R@1 | .67 / .704 | yes pretrain | [V], flagged |
| Logos (RGB, supervised RSL) | WLASL2000 frozen + linear head | 62.44 | no | [V-A] |
| PhonSSM | supervised, WLASL2000 | 72.1 (claimed) | no | [V], flagged |

**Which of these use only unlabeled data**: raw feature baselines, MASA, SignRep, SHuBERT, SignDino, SSL-SLR, SignBERT/BEST/STC (pretraining only). **The only label-free skeleton model with a frozen ASL Citizen retrieval number is MASA (27.96); no 2026 newcomer changed that** (SignReGraph, Multimedia Systems 2026-10-10, DOI 10.1007/s00530-026-02686-z, is a closed-access supervised GCN with masked frame reconstruction [S]). Our PLctx (16.7) sits between raw angle pooling (8.81) and MASA (27.96).

Frozen readout caveat from other domains [V]: MAE-style video features are 10-20 points weaker under linear than attentive probing (VideoMAE SSv2 41.3 vs 61.2; V-JEPA 50.1 vs 69.5); retrieval-style evaluation of motion-MAE skeleton features is 23 points below linear (MAMP 62.0 vs 84.9).

---

## 7. Candidate strategies ranked for our setting (max 8)

Ranking criterion: (evidence strength x expected frozen-retrieval effect) / cost, with ties broken by how directly the evidence targets our observed failures (movement R2 < 0, handshape R2 ~0.3, saturation). Effects are [I] unless the evidence is cited as a delta.

**1. Readout and layer diagnostics on the existing PLctx checkpoints (S; do first).**
- Mechanism: the information is in the encoder (supervised finetune ~55%); frozen mean-pooled last layer may not expose it. Candidate readouts: per-layer features (HuBERT: intermediate layers best, last degrade), multi-layer weighted sum (SHuBERT +7.9 BLEU-1 over last layer), motion-energy/hand-height weighted pooling (SignRep +12.48), bins2/chamfer (project +5.9/+9.4 [P]), attentive pooling as an upper-bound check (V-JEPA +16.1 SSv2).
- Expected effect on frozen R@1: +2 to +8 (largest of any single item per unit cost). Risk: low; weighting hyperparameters can overfit val (select on val only, report test once). Cost S.
- Minimal test: for the 200-epoch checkpoint, R@1 val for {mean, weighted, bins2, chamfer} x {last, penultimate, mid layer, layer-sum}. Decision: if best readout >= +4 over mean/last, the bottleneck is readout and Q3 items shift to second priority.

**2. Movement-explicit targets and multi-span latent targets (S-M).**
- Mechanism: predict motion-difference at interval(s) (MASA k=3, MAMP) and/or EMA-teacher latents pooled over 5/10/30-bin spans plus global (GFP); teacher input = temporal differences of unmasked pose (GFP T6: motion input best). Directly addresses movement R2 < 0.
- Evidence: MAMP +10.1 linear (84.9 vs 74.8); GFP retrieval 70.9 vs MAMP 62.0; MASA +1.8 to +4.2 (+5.4 with motion-aware masking); OpenHands warns against per-frame direction labels. SignRep targets: relational > absolute (+12-14 DCG) [V-A].
- Expected: movement probe R2 from <0 to >0.2; R@1 +2 to +6 [I]. Risk: noisy 2D jitter at short intervals; leakage when target is a function of visible input (Human-JEPA note). Cost S-M (one extra head + target channels).
- Minimal test: k in {2,4,8} bins; report movement R2, Yin movement-pair wins, R@1 val.

**3. Rest/activity handling at training and readout (S).**
- Mechanism: ~first third and last quarter of isolated clips are non-discriminative (SSL-SLR); activity weighting is +12.5 in SignRep. Train: mask-loss weighting by hand motion/height, exclude rest bins from targets; remove blind flip/spatial jitter, add dominant-hand canonicalisation and speed jitter (sec 4.3).
- Expected: +1 to +3 beyond item 1 (partly overlapping). Risk: weighting leaks signer speed (identity); canonicalisation errors for ambidextrous signs. Cost S.
- Minimal test: sec 4.5.

**4. Measured data step: 2.6x (105 h) with scale curve (M).**
- Mechanism/evidence: sec 1. Concave curves: +4 to +14 fine-tuned per 4 doublings (BEST/STC), +1.0 per 11x (V-JEPA 2), text-supervised scale far larger. MASA at 3.6x our data is 27.96.
- Expected: +1.5 to +5 R@1 [I]. Risk: mixed domains (ASLC webcam hands vs How2Sign studio), no signer IDs in How2Sign, leakage via MS-ASL/WLASL source videos. Cost M (data cached already; 4 pretraining runs).
- Minimal test: sec 1.4 with the pre-registered slope rule.

**5. Round-2 cluster targets (HuBERT recipe) with ASL-LEX purity audit (M).**
- Mechanism: refine targets via k-means on PLctx layer features (PNMI 0.28 -> 0.64 -> 0.70 gave WER 18.4 -> 11.9 -> 10.8 in speech); predicts discrete units per channel, as in SHuBERT/BEST (token vs regress +1.9 at equal masking).
- Expected: uncertain, 0 to +5 [I]; the purity audit (handshape / location / path-movement PNMI, signer PNMI) is itself a deliverable. Risk: signer clusters; handshape units on 2D keypoints may be impure (3D-LEX glove 0.49 vs 2D MediaPipe handshape 46% on 37 classes). Cost M.
- Minimal test: sec 3.4 with the PNMI gate.

**6. Auxiliary future-block / multi-horizon prediction head (M).**
- Mechanism: add past-to-future latent prediction (multiple horizons) as an auxiliary to masked prediction; OpenHands DPC (+3.5; few-shot +20), CPC multi-step (28.5 -> 64.6), Human-JEPA (+8.6 over block-only in continued pretraining).
- Expected: movement encoding may improve; R@1 effect uncertain (V-JEPA from scratch prefers bidirectional multi-block). Risk: dominated by interpolation-like shortcuts if future context leaks; evidence regime mismatch (RGB, adaptation). Cost M.
- Minimal test: mask mixture ablation (sec 2.6 test B), fixed compute.

**7. Larger unlabeled pose corpus: SignVerse-2M / YouTube-ASL subset (L; conditional on item 4 slope).**
- Mechanism: SignRep-scale diversity at 300-1000 h of pose; cross-lingual pose is acceptable (OpenHands, Uni-Sign). SignVerse-2M provides 3,207 h DWPose (extractor mismatch with RTMW-x). D1: 94 GB free, YouTube-ASL not downloaded (~76 GPU-days for 3,000 h); FP16 subset ~1,000 h feasible.
- Expected: unknown; the published frozen evidence is only the RGB SignRep. Risk: highest cost; extractor domain shift; test overlap (111 MS-ASL videos in YouTube-ASL). Cost L.
- Minimal test: pretrain on a 100-300 h SignVerse-2M ASL subset only if item 4 shows a positive slope; evaluate frozen ASLC val.

**8. Weak phonological supervision track as an upper-bound probe (S-M; not SSL).**
- Mechanism: ASL-LEX aux heads (handshape + minor location first) on frozen PLctx features or during a short adaptation stage; Kezar +8.7. Use to bound how much of the gap is *recoverable from the features* versus *absent from them*.
- Expected: not a frozen-SSL improvement; informs whether items 2/5 can possibly help. Risk: contaminates the "label-free" claim if used in the main number. Cost S.
- Minimal test: linear probes for 16 ASL-LEX features on frozen PLctx vs raw hand-angle features vs SignRep, with a gloss-disjoint split (see A gap 4).

### The most valuable investment, as the evidence reads
- **Objective/target/readout > data at the current regime.** The best-controlled evidence (same data, different objective: R@1 0.39 vs 37.47/49.95; same data, different target: +10.1 linear NTU, +4.2 WLASL) is an order of magnitude larger than the best-controlled scale evidence for label-free models (+1.0 for 11x videos; +2.3 BLEURT for 10x sign hours). A readout change (+12.48 SignRep; +16 V-JEPA) has comparable size to the target changes and costs a day.
- **Data is not excluded.** No frozen scale curve exists for sign; the concave fine-tune curves suggest a 2.6x step would give a few points. Buy that information with the 105 h step before any acquisition.
- **Counter-evidence to weigh**: SignRep's gap over pixel MAE might partly reflect its 3D pose teacher and RGB capacity (A sec 2 confounds), which a 2D-skeleton model cannot copy by changing the loss alone. If items 1-3 leave R@1 < 28, the remaining gap to MASA is likely data and motion target together.

---

## 8. Verification log and open items

Fetched and grepped this session (raw HTML via `arxiv.org/html/<id>`, tags stripped; numbers matched to table captions): 2503.08529 (SignRep T4 incl. 49.95 / 37.47 / 71.21 / 61.40, weighting Eq. 11-12), 2411.16765, 2501.15187, 2402.09611, 2407.01264, 2306.15162, 2110.05877, 2405.20666, 2609.03695, 2509.21223, 2605.01720, 2606.28667, 2509.04745, 2302.05075, 2406.10501, 2509.05188, 2609.06296, 2106.07447, 1807.03748, 1909.04656, 2101.07974, 2404.08471, 2506.09985, 2308.07092, 2509.03609, 2608.21160, 2608.27395, 2606.23256, 2412.11553, 2302.05759, 2409.01901, 2509.18309, 2609.18772, 2604.08761, 2607.09263, 2607.11215, 1911.12409 (abstract-level only for sec 2.2 table; its numbers are quoted from MAMP/GFP tables).

Not re-verified (taken from A/L4/P3/D1): SignMatch (2609.01886), Bootstrapping paper (+5 handedness), Vandendriessche et al., SignMAE (2605.02094), Logos, SignRep Table 5/6 (re-confirmed only Table 4), SLiM (2603.10648, fetched but not used here), SignSeek's MAC definition.

Corrections to prior docs:
1. A claims "nobody has published a label-free skeleton model pretrained at YT-SL-25 scale" and "MASA 27.96 is the label-free skeleton best". Refinement: OpenHands pretrained a label-free pose ST-GCN on 1,129 h (fine-tune only). MASA remains the only label-free skeleton with a frozen ASLC retrieval number.
2. A cites STC linear numbers as "Table VII"; in the fetched arXiv version they are **Table V** (semi-supervised is Table VII). Numbers match.
3. MASA's 27.96 on ASL Citizen is clean (ASLC absent from its pretraining sets); its WLASL (14.38) and NMFs-CSL (54.69) numbers are transductive (marked † in SignSeek T2).
4. SignRep's paper does not state YT-SL-25 hours; ~3,000/3,207 h comes from SignVerse-2M (3,207 h) and SHuBERT (~3,200 h) [V].
5. S-JEPA: no arXiv preprint found; cite ECCV 2024 [S].

Items searched and not found: any slow-feature/temporal-coherence SSL for sign pose; any unit-purity evaluation of unsupervised sign units against ASL-LEX; any ablation of dropping rest frames at train time for pose SSL; any frozen sign scale curve; any SSL evaluated on Yin's minimal-pair protocol; any use of ASL-LEX minimal pairs as hard negatives. Searches were arXiv-HTML fetch of candidate papers plus WebSearch (standard/extended), 2026-10-17; absence claims are scoped to those searches.

Reliability notes: SignDino and PhonSSM are single-group preprints with claims that could not be cross-checked (SignDino smooth scaling curve and tight seed stds; PhonSSM "zero-shot" on a corpus containing the test source). Human-JEPA and LeVJEPA are Aug 2026 preprints; their regimes (continued pretraining; ImageNet probing) limit transfer to a from-scratch skeleton model.
