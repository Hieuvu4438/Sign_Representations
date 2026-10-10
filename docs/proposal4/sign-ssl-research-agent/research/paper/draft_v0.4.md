# Averaging Hides What Frozen Sign Encoders Know: Readout Matters for Self-Supervised Sign Representations

*Draft v0.4 (2026-10-06). All tables are regenerated from raw rank files by `method/p4_signssl/scripts/make_paper_tables.py` → `paper/tables.md`, `paper/tables.json`, `paper/fig_readout_gap.png`. Internal; every number traces to an artifact listed in `research/EXPERIMENT_LEDGER.md` / `research/CLAIM_LEDGER.md`.*

## Abstract

Self-supervised sign-language encoders are commonly compared by frozen dictionary retrieval after pooling short-window features into one clip vector. We ask how much of the measured quality is a property of that pooling. Re-implementing the frozen protocol of SignRep, a recent RGB sign encoder, we reproduce its ASL Citizen result (R@1 49.0 vs 49.95 reported). Keeping the released encoder and its window embeddings fixed, a training-free readout that keeps the order of two temporal segments raises test R@1 to 54.9 (+5.9, 95% CI [3.9, 7.5], 11 signer clusters), and matching window sets raises it to 58.4 (+9.4 [8.2, 10.6]); on WLASL2000 and MS-ASL the gains are smaller (+2.0/+1.2 and +1.9/+4.5). Diagnostics show that the averaged embedding is nearly invariant to time reversal and that retrieval errors share handshape and location with the true sign far more often than movement. We also report three method attempts that did not help under matched controls — continued pretraining with cross-window latent prediction, a conditional cross-articulator forecasting objective, and label-free distillation of set similarity into a single vector. We argue that new SSL objectives for signs should be compared against the original encoder with an order-aware readout, not only the averaged one.

## 1. Introduction

- Frozen evaluation is the standard way to compare SSL sign representations because it isolates the representation from downstream training. In practice the representation is a sequence (windows or frames) and the benchmark score depends on how it is reduced to one vector.
- SignRep [S01] computes 16-frame window embeddings and averages them, weighted by a predicted hand-activity score. Every sign SSL method we surveyed uses an order-insensitive clip readout (mean, weighted mean or attentive pooling) for isolated-sign evaluation [notes P101–P115].
- Questions: (Q1) does the averaged readout under-state what the encoder knows; (Q2) what does averaging discard; (Q3) do encoder-side objectives that target temporal structure help once the readout is fixed?

## 2. Setup

- Encoder: released SignRep checkpoint (Hiera-B), vendored commit 06f40b5d, verified to reproduce vendor outputs (min window cosine ≥ 0.99996 in bf16).
- Windows: 16 consecutive native frames, stride 2; window embedding = SignRep `features` (z^avg); activity weight γ = clip(max(P_LH, P_RH), 0, 1) from the released activity head.
- Retrieval: gallery = official train split; query = val (development) or test (single confirmatory run); class score = max cosine over the class's gallery clips; ties counted against the query; metrics DCG, MRR, R@k as in ASL Citizen.
- Readouts on the *same* windows: R0 — SignRep's activity-weighted mean; R1 — bins2: restrict to the active span (γ ≥ 0.5), split it into two halves by window index, activity-weighted mean per half, concatenate the L2-normalised halves (a single 1536-d vector); R2 — chamfer@20: rerank the 20 best classes of the active-span mean by symmetric chamfer similarity between window sets.
- Data: ASL Citizen (signer-disjoint: 35/6/11 signers; 2,731 glosses), WLASL2000 (local pre-trimmed clips, all videos present).
- Statistics: paired bootstrap clustered by signer (ASL Citizen) or class (WLASL), 10,000 resamples, Holm correction over the two primary comparisons. The confirmatory protocol (readouts, thresholds, decision rules) was locked before the test split was extracted; the test split was evaluated once.

## 3. Results

### 3.1 Reproduction (C011)

R0 on ASL Citizen test: DCG 70.4 / R@1 49.0 / R@5 79.3 (SignRep reports 71.21 / 49.95 / 80.09). On WLASL2000 val R0 is 30.4 and on test 29.4 (reported test: 29.92).

### 3.2 Readout gap (C012, C015, confirmatory)

![Readout gap](fig_readout_gap.png)

*Figure 1. Test R@1 of the same frozen SignRep window embeddings under three readouts; labels give R@1 for the averaged readout and the gain over it for the others. Exact values and CIs: `tables.md`.*

| Test set | R0 (paper) | R1 bins2 | R2 chamfer@20 |
| --- | --- | --- | --- |
| ASL Citizen R@1 | 49.0 | 54.9 (+5.9 [3.9, 7.5]) | 58.4 (+9.4 [8.2, 10.6]) |
| ASL Citizen MRR | 62.3 | 67.1 | 69.8 |
| WLASL2000 R@1 | 29.4 | 31.4 (+2.0 [0.9, 3.1]) | 30.6 (+1.2 [−0.2, 2.5]) |
| MS-ASL R@1 (separate locked protocol, fresh holdout) | 40.9 | 42.8 (+1.9 [0.8, 2.9]) | 45.5 (+4.5 [2.9, 6.1]) |

On ASL Citizen R1 improves for 10/11 test signers and R2 for 11/11. MS-ASL test clips overlap YouTube-ASL videos that may be in SignRep's pretraining data, so its absolute numbers are not clean; the comparison is between readouts of identical embeddings. A label-free PCA-whitening of the single-vector readout, which helped on ASL Citizen validation (+2.6), lowered MS-ASL test R@1 by 2.4 points and increased the same-signer distractor penalty on WLASL validation, so we do not recommend it; whitening windows before set matching raised MS-ASL chamfer R@1 by 2.4 (secondary analysis).

Development analyses on ASL Citizen val (official gallery): simple single-vector pools do not close the gap (normalised mean 58.2, max 60.0, GeM 56.0, active-span weighted mean 59.5 vs R0 59.2); ordered bins K = 2/3/4 give 64.9/65.2/65.0; DTW alignment was worse than chamfer in a leave-one-signer-out pilot (51.6 vs 56.7 R@1); adding a temporal-position penalty to chamfer does not help (57.1 vs 56.8).

### 3.3 Reference check with SHuBERT (val only, not a main result)

SHuBERT produces contextual per-frame features over the whole clip. Using the block outputs (residual stream) of its last layer, frozen leave-one-signer-out retrieval on ASL Citizen val gives 4.2 R@1 with a mean over all frames and 10.3 with a mean over frames where hands and face are tracked; ordered two-segment and set readouts raise this to 13.9 (+3.5 [2.2, 4.1]) and 14.3 (+4.0 [2.4, 5.1]), 5/6 signers. The per-layer outputs that the authors' feature script returns (the FFN branch before the residual addition) are near chance for frozen kNN (≤ 4.6 R@1 at any layer), so the choice of internal feature matters as much as the readout for this encoder.

### 3.4 What averaging discards (exploratory, val)

- *Time reversal*: the averaged embedding of a reversed clip has cosine 0.959 with the original (same gloss, different signer: 0.561). Querying with reversed clips costs only 2.4 R@1 points (50.3 → 47.9); shuffling frames within windows costs 6.3 points, freezing each window to its middle frame 8.4 points.
- *Phonology (ASL-LEX 2.0, evaluation only)*: among rank-1 errors, the retrieved sign shares the true sign's handshape 44% of the time (5% by chance), its location 46% (16%), its path movement 27% (17%).
- *Who gains*: the chamfer gain on val is +10.8 points for signs with non-repeated path movement, +7.3 for repeated movement and +9.6 for signs without path movement, i.e. not movement-specific.
- *What set matching exploits*: one-directional query→gallery matching (each query window takes its best gallery window) reaches 68.7 R@1 on a 3,000-query val subset, symmetric 68.3, best-half-of-windows 63.6, gallery→query 62.1, averaged 58.9 — the benefit is phase coverage without temporal alignment, not discarding unmatched windows.
- *Gallery size*: subsampling the gallery to 3/7/15 exemplars per class leaves chamfer gains of +7.9/+9.5/+9.6.
- *Signer nuisance*: on WLASL, removing the query signer's own clips from the gallery raises R0 from 30.4 to 32.8, i.e. same-signer clips of other signs act as distractors; among unlabelled ASL Citizen train clips, the nearest neighbour under set matching shares the signer 49% of the time and the sign only 44%, and the most confident 10% of neighbours share the signer 82% of the time.

### 3.5 Encoder-side objectives under matched controls (negative results)

- *Cross-window latent prediction* (continue SignRep on unlabeled ASL Citizen train videos; each masked window predicts the frozen teacher's embedding of the window 8 frames earlier/later): +0.51 R@1 over the released model, but the matched same-window control gives +0.44 and anchor-only +0.38 (val, leave-one-signer-out); the prediction target adds +0.07 [−0.12, 0.38].
- *Conditional cross-articulator innovation* (forecast the next state of one articulator from its own past vs. its own plus another articulator's past; 2-D whole-body keypoints, 5 group folds): 0 of 36 articulator-pair × horizon gates pass on isolated signs (ASL Citizen), on continuous signing (8,000 How2Sign sentences), and on continuous signing with an appearance (DINO hand/face crops) + 3-D hand-landmark teacher (1,200 sentences); adding the other articulator's history never reduces held-out forecast error by ≥ 5% (best +0.1% on How2Sign), and ridge regression agrees.
- *Longer windows*: a 32-frame student initialised from SignRep and trained to encode the teacher embeddings of its two 16-frame halves in order fell to 41.8 R@1 (val, leave-one-signer-out) vs 49.5 for the original, and below an order-free control (43.7); without an anchor to the original space, longer-window continuation degraded retrieval.
- *Mined positives*: because confident unlabelled neighbours are mostly same-signer pairs, we did not train on them (rejected at the diagnostic stage).
- *Label-free set-to-vector distillation* (train a single-vector student to reproduce chamfer similarities among unlabeled train clips): 60.1 R@1 with a transformer student and 64.4 with a bins2-initialised soft-segment student, both below the training-free R1 (64.9) on val.

### 3.6 Does the readout change conclusions about objectives? (val, leave-one-signer-out)

For the four continuation checkpoints of §3.5 (original, cross-window prediction, same-window control, anchor-only), the ranking is the same under both readouts, but the gains over the original encoder shrink from +0.39…+0.51 R@1 with the averaged readout to +0.08…+0.24 with the ordered readout (all CIs include 0 under the latter). Part of what an objective appears to add under averaging is information the original encoder already had and a better readout recovers.

## 4. Discussion

- The frozen SignRep window embeddings carry substantially more retrievable information than the averaged readout exposes, especially under signer shift (ASL Citizen). Benchmarks of SSL sign encoders that only report averaged readouts conflate representation and readout quality.
- The released encoder plus an ordered two-segment readout is a stronger, training-free baseline (+5.9 R@1, same storage class: one vector) that new objectives should be compared against.
- Objectives that change the encoder at the scale we could afford (≈ 4k steps) moved retrieval by < 1 point regardless of the target, which is small relative to the readout effect.

## 5. Limitations

One encoder in the confirmatory test-set claim (the SHuBERT replication is on val only and at a low absolute accuracy); three ASL datasets with different effect sizes (largest on signer-disjoint ASL Citizen) and an unexplained difference; WLASL is not signer-independent and SignRep selected its checkpoint on WLASL validation; the ASL Citizen test split had been evaluated by unrelated projects in the same repository (no results from them were used); ASL-LEX codes describe initial morphemes only and lack orientation; no review by Deaf signers or linguists; method pilots used one seed and short continuation budgets; the conditional-innovation tests used local teachers (2-D keypoints; author-fine-tuned DINOv2 crops + MediaPipe 3-D hands) and at most 8,000 continuous-signing clips, not the YouTube-SL-25-scale data and generic DINOv3 teacher of the original design.

## 6. Reproducibility

Commands, hashes and assets: `research/reproducibility_checklist.md`; code `method/p4_signssl/`.
