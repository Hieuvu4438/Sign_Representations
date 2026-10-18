# R2 verification review: proposal17 v1.1 against R1

Reviewer: same independent reviewer as R1. Date 2026-10-17. CPU only (<= 4 threads), one process at a time, no test split, no repo file modified except this report.
Scratch scripts and outputs: `/tmp/claude-1007/-home-haipd-Sign-Representations/32c5829a-820b-4c3b-9d20-e66b08713d70/scratchpad/review/chk12.py` to `chk16.py` (chk1-11 are from R1). New outputs: `chk13.py` (NaN-safe morph), `chk14.py` (dose comparison), `chk15.log` (long full-batch head), `chk16.py` (routed N per signer and K-means seed).

## 0. RETRACTION of one R1 statement

R1 section (c) item 6 said the morph perturbation "implementation matches its docstring". That was wrong. The wrist keypoint is NaN in 0-90 % of frames of many clips while the hand keypoints are observed, and `morph()` propagates the NaN into the hand. See finding N1. The v1.1 dose table inherits the bug.

## 1. F1..F15 status

| ID | Status | Evidence (v1.1 section) |
|---|---|---|
| F1 | **Fixed** | §2.8 and §0.5 replace eta2 by scale-free decode (89.1 to 94.6 %), kNN (71.9 to 84.4 %), PR (7.5 vs 83.5). All match `retention_scalefree.json`. Own whitened eta2 (0.055 to 0.031) goes the other way (mine: 0.023 to 0.029, PCA-512); the doc discloses that the whitened eta2 is uninformative. |
| F2 | **Fixed** | §2.10 full-pool table equals `nn_precision_fullpool.json` (CORAL mutual 44.8 %). Threshold relabelled [I]. Heads from CORAL-geometry positives reported (+3.0 signer id, +0.8 label-free). Minor: "+2.6...+3.0" should be +2.5...+3.0 (31.97/29.44). |
| F3 | **Fixed** | §2.5 and §5.3: NMI 0.332 vs 0.358 random, AMI -0.016 vs -0.0005, rule made log-only; values match `cls_early.json`. |
| F4 | **Fixed (minor residue)** | §0.6, §2.11 say "không phán quyết được", deltas from JSON. Residue: §7 risk row is still titled "Whitening không bền" (contradicts §0.6/Z2). |
| F5 | **Partly fixed** | §2.4 reworded (per-signer whitening 41.5, "không phải ánh xạ tuyến tính"). Composition control changed to "signers >= 1500 glosses" (12 signers, 80 % of clips): CORAL-whiten +8.3. My R1 control with >= 2400 glosses (9 signers) gave +6.5. The softer threshold was chosen after the fact; "một phần nhỏ phụ thuộc thành phần" understates a ~28 % shrink. |
| F6 | **Partly fixed** | K-free grid + silhouette done (§2.5), 8-place inventory present (§2.11). Residues: (i) the "mean over K" range Kq in {4,6,8,12,16} is centred on the true signer count 6 (silhouette picked K=24, no gain); (ii) §2.11 CV table row is mislabelled (N3); (iii) test has 11 signers, range not validated. |
| F7 | **Not fixed** | The new dose study is built on a NaN-propagation bug (N1). The "dose-independent shift 0.29-0.42" is an artefact. The 28-signer regression disclaimer was added but E-canon NO-GO (§3.2, §8.1) now rests on contaminated data. |
| F8 | **Partly fixed** | §6.5 rewritten around layer G (30-33) and layer J (+0-4). Still: G forecast includes whitening whose robustness §0.6 calls undecided; "end-to-end val 30-36" top end exceeds G + J ranges; test 0.73x ratio was measured for raw only, not for transductive/geometry gains. |
| F9 | **Partly fixed** | R demoted, redesigned (§5.6), gate G-R. The mini-batch emulation used to justify demotion has flaws that can explain its failure (N8, my control). Demotion is reasonable, the evidence wording is not. |
| F10 | **Fixed** | §5.3 C-fixed default, T1-shuf and T1m in the economy plan (§6.1), AMI log-only, T1o separate. |
| F11 | **Fixed (new minor)** | §4 H1 three-way interval, H3 non-inferiority, H5 no-harm, t-CI over signers. New minor: KEEP rule "no signer worse by > 1 point" (§6.3) is below the noise of P12/P5 (about 600-700 clips, binomial SE about 1.4 points). H3 margin tau is not given a number. |
| F12 | **Fixed** | §2.3 +3.4 per doubling (8-16: +4.0; 16-35: +3.8), labelled as gallery-time effect; H7 now a design rule. |
| F13 | **Fixed** | §2.2 numbers equal `ladder_extra.json` (66.0 / 65.4 vs 35.9 raw, 44.4 whiten; in-sample 67.3 / 68.0; 96-98 % recovered). Headline 66.0 is k=64, chosen on the evaluated glosses (k=96: 65.4); immaterial. |
| F14 | **Fixed** | ABX within/across split, Bootleg nuance, [I]/[V], probe caveats present in §3.1, §3.2, §2.8. |
| F15 | **Mostly fixed** | NaN removed from JSON (checked: no `NaN` in any results file), delta rounding, 30 variables, k notes fixed. Residual numeric slips listed in section 2. |

## 2. New numbers vs JSON (check 2)

Verified equal to JSON (to the rounding shown): §0 items 1-6 except those below; §2.2 table and k grid; §2.4 table, heavy-gallery numbers (79.7 %, 32,015 clips, 21.05/27.96/30.18/36.24/39.57); §2.5 CLS table (AMI, ARI, decode 85.9/88.7/94.6, AMI gloss), monitor numbers, K-free table (A 29.45 / 33.41 [29.26-36.24]; C 29.44 / 31.56 [30.24-32.59]; B 25.35 / 24.69), routed table (all 15 cells, +1.8...+3.7, random 20.75-22.19); §2.6 mini-batch table (17.78 / 20.01 / 12.27 / 15.55 / 21.20 / 19.41); §2.7 style numbers and decode 94.2 % (94.23), dose table (all cos, shifts, delta R@1 values); §2.8 retention table (89.1/94.6/PR 7.5/83.5/71.9/84.4); §2.10 precision table and head tables (also half-pool 8.7/14.6, 14.7/21.8, 26.4/36.6, 28.2/41.3); §2.11 `ladder_cv` deltas.

Mismatches found:
1. §2.5 table and §0 item 3: AMI(signer) train at 160 epochs "0,72"; JSON 0.7149 (should be 0.71).
2. §2.11 `ladder_cv` table: row "CORAL không nhãn (K chọn bằng silhouette) + whiten 23.9 / +7.61" is the v1.0 `ladder_cv.py` result with K_q = 6 (the true number of signers) and K_g = 60. The silhouette version (`kfree_cluster_corral.json` B) is 25.35 chosen / 24.69 mean, i.e. +9.1 / +8.4 over raw. Label and number do not match.
3. §2.10: "+2,6…+3,0 trên pool đầy đủ" - JSON gives +2.53 and +2.97.
4. §2.7: "SE ≈ 0,9" - paired SE on the 1,500 balanced clips is 0.6-0.7 (unpaired 0.9). Harmless, but then the "no change" threshold is tighter than stated.
5. §2.4: heavy threshold differs from the one in R1 (see F5), so the numbers match JSON but not the control R1 asked for.
6. §0 item 3 headline "28,6 (+3,7)" is the best of 5 K values; default K=35 gives 28.32 (+3.4), mean over the 5 K is 27.90 (+3.0). Stated range "K = 8-48: 27.9-28.6" is correct but omits K=70 (26.7).
7. §2.8 prose: whitened eta2 0.055 to 0.031 is quoted; my PCA-512 whitened eta2 is 0.023 to 0.029. Different whitening, same conclusion (not informative).

## 3. New code: bugs and leaks (check 3)

- **`routed_coral.py`**: inductive. Query transform uses only the query's own CLS (cosine to gallery centroids), gallery-fitted cluster statistics and the gallery-fitted PCA-whiten. No val statistic or label enters the fit. Gallery clips are in-sample for their own cluster statistics (same as the coral-head setting), which is the intended inductive asymmetry. Caveats: single K-means seed per K and n_init=3; K swept on val (K=35 default is not val-tuned, the headline max is).
  - Reproduction (chk16): K=35, K-means seeds 0/1/2 give +3.38 / +3.71 / +3.45 over whiten (paired SE 0.55); K=16 +3.67. All 5 val signers improve for every run (per-signer +1.6...+5.4). The effect is real on this checkpoint and half-subset.
- **`kfree_cluster_corral.py`**: no label or val leakage. Silhouette uses features only, on 4,000-point subsamples. It is transductive on the query set (declared). The summary range Kq 4-16 is an informed prior (true #signers = 6).
- **`coral_head_mb.py`**: the estimator is as described in §5.6/v1.0 in structure but has issues that can explain failure independently of the method:
  1. EMA statistics are of Z = X W with W changing every step (lag; B=256 is worst, 12.3, which is inconsistent with a pure small-batch-noise story).
  2. Cluster covariance EMA and global covariance are initialised from the first batch (rank <= 31 in 512-d, plus 1e-3 I) without bias correction.
  3. The batch covariance term is `(Z_k - mu_EMA)(Z_k - mu_EMA)^T / n` with n about 1.4: rank-1, and its squared error loss penalises ||z - mu||^4 (shrinks within-cluster deviations, which carry gloss signal), a biased gradient.
  4. 3000 steps, lr 1e-3, weights (0.25, 1, 0.25), against 250 steps / lr 2e-3 in the full-batch head. I controlled for this (chk15): the same weights, lr 1e-3, **full-batch statistics**, CLS-48 true clusters: **27.78 at 250 steps and 28.44 at 1500 steps** (no collapse, above whiten 24.94); **shuffled clusters 23.12 / 22.15** (below whiten). So training length and missing anchor are not the cause; the mini-batch estimator is. It also shows that shuffled clusters hurt even with exact statistics (-1.8...-2.8), which makes the mini-batch "shuffled 19.41 vs true 20.01" comparison uninformative about specificity.
  - Not a leak; it is a faithful emulation of the v1.0 estimator but not of "the training mechanism". It does not touch val.
- **`nn_precision_fullpool.py`**: correct. Full pool 40,154, cross-group exclusion by mask, mutual = NN of NN. My independent full-pool run matched (raw 14.29/18.96/25.85, whiten 22.92/25.19/33.44, CORAL 32.35/28.93/44.78). Half-pool part drops zero vectors (20,060 clips), so the whiten reference there is 24.98, not 24.94 (the doc quotes 24.9).
- **`cls_early.py`**: the even/odd ARI is inflated. Windows are stride 16 at 64 samples; from metadata, **74.1 % of train clips and 66.7 % of val clips have exactly one window**, so even and odd means are the same vector and the assignment is identical. Only 11.5 % / 10.9 % of clips have >= 4 windows. ARI 0.97 train / 0.80 val is therefore an upper bound and does not demonstrate window-to-window stability (N2). The AMI and decode values are fine. Also the pilot checkpoints are the 4k/16k-step 3,000-clip pilots, about 3-13 epochs of the real data; they are a proxy for "early", not for epoch 10/40 of the full run, and all three were trained without T/R.
- **`morph_dose.py` and patched `emb_variants.morph`**: the nonuni mode and the scale parameter are implemented as described (log-uniform in [1/s, s], per arm, per hand, per finger base joint). But `morph()` (all non-face modes) computes `w_new = e_new + sa*(w_old - e_old)` and then moves the whole hand with `w_new - w_old`. Where the body wrist (LWRI/RWRI) is NaN the whole hand becomes NaN. See N1. `morph_dose.py` itself (R@1, subspace energy) is fine.

## 4. Judgement of new claims (check 4)

(a) "Inductive CLS-routed normalisation N gives +3.7 over whiten": **supported on what was measured, over-stated in generality**.
- Real: reproduced (+3.4...+3.7, SE 0.55, 5/5 signers, stable over K-means seeds), random routing -2.7...-4.2 below whiten.
- Selection: headline is best of K in {8,16,35,48,70}; honest figure +3.0...+3.4.
- Scope: one checkpoint, half gallery (20k), 5 effective signers, no CV on train signers, no full set, no other seed of the model. v1.1 states this in §5.2 and G-G, which is fair.
- Missing controls: K=1/global (equals whiten), routing by nearest `grp` cluster instead of CLS (is CLS special?), kNN-local whitening from the query's gallery neighbours (an inductive CLS-free alternative), and comparison to the +1.0 of local-mean on full val. Random routing is a weak null (it breaks the transform, so any wrongly routed rule loses).
- Gallery signers are in-sample for their cluster statistics; query signers are unseen, so a real unseen-signer test (test has 11) is needed.

(b) "Mini-batch R emulation fails": the doc's own caveat (no anchor, "không chứng minh R thất bại trong encoder") is fair. The §0.4 wording "mô phỏng đúng cơ chế mini-batch" is too strong: the emulation has the estimator defects in section 3, the control shows full-batch statistics with identical lr/weights/steps do not collapse (28.44), and true vs shuffled clusters are indistinguishable (20.0 vs 19.4) so specificity is not measured. Right conclusion: "the v1.0 estimator is unstable; the v1.1 teacher-bank/low-rank design is untested".

(c) Forecast §6.5 (G alone 30-33, end-to-end 30-36, test 22-26): G alone = whiten 29.4 + N 3.4 ≈ 32.8 is arithmetically consistent, but it adds the undecided whitening gain and an N effect never measured on the full gallery. "36" is unsupported. The 0.73 test/val ratio comes from raw R@1; there is no evidence it applies to transductive/geometry gains with a different signer population.

(d) §0 item 6 on CV and whitening: **fair** (says it cannot decide). Residual inconsistency: §7 row title and §3.1 T-iso row still lean on "whitening không bền".

(e) Thesis shift toward a no-train geometry layer: **supported by the data, but it changes what is being proposed.**
- The best evidence (N, whiten-per-signer, CORAL) is inference-time and could be applied to any frozen embedding; the proposal's own SignRep-with-same-stack test is still "optional" (+6-10 GPU-h). Without it, "SA-JEPA beats X" would compare a stack to a raw baseline.
- N depends on a CLS code that exists only in this model family.
- The training layer J has essentially no direct diagnostic support: T has CMN +2.2 at the pooled-embedding level; D1-D3 have none; R failed in emulation.
- The title still says SA-JEPA.

## 5. New Critical / Major problems introduced or exposed by v1.1 (check 5)

- **N1 (Critical)**: morph NaN-propagation bug invalidates the dose table (§2.7), §0 item 5 bullet 1 ("dịch ≤ 0,42, R@1 không đổi"), the E-canon NO-GO (§3.2) and §8.1; it also affects v1.0 `morph_effect.py`.
  - Evidence (chk12): with scale = 1.0000001 (no perturbation) `hand`/`arm` modes change 5.4 % of observation flags and move keypoints by up to 639 px, because LWRI is NaN in 0-90 % of frames (e.g. 46 %, 41 %, 90 %) while hand joints are observed.
  - This is why the shift is 0.29 already at +-15 % and does not grow with dose (hand mode, which scales only the hand, gives the same 0.28-0.29 as arm mode).
  - NaN-safe re-run (chk13/14, same 1,500 balanced clips, same clean embeddings; hand translated by the finite elbow/wrist delta and scaled about the hand root; flags unchanged at scale ~1: 2e-5):

    | Mode, scale | cos to clean | shift norm | delta R@1 (paired SE) |
    |---|---:|---:|---:|
    | all +-15 % | 0.983 | 0.088 | -0.67 (0.39) |
    | all +-50 % | 0.950 | 0.225 | +0.00 (0.60) |
    | arm +-50 % | 0.950 | 0.215 | **-1.47 (0.57)** |
    | hand +-50 % | 0.997 | 0.049 | +0.33 (0.32) |

    Real shifts are 2-6x smaller than reported at small dose and grow with dose. R@1 is robust, but arm length at +-50 % costs 1.5 points (about 2.6 SE), not "không đổi". The real shifts (0.09-0.23) are far below the cross-signer 0.90, so "perturbation is too weak" remains the honest reading and E-canon NO-GO is not established.
- **N2 (Major)**: stability evidence (ARI 0.92-0.97 / 0.79-0.83) is mostly trivial (74 % / 67 % single-window clips). Needed: ARI on clips with >= 4 windows, or on halves of the clip with independent windows.
- **N3 (Major)**: §2.11 CV row presents an oracle-K number as K-free (mismatch 2 in section 2).
- **N4 (Major)**: N lacks the controls in section 4(a) and any held-out-signer evidence; it is now the centre of the proposal (layer G, G-G gate).
- **N5 (Major)**: mini-batch failure is attributed to the method where the estimator and bank design are at fault (section 3); the redesigned R has no CPU emulation that passes yet (acknowledged, gated by G-R).
- **N6 (Minor)**: composition control threshold softened (>= 1500 glosses); KEEP rule "no signer worse than 1 point" below noise; H3 margin unspecified; "+3.7" is best-of-K; numeric slips of section 2.

## 6. The single most important thing still missing before GPU approval

A CPU-and-tiny-GPU gate that tests the premise of the GPU spend, which is layer J, not layer G. All positive evidence in v1.1 is for a no-train inference layer (N, whitening), whereas the 60-80 GPU-h programme buys training changes (T, D1-D3, R) with no diagnostic of their own. Concretely:
1. Run G-G first on the **full 40k gallery with unseen-signer evidence** (val plus held-out-signer CV from a model trained without those signers, or at least multi-seed K-means and the missing controls above), with SignRep + same stack made mandatory.
2. Before the Phase 2 matrix, run a <= 1 GPU-h pilot (4k-step ASLC002 recipe: A0, T1f, T1-shuf) and check that T changes signer retention (same-signer kNN, decode) and raw R@1 in the predicted direction. If T does nothing at that scale, the J layer has no support and the proposal should be re-scoped to layer G plus data axis.

Also fix before approval (cheap): the morph NaN bug and re-run `morph_dose.py` (chk13's NaN-safe `morph` can replace the function), the §2.11 label, and the ARI definition.
