# R1 independent review: proposal17 SA-JEPA and CPU diagnostics

Reviewer: independent, adversarial. Date 2026-10-17. Scope: `SA_JEPA_Proposal_VI.md` (all), `diagnostics/*.py`, `diagnostics/results/*.json`, `literature/Q1-Q3` (use only), `method/phonojepa/pj/{model,losses,train}.py`.
Only train and val were touched. No GPU. No repo file modified. Scratch scripts: `/tmp/claude-1007/-home-haipd-Sign-Representations/32c5829a-820b-4c3b-9d20-e66b08713d70/scratchpad/review/chk1..chk11.py` (outputs next to them).

## (a) Verdict

The diagnostics are mostly sound. I found no test-split use and no val label or val statistic inside any fit that is called inductive. The retrieval metric, CORAL, the coral-head result, the ladder numbers and most of the section 2 tables reproduce. The main empirical story (a signer-specific alignment helps a lot, supervised linear geometry has large headroom) survives my re-runs. Three headline inferences do not survive, and the design has several flaws that would waste GPU time.

1. "Training does not amplify signer" is an artefact of using a variance-weighted eta2 on a random encoder that is extremely anisotropic. Scale-free measures show the opposite direction.
2. The NN-positive precision table, and therefore the "gate 40 % not reached" claim, was computed on a 12k-clip pool. On the full 40k pool the oracle-CORAL geometry already passes 40 %.
3. The label-free stop and KEEP rule (NMI(cluster, gloss) <= 0.2) fires on random clusters, so it is unusable as specified.

Additional problems:
- The forecast of 30-34 raw val double-counts the whitening gain.
- Module R is under-specified (masked student view, about 1.4 clips per cluster per batch, balancer frozen at warm-up).
- Cluster dissolution (the encoder removing the signal it clusters on) is not guarded against.
- Several hypotheses are not falsifiable as written.
- The controls named in H1/H2 are absent from the arm list.

Recommendation: do not approve GPU spend until findings 1-6 and the tests in section (d) are done. They are all CPU-cheap.

## (b) Findings

Severity: Critical = invalidates a headline claim or leaks labels/test. Major = materially weakens a claim or design. Minor = wording or precision.

### F1 (Critical). "JEPA training does not amplify signer variance" is not supported; scale-free measures say the opposite
- Location: proposal §0 item 5 bullet 2, §2.6 "Đọc [D]", §8 item 8. Code: `analyze_emb.py:31`, `ceilings.py` (eta2), `emb_variants.py:112`.
- What is wrong: `eta2()` is variance-weighted (sum of squares over all dims). The random-init encoder embedding is extremely anisotropic, so its eta2 is dominated by a handful of directions. Comparing 0.27 (random) with 0.14 (trained) compares unlike things.
- Evidence (my run, chk5, train half, `grp_fin`):

  | Statistic | Random encoder | Trained teacher |
  |---|---:|---:|
  | participation ratio of total covariance | 7.6 | 83.5 |
  | eta2_signer, variance-weighted (as in proposal) | 0.270 | 0.138 |
  | eta2_signer in PCA-512 whitened coordinates | 0.023 | 0.029 |
  | 35-way linear signer decode, standardised, held-out clips | 88.4 % | 94.8 % |
  | same-signer top-1 NN (proposal's own number) | 72.4 % | 84.0 % |

  Every scale-free or retrieval-relevant statistic says training increased signer decodability by 1.3 to 12 points. The proposal's own kNN row already says so. The "random control" itself is fair as an init floor: `Encoder()` default init equals the init `PhonoJEPA` uses for the student. It is a single seed, though.
- Fix: delete the claim, or restate it as "training reduces the variance-weighted signer share but increases signer decodability". The Littwin/Sobal hypothesis is not refuted. Replace eta2 with whitened-eta2, linear decode accuracy, or kNN rate.

### F2 (Critical). Cross-signer NN precision table (§2.8) used a 12k pool; the 40 % gate is passed by the best geometry at full pool
- Location: §0 item 5 bullet 3, §2.8 table, §3.3 X1, §8 item 2, H10. Code: `signer_norm.py:69` (`n=12000`, neighbours searched only inside the 12k subsample).
- What is wrong: precision of "nearest cross-signer neighbour has the same gloss" depends on the pool size. At 12k there are about 4.4 clips per gloss, versus 14.7 in the real gallery.
- Evidence: I recomputed on the full 40,154 pool (chk2):

  | Geometry | top-1 (12k pool, JSON) | top-1 (full pool) | mutual precision (12k) | mutual precision (full) |
  |---|---:|---:|---:|---:|
  | raw | 9.8 | 14.3 | 19.3 | 25.9 |
  | whiten | 16.5 | 22.9 | 25.9 | 33.4 |
  | CORAL(signer) + whiten | 24.0 | 32.4 | 35.9 | **44.8** |

  - Full-pool raw 14.3 matches `nuisance_decomp.json` (14.4). Whiten 22.9 and 33.4 match `nn_head.json` signer round 1 (22.92) and signer_mutual (33.44).
  - So the table in §2.8 is internally inconsistent with two other JSONs. With the oracle-signer CORAL geometry the mutual precision is already 44.8 % > 40 %.
- Other problems with the gate:
  - H10 ("precision tương hỗ >= 40 %") has no pool, geometry or exclusion rule, so it can be satisfied by A0 plus the post-hoc stack today.
  - The 40 % threshold is labelled `[V]` in §2.8. `literature/Q2` lines 303 and 404 mark it `[I]` (inferred). It is also not the same quantity: the literature starting points of 51-61 % are kNN classification accuracies of the encoder, not mined-pair precision.
- Second weakness: the NN-head test used only whitened-geometry positives (precision 33 %), never CORAL-geometry positives. The head is also weak: gloss-oracle positives reach only 38.8 against LDA 50.9, and augmentation is only Gaussian noise (`nn_head.py` `noise=0.1`). Linear interpolation of its own curve gives about break-even near 45 % precision, so "NO-GO for a frozen-feature MLP head" is roughly right, but this says nothing about an encoder-level STARS stage.
- Fix: recompute §2.8 on the full pool for raw, whiten, CORAL-signer and CORAL-CLS-cluster. State pool size, exclusion rule and relabel the threshold `[I]`. Rewrite NO-GO as "head on frozen whitened features, 12k..40k pool".

### F3 (Major). Label-free monitor rule NMI(cluster, gloss) > 0.2 is broken by estimator bias
- Location: §5.2 line 287, §6.3 KEEP rule (line 366), §7 risk table. Related: gloss-label use in a training control rule.
- Evidence (chk11): on a 16,384-clip bank (as specified) with 2,731 glosses and K = 48:
  - CLS k-means: NMI(gloss) = 0.337, NMI(signer) = 0.714.
  - Random clusters: NMI(gloss) = **0.359**.
  - AMI(gloss) of CLS clusters = -0.015, AMI(signer) = 0.709.
  - So "stop if NMI(gloss) > 0.2" always fires, and "KEEP requires NMI <= 0.2" can never pass. The second clause (> NMI(signer)) happens to be harmless here only because signer NMI is large.
- Positive evidence kept: CLS carries essentially no gloss information. eta2_gloss(CLS) = 0.131, which is at the null value 0.136 for N = 20k, K = 2731. same-gloss NN = 0.03 % (chance level). So the leak risk is low, but the proposed monitor cannot show it.
- Also: a stop rule driven by gloss labels contradicts "method chính không dùng nhãn gloss" if it ever alters a run. It should be logging only.
- Fix: use AMI with a permutation baseline, report it, and make it log-only.

### F4 (Major). The "whitening is fragile" inference from `ladder_cv` is confounded
- Location: §0 item 6, §2.9, Z2, §8 item 3. Code: `ladder_cv.py`.
- What is wrong: CV queries are train signers that the encoder saw during SSL. Val signers are unseen. The whitening gain is plausibly larger for unseen signers.
- Evidence:
  - `ladder_cv.json`: whiten delta +1.41 mean (per split +2.07, -0.95, +3.07, +0.62, +2.23). The proposal rounds the means to +1.5 / CMN +5.5 / CORAL +9.3 / cluster +7.7. JSON deltas are 1.41 / 5.43 / 9.23 / 7.61.
  - CV raw is 16.2, below val 22.7, so "absolute R@1 is optimistic" (README, §2.9) is wrong in direction.
  - I re-ran with held-out queries from heavy signers (>= 2400 glosses, balanced coverage, chk10): whiten delta +0.8 (-2.0 to +2.3), CORAL minus whiten +8.7. So the CORAL robustness replicates, and the whiten smallness replicates, but nothing separates seen from unseen signers.
  - On val the per-signer whiten gains are all positive, 4.6 to 8.3. Signer-level t CI with 5 signers: +6.5 [4.6, 8.4].
- Fix: write "whitening gain is +6.7 on unseen val signers and +1.4 on seen-signer CV; the CV cannot adjudicate". Train an SSL model without some signers if this matters (GPU), or drop the "mong manh" wording.

### F5 (Major). "Signer is a per-signer linear map" is stated as fact; alternatives not excluded
- Location: §0 item 2, §2.3 title, §3 synthesis (i). Code: `signer_norm.py:44` (`coral`).
- What I verified: the formula is correct (toy, chk1: residual covariance difference 1e-6). The gain is real: per-signer t CI for CORAL minus whiten is [6.9, 13.2]. It is signer-specific: label-shuffled coral head 24.59 against whiten-init 24.94, and iso-only 25.06 (reproduced, see section (c)).
- What is not shown:
  - CORAL+whiten is algebraically close to per-signer whitening (global colouring then global whitening cancel). I ran per-signer whitening without colouring (chk3): **41.5** against CORAL 38.5. So the effective operation is generic per-domain whitening. This fixes per-domain anisotropy and hubness; it does not show that signers are linear maps of a shared latent. Per-signer isotropic scale and centring gives 31.65, equal to CMN, so scale heterogeneity alone is not it.
  - Gloss-coverage heterogeneity is large and unexamined. Train signers sign between 3 and 2,728 distinct glosses (nine "heavy" signers hold 66 % of clips). Per-signer means and covariances of low-coverage signers contain gloss-subset structure, and the `m.sum() < 200` branch degenerates to mean-only. Control (chk3): gallery restricted to the 9 heavy signers gives whiten 27.5, CMN 29.2, CORAL 34.05, so CORAL minus whiten shrinks from +9.1 to +6.5 and CORAL minus CMN from 6.9 to 4.8. The signer effect survives but about 30 % of the headline is composition or sample-size dependent.
  - The head ablation ranks cov-only (29.3) above mean+cov (28.1) while the mean term is left unminimised. This is hard to square with "signer = full linear map".
- Fix: say "per-signer second-order whitening removes most of the signer-attributable nuisance"; keep the mechanism as a hypothesis. Add the heavy-signer gallery table.

### F6 (Major). "Label-free" is used where signer or gloss information enters (inventory)
1. §2.1 row 5 and §2.3: query clustering uses K = 6, the true number of val signers (`geo_ladder.py:31`, `ladder_cv.py:55`). With K = 12/24/48 the gain vanishes: Kq = 24 gives 29.5 against whiten 29.4 when the gallery side is also clustered (`emb_signer_cluster.json`). A label-free K estimator is required for Z3.
2. §0 item 3 and C module: "CORAL on CLS clusters = oracle (32.3 vs 31.7)". That is the best of 6 (Kg, Kq) combinations on val; the mean of the six is 30.65, below the oracle, and K = 140 gives 28.5/29.4. Gallery K = 35/70 and the choice of K = 48 come from the signer count (35) and signer-label AMI.
3. §0 item 3 calls CLS a "gần như hoàn hảo" signer fingerprint. Train AMI 0.71 is measured on clips the encoder was trained on. Val AMI is 0.44 (purity 0.73). CLS same-signer NN 96.6 % is within the seen train half. No unseen-signer figure is given for it.
4. §5.2 stop rule uses gloss labels (see F3). K selection uses signer-label AMI.
5. §5.6 "whitening fit không nhãn": k = 512 and power are chosen by R@1 on a labelled inner hold-out (`geo_headroom.py`), at the edge of the grid.
6. §2.1 row 2 "quy nạp, không nhãn" (local-mean k = 200): val-selected. `local_norm.json` has 12 configurations; rung 2 is the maximum, 30.41 without whiten. With whiten the same setting gives 29.86 and the best is 30.22; no CI separates it from whiten ([-0.2, +2.3]).
7. Z1 "hands only, free": chosen on val among 11 channel combinations (`channel_analysis.json`). The advantage is +2.4 raw but only +0.6 after whiten (ceilings: 25.5 against 24.9).
8. §6.2/§5.2 "signer id only at arms with suffix m": the economy plan drops R1m and T1m, so the diagnostic arms promised in §6.3 (iii) do not exist (see F10).

### F7 (Major). Style-variable and morphology evidence is too weak to justify "NO-GO retarget" (V1, E-canon)
- Location: §0 item 5 bullet 1, §2.5, §8 item 1. Code: `style_vars.py:184-191`, `nuis_pc.py:26`, `emb_variants.py` `morph`, `morph_effect.py`.
- (i) The "R^2 about 0 on held-out signers" regression predicts a 768-d within-gloss residual from 30 clip-level variables using GroupKFold over signers. That is 28 training signers per fold for the signer-level mapping (35 signers in 5 folds), a power failure by construction. Held-out-clip R^2 on the signer component is +41.5 %, and the held-out-signer R^2 is -17.9 %. The aggregate -1.7 % is a mean over positive PCs (PC1 +45 %, PC3 +37 %) and strongly negative ones (PC4 -55 %). One shared ridge alpha, 768 outputs, no per-direction selection.
- (ii) The face perturbation is a no-op by construction (cos 0.9999996). Face tokens are eye-frame normalised and hands do not depend on face size, so "all" includes a vacuous component. Only arm and hand scaling (+-15 %, clip-constant, global factor) are tested. The implementation is correct: shift norm 0.29 and the shift is in the signer subspace (23-27 % energy against 4.4 % random).
- (iii) Subset: the first 1,500 of every-2nd val clips are P21 52 %, P12 23 %, P5 20 %, P39 4 % (59 clips), P26 absent, P13 one clip (chk1). The clean R@1 on this subset is 12.3 against 18.3 for the half. The ids are asserted aligned (`morph_effect.py` assert), so there is no alignment bug.
- (iv) No dose-response (+-30 / 50 %), no non-uniform proportion changes, no paired noise estimate. R@1 +1.1 is described as "no change" but is not tested for significance.
- (v) The proposal's own text says the shift is "cùng hướng nhưng quá nhỏ". That argues for a bigger dose, not for "no analytic retarget".
- Also: `signer_id_acc_from_embedding` = 90.7 % uses unscaled L2-normalised features (`style_vars.py:161`). With standardisation I get 94.7 % against 90.1 % in the same setup (chk4), so "style variables decode signer as well as the embedding" is biased towards parity.
- Fix: downgrade V1/E-canon to "untested at adequate dose", add the dose-response and a signer-balanced subset.

### F8 (Major). Forecast double-counts the whitening gain
- Location: §6.5, H2. Evidence: the coral head gives +6.1 over whiten on the full set (29.4 to 35.5; +3.2 on the subset). The proposal says internalising "half to two thirds of +6" gives val 30-34 raw, which is +7 to +11 over A0 raw. Half to two thirds of +6.1 is +3 to +4 over a whitened baseline. Over raw that is about 25.7-26.7, unless the encoder also internalises all of the whitening gain (+6.7 on val, +1.4 in CV).
- With a realistic 26-27 val the test extrapolation (x0.73) is 19-20, not 22-25. H2's threshold `A0 + max(tau; 3)` is consistent with +3, but the text forecast is not.
- Fix: state the forecast in terms of "head gain over whiten" and "whiten gain internalised", with the second as an explicit unknown.

### F9 (Major). Module R, as specified, is not implementable as the diagnostic suggests
Checked against `pj/train.py`, `pj/model.py`, `pj/losses.py`.
- (i) What R acts on: `h_s` (student tokens) on masked views, pooled over visible tokens. In the log the near view hides about 50 % and the far view about 12 %. The retrieval embedding is the EMA teacher with all tokens visible, group-mean per window then averaged over windows. Statistics on masked, window-level student pooling are a different object, and the proposal does not say which view (near/far) is used. Mean-over-windows of clip embeddings also changes the covariance. Mitigation: use the far view, or distil alignment onto the teacher full-view pooling.
- (ii) Batch statistics: 32 clips per batch, K = 48. With real cluster sizes, a batch has 22.5 distinct clusters and 7.2 clusters with >= 2 clips, 1.4 clips per present cluster (chk11 simulation). The batch part of Sigma_k (weight alpha = 0.25, gradient only there) is rank <= 1-3. The covariance gradient is scaled by alpha^2 = 0.0625 and is essentially a per-clip rank-1 outer-product pull. The mean part is fine. `a_Sigma = 1` should therefore not be expected to behave like the head experiment, which used full-batch statistics over the 20-40k train set.
- (iii) Balancer: `LossBalancer` freezes each term's divisor after `int(0.1*steps)`. L_R is zero-ish or huge at initialisation (EMA covariances from untrained embeddings) and changes by orders of magnitude later. A frozen divisor mis-scales it for the rest of training. The "10-20 % of loss" target cannot be guaranteed.
- (iv) Isotropy target I is arbitrary relative to the scale of group-mean pooled LayerNorm outputs; use a correlation target or normalise first.
- (v) Cov-only beat mean+cov in the head, and the defaults `a_mu = 0.25`, `a_iso = 0.25` were never run in any diagnostic (head used 1.0/1.0/1.0 and iso weight 1.0).
- Fix: specify the view, a minimum-samples-per-cluster rule (or use only mean + diagonal/low-rank covariance), a moving normaliser instead of a frozen divisor, and a unit test that L_R decreases on real A0 embeddings.

### F10 (Major). Module T/C feedback and control coverage
- Cluster dissolution: T removes the reward for encoding the signer offset in tokens, and R equalises token statistics. The CLS is trained only through the predictor. Once the predictor no longer needs signer information, the CLS may stop encoding signer and the clusters dissolve (the trivial way to lower L_R and the T penalty is to make clusters random). The CLS-signer property was measured on a checkpoint trained without T/R, and not at 10/40 epochs (the screening budget). The proposal monitors AMI(cluster, signer) only for diagnosis, with no mitigation. Mitigation: freeze clusters from the existing A0 CLS for the first screen, or add a warm start and an alarm when AMI(cluster, signer) drops.
- At initialisation the random-encoder CLS has eta2_signer 0.36 and same-signer NN 64.6 % (`emb_analysis.json`), so early clusters are input-statistics clusters.
- Mismatch of diagnostic and mechanism: CMN +2.2 (oracle signer, pooled clip embeddings, post-whiten) is the evidence for T; T acts on token-level targets with per-cluster EMA centres from 1-2 clips per step. The CLS-cluster = oracle result is for CORAL only. No diagnostic tests CMN with CLS clusters.
- Controls named in hypotheses but absent from the arm list (§6.1): H1 lists T1-shuffle, H2 lists R1-shuffle; §6.1 contains only TR-shuf. The economy variant (§6.2) drops T1m and R1m, so a null TR cannot be attributed to cluster quality versus a false hypothesis. G-SA then sends everything to Plan B without discriminating.
- Summary inconsistency: §0 says "~11 arm", §6.2 says "~13", §6.1 lists 13 trained arms plus B*.
- Fix: add T1-shuf and R1-shuf (cheap), keep T1m or R1m in the economy plan, add the frozen-cluster arm.

### F11 (Major). Hypothesis and decision-rule defects
- H1: confirm at `raw >= A0 + tau` (tau >= 1.5), refute at `Delta raw < 0.5`. The interval 0.5 to 1.5 is undefined.
- H3: equivalence margin |Delta| <= 1 between two single-seed runs. tau is 2.5 x SD_seed(A0) with SD from 3 seeds (relative SE about 50 %), and the SD of a difference of two single-seed runs is about 1.4 x SD_seed. A +-1 margin is likely below noise, so TOST cannot succeed.
- H4: probe mean-of-4 shift of +3 with CI excluding 0; the per-parameter probe CIs are +-3-6 (gloss bootstrap only, unpaired, no signer clustering).
- H5: A0 is flat to 160 epochs by the proposal's own admission, so "D1 at 160 >= D1 at 40 - 0.3" is satisfied by A0 and does not discriminate D1. The "plateau evidence" covers only 160 to <= 180 epochs (`report.json` steps 205,640-225,640, about 164-180 epochs), not "~200"; Bootleg degrades at 600-1,200.
- H7 threshold (slope >= 1.5 per doubling signers) is derived from the gallery-signer curve, which is a different manipulation (see F12).
- H9 thresholds (2, 3) are arbitrary.
- Val has 3.8 effective signers (Kish: 10,304^2 / sum n^2). The rule "at least 4 of 5 signers improve" has a one-sided null probability of 6/32 = 0.19; 5/5 gives a two-sided sign-test p = 0.0625. The 1000-resample bootstrap CI is not wrong (I compared the code's set-based gloss resampling with a multiplicity bootstrap and signer-only: widths agree to 0.3 points), but with 5 clusters it is anti-conservative; my signer-level t intervals (df = 4) are used above.
- Everything (readout, K, hyper-parameters, arms) is tuned on the same 5-signer val, then one test shot. Declare val selection optimism.

### F12 (Major). Gallery curves are mis-cited and mis-used for H7
- Location: §2.2 gallery curve, §3.3 Y1, H7. `geo_headroom.json`.
- Y1 says "R@1 +about 2 per doubling of signer count". Data: 8 to 16 signers +4.0, 16 to 35 (1.13 doublings) +3.8, about 3.4 per doubling.
- The signer subsets in the gallery curve also change clips per gloss. The clips-per-gloss "16" point is the full gallery (22.706 against 22.719), so the 8 to 16 step is not a doubling of the real gallery.
- The gallery-signer slope (references available at retrieval) is not the pretraining-signer slope that H7 tests.
- Fix: relabel as retrieval-gallery effect; rederive H7 expectation from literature only.

### F13 (Major). The LDA ceiling is not evidence for "information present, geometry wrong" as stated (but the conclusion survives a better test)
- Location: §0 item 1, §2.1 row 7. Code: `geo_headroom.py` `fit_lda`, `geo_ladder.py`.
- LDA is fit on the same 2,731 glosses it is scored on, with the gallery in-sample. The headroom ratio LDA/raw is larger for the random encoder (21.2/4.2 = 5.0) and the input tokens (12.9/3.4 = 3.8) than for the trained encoder (40.1/18.3 = 2.2). So "headroom" is generic to supervised versus unsupervised readouts. The LDA grid best is at the lower edge (k = 96, grid starts at 96), so the ceiling is not even tuned (k = 64 is higher in my run).
- Test that does support the claim (my run chk8, held-out 819 glosses, gallery and queries restricted to them, partition seed 0 as the probe):

  | Readout | R@1 |
  |---|---:|
  | raw | 35.9 |
  | whiten | 43.8 |
  | LDA fit on the DISJOINT 70 % glosses (k = 64 / 96 / 128) | 66.0 / 65.4 / 65.3 |
  | LDA fit on the same 819 glosses (proposal-style) | 68.0 |

  A gloss-agnostic linear map recovers 94 % of the same-gloss ceiling. So the headroom is geometric and generalises, which is stronger than the proposal's evidence. Add this table and drop the old argument.

### F14 (Major). Probe, literature and report comparisons
- D-LEX v3 comparison (§2.6 probe table, vs `proposal16/diagnostics/results/probe_v3_results.json`): all nine "descriptor thô tốt nhất" numbers match the best of nine descriptors per parameter. Protocol: same 70/30 gloss partition seed 0 (819 test glosses both), same PCA 256, logistic C = 0.05, balanced accuracy, same 14,0xx train clips (14,016 vs 14,000). Differences: val queries are every-2nd clip (1,527 against 3,086), bootstrap 300 against 500 resamples. It is protocol-matched in the way that matters (point estimates). The CIs are wider, gloss-only, unpaired.
- "Encoder beats random encoder on every motion parameter" is not tested with paired CIs. Movement 35.7 [30.4, 41.7] against random 27.6 [23.1, 32.8] overlap. Selecting best-of-9 raw descriptors per parameter makes the baseline slightly optimistic, which is conservative for the claim.
- The proposal text is unchanged by the above except the Movement claim (call it "suggestive").
- Literature use:
  - ContentVec "No-Cond 5.13 -> 5.76" (§3.2 C-pred) is ABX-within; the ABX-across used in the T row would be 6.32 -> 7.08.
  - Bootleg D1: concatenation vs data2vec-style averaging is +2 in Table 3, with "every 4th layer -1", which is closer to the proposed blocks {2,4,6}; this nuance is omitted.
  - 40 % gate: `[I]` marked as `[V]` (see F2).
  - "tiếp tục train 160 to ~200 epoch": see F11 H5.
  - Other figures I spot-checked match: ContentVec 6.32/6.92/7.78, Bootleg 61.0 to 43.8, V-JEPA 2.1 62.5 to 72.1, seq-JEPA 87.4 to 79.3, IWM 74.5 vs 67.5, MASA k = 3 49.06 / k = 1 47.64, OpenHands 91.2 to 94.7, GFP 62.0 to 70.9 (the last only in a summary line).

### F15 (Minor). Statistical and arithmetic details
- eta2_gloss and "75.4 % within-gloss variance" (§0, §2.2) are in-sample and upward biased in gloss share: null eta2_gloss with shuffled glosses is 0.068 at N = 40,154 and 0.136 at N = 20,077 (chk3). The §2.6 table values 0.24-0.32 therefore contain 0.136 null; differences between rows are valid, absolutes are not.
- CV deltas in §0 item 6 and §2.9 are computed from rounded means: JSON gives whiten +1.41, CMN +5.43, CORAL +9.23, cluster-CORAL +7.61 (proposal +1.5, +5.5, +9.3, +7.7). LDA +25.79 is right.
- Numbers inconsistent between text and JSON: "~25 biến" (§2.5) vs "30 biến" (§0 item 5; the JSON has 30); LDA column of §2.7 is k = 256 (47.6) while §2.1 uses k = 96 (50.9) without saying so; per-layer "LDA 33-41" mixes k = 256 (32.9-36.6) and k = 96 (37.3-40.9) values.
- Error lift (§2.7) is not comparable across base rates. Base agreement ranges from 0.04 (Handshape) to 0.72 (WristTwist); the maximum possible lift for WristTwist is 1.38, so 1.04 is "11 % of the headroom", not "absent". Using (agree - base)/(1 - base): raw Movement 0.073, WristTwist 0.106, Repeated 0.128, FlexChange 0.167, Contact 0.270, MajorLoc 0.443, SignType 0.642. Movement is still lowest; WristTwist, Repeated and FlexChange are close to each other.
- 17 of 20,077 stride-16 train clips have all-zero `grp_fin` vectors (chk7); the stride-2 official gallery also has 39 zero vectors (`baseline.log`). Negligible but should be stated. The stride check "cos 0.9975" I measured as 0.9979 (train) / 0.9989 (val); it is not stored in any JSON.
- `morph_effect.json` contains `NaN` (face-mode shift energies), which is not valid strict JSON.
- Budget: the 40-epoch PLctx cost 1.01 h wall (14,580 s / 4 for 160 epochs), so 1.3-1.6 h per arm is realistic. Not budgeted: stride-2 full-gallery extraction for 3 checkpoints x 16 arms, pose extraction for MS-ASL/WLASL (Phase 4), CPU time of the stack evaluation.

## (c) Checked and found correct

1. `common.class_max_ranks`: rank = number of classes with score >= target (target included), so pessimistic ties and no off-by-one. An independent brute-force loop on 300 queries agrees exactly. Raw R@1 22.7193 / R@5 45.9433 / R@10 56.7838 reproduced.
2. `signer_norm.coral`: formula correct (toy covariance residual 1e-6). `geo_headroom.fit_leace` matches the LEACE closed form for row vectors. `fit_lda` and `fit_pca_whiten` are standard. LEACE and CMN give the same total hit count (3,262 of 10,304) by coincidence, confirmed from per-signer counts.
3. No leakage found: whiten, LEACE, LDA and k-means for the gallery are fit on gallery only (also in `ladder_cv.py` after removing held-out signers from every fit). Transductive pieces (CORAL on queries, query k-means) are all declared. The test split is never opened (only `train.csv` and `val.csv` are read; `common.manifests()` returns train and val only).
4. `coral_head.py` is inductive: the loss and initialisation use train embeddings and train cluster ids only; evaluation takes the final step, with no val early stopping. My re-run (chk6, subset): true signers 28.144 (exact), signer labels shuffled preserving sizes 24.59 (whiten-init 24.94), isotropy-only 25.06. The head gain is signer-specific.
5. Bootstrap CIs: set-based vs multiplicity-weighted gloss resampling widths agree (raw CI width about 10.4 points; CORAL minus whiten [7.5, 11.3] vs [7.5, 11.3]); signer-level t intervals (df = 4): CORAL minus whiten +10.0 [6.9, 13.2], whiten minus raw +6.5 [4.6, 8.4]. Both exclude 0.
6. Morph perturbation: implementation matches its docstring (arm scaling about the shoulder, hand about the new wrist, face about the nose tip). The perturbation is applied before canonicalisation (`canonicalise` only mirrors dominant hand; the scale anchoring happens in `Fe.frame_features`). Ids for the first 1,500 clips are asserted equal between clean and perturbed files.
7. Probe split: train and val signer sets are disjoint (checked), glosses partition seed 0 with 819 test glosses; `D-LEX v3` best-descriptor numbers in the proposal match the JSON.
8. Section 2.1-2.4 and 2.6-2.7 numbers match the JSONs: ladder rows and CIs, per-signer rows, nuisance-decomp table, signer-norm and CLS tables, coral-head tables, probe table and the +x/+y differences, channel and layer numbers (note: LDA mixed-k remarks in F15).
9. Random control `emb_variants.load_encoders` uses the default init identical to the student init in `PhonoJEPA`; same pipeline, same masks.
10. Cross-checked reproductions: nn_head precision values (22.92 / 33.44) equal my full-pool whiten numbers; nuisance-decomp cross-signer top-1 (14.4) equals my full-pool raw number (14.3).

## (d) Cheap tests to run before approving GPU spend (ranked)

All CPU, each less than about 30 minutes.

1. **K-free cluster-CORAL**: estimate K on the query set with a label-free criterion (silhouette / eigengap / gap statistic) and re-run rung 5, Z3 and the CLS-CORAL result on val and on the 5 CV splits, reporting mean over K candidates rather than best. Decides whether any "label-free" claim in §0 item 3, §2.1 row 5 and Z3 stands.
2. **Full-pool NN-positive gate**: recompute §2.8 on the 40k pool for raw, whiten, CORAL-signer, CORAL-CLS-cluster (label-free exclusion), and fix the H10 definition (pool, geometry, exclusion, metric). Re-run `nn_head.py` with positives from the CORAL geometry once.
3. **Monitor repair and cluster stability**: replace NMI by AMI with a permutation baseline; measure CLS-cluster agreement (ARI) between two window offsets of the same clips; measure CLS AMI(signer) and kNN-signer on an early or pilot PLctx checkpoint (the 4k-step pilots under `runtime/runs`) as a proxy for epoch-10/40 behaviour of module C.
4. **Morphology dose-response on a signer-balanced subset**: arm/hand scale +-30 % and +-50 %, plus a non-uniform variant, 30-40 s per mode per 1,500 clips with the existing extractor. Determines whether E-canon and V1 stay NO-GO.
5. **R-statistic feasibility on real embeddings**: simulate batches of 32 clips with K = 48 CLS clusters, estimate Sigma_k from the batch-mix estimator with alpha = 0.25, and compare against the full-set CORAL transform in the L_R value, its gradient norm and the resulting sign agreement; report the initialisation-versus-late magnitude ratio of L_R using A0 and a pilot checkpoint (checks the frozen balancer).
6. **Composition control for CORAL and a mechanism arm**: add the heavy-signer gallery table (F5) and a per-signer-whitening arm to the ladder; add the held-out-gloss LDA ceiling (F13). These change the wording of §0 items 1-2 at no cost.

## Summary (12 lines)

1. Counts: Critical 2, Major 12 (F3-F14), Minor 1 consolidated list (F15).
2. Verdict: diagnostics mostly sound and reproducible; no test or val-label leakage found; three headline inferences fail; do not approve GPU spend until F1-F6 are addressed.
3. F1 (Critical): "training does not amplify signer" is an anisotropy artefact (random PR 7.6 vs 83.5); whitened eta2 0.023 vs 0.029, linear signer decode 88.4 to 94.8 %.
4. F2 (Critical): NN-precision table used a 12k pool; full pool gives raw 14.3 / whiten 33.4 mutual / CORAL-oracle 44.8 % mutual precision, so the 40 % gate (an `[I]`, mislabelled `[V]`) is already passed; H10 is ill-defined.
5. F3: NMI(cluster, gloss) stop/KEEP rule fires on random clusters (0.359 vs threshold 0.2); use AMI and make it log-only.
6. F4: ladder_cv does not show whitening is fragile (seen vs unseen signers confounded; CV raw 16.2 below val 22.7); the CORAL gain replicates (+8.7 on heavy signers).
7. F5/F6: "signer = linear map" is overstated (per-signer whitening alone gives 41.5; heavy-signer gallery shrinks the gain from +9.1 to +6.5); "label-free" hides K = #signers, best-of-grid selection and val-tuned choices (8 places listed).
8. F7: the "no analytic retarget" NO-GO rests on a 28-signer regression, a vacuous face perturbation and an unbalanced 1,500-clip subset.
9. F8/F9/F10: forecast 30-34 double-counts whitening (realistic about 26-27); module R has batch stats from about 1.4 clips per cluster and a frozen balancer; cluster dissolution unguarded; T1-shuf/R1-shuf controls missing from the arm list.
10. F11/F12: several hypotheses not falsifiable (H1 gap, H3 margin below noise, H5 non-discriminating); gallery-signer curve misquoted (+2 vs about 3.4 per doubling) and misused for H7.
11. F13 (positive): a supervised linear map fit on disjoint glosses reaches R@1 66.0 against 35.9 raw and 43.8 whiten on held-out glosses, which supports the "geometry" claim better than the stated evidence.
12. Reproduced exactly: raw R@1 and ranks (brute-force), CORAL formula, coral-head 28.14 with shuffled-label control 24.59, per-signer t CIs excluding 0; report at `/home/haipd/Sign_Representations/docs/proposal17/reviews/R1_independent_review.md`.
