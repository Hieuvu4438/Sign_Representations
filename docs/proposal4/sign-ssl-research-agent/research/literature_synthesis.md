# Literature synthesis (T01, cycle C001)

Date 2026-10-05. Scope: 52 notes (P101–P115 sign SSL/pretraining, P201–P215 video SSL mechanisms, P301–P322 evaluation/data/phonology), queries in `search_log_A.md`, `search_log_B.md`, `search_log_C.md`. Coverage is **provisional**: one search round per family, no forward-citation snowballing round yet (blind spots listed at the end). Reported numbers below are REPORTED_RESULT and come from different settings; they are not a ranking.

## 1. What information each family keeps (input → target → inference)

| Family | Examples | Pretraining target | Inference input | What is structurally lost |
| --- | --- | --- | --- | --- |
| RGB masked video + pose-derived targets | SignRep (anchor), SignMAE (P113) | Per-frame geometric priors (angles, keypoints, distances, hand activity) decoded from **one pooled 16-frame vector** (SignRep §4–5; vendored `dict_head_v6`) | RGB only | Context beyond 16 consecutive native frames (≈0.5 s); whole-sign order (bag of windows at readout, C005) |
| RGB pure MAE | SSVP-SLT (P114), SSVT-ISLR (P115), Hiera/VideoMAE (P208) | Pixels | RGB | Sign-specific geometry unless learned from pixels; SignRep Table 4: HieraMAE features R@1 0.25–0.39 on ASL Citizen frozen retrieval |
| Crop-feature contextual models | SHuBERT (P101), SignDino (P102) | k-means IDs of DINOv2 hand/face/body streams (SHuBERT); DINO self-distillation long→short temporal crops + Gram anchoring (SignDino) | Hand/face crops + pose pipeline + DINO extractors | Full-frame context; depends on tracker/detector |
| Pose SSL | SignBERT/+ (P105/P106), BEST (P107), MASA (P108), STC (P104), SSL-SLR (P103) | Masked joints / motion residuals / tokens / consistency | Pose only | Appearance detail (mouth/face/handshape beyond keypoint accuracy) |
| Text/gloss supervised | SignCLIP (P110), Sigma (P109), Uni-Sign (P111), SignSeek (P112) | Text or gloss alignment | Pose (mostly) | — (different supervision class; not SSL-only comparators) |

## 2. Temporal granularity and readout

- Every surveyed method uses an **order-insensitive clip readout** (mean / weighted mean / attentive pooling) for isolated-sign evaluation (C007). SignRep's readout averages independent 16-frame windows weighted by predicted hand activity (C005).
- Evidence that longer context matters exists but is indirect: SignDino ablation (P102 Table 4) — removing the long-window teacher drops ASL Citizen R@1 0.704 → 0.65 (full fine-tuning regime); V-JEPA (P201 Table 3) loses heavily under plain average pooling vs attentive probe (73.7 → 56.7 K400); V-JEPA 2 cooldown at 64 frames (P202). None tests a frozen sign retrieval embedding.
- SignRep authors list the 16-frame window as a limitation (v1 App. A8).

## 3. Supervision and evaluation settings are frequently mixed

- Label-free sign SSL papers report trained heads on ASL Citizen (SHuBERT LoRA ≈0.65 R@1; SignDino frozen+linear 0.63; SSL-SLR linear on 500 classes) — not frozen dictionary retrieval (C006). Only SignRep (label-free) and SignSeek (gloss-supervised, pose) report frozen retrieval (SignRep 49.95 R@1, SignSeek 57.66).
- Cross-setting tables in recent work: SSL-SLR Table 6 places fine-tuned numbers next to SignRep frozen retrieval (P103); SHuBERT cites SignCLIP's fine-tuned+linear number (P110).
- Gallery design changes results (P311: per-gloss centroid gallery 0.6604 vs all-instance 0.6111 Rec@1 for an ST-GCN), so the gallery rule must be fixed and identical across compared encoders.

## 4. Linguistic structure and diagnostics

- ASL Citizen glosses come from ASL-LEX 2.0 (2,723 codes); 2,725/2,731 glosses join to `signdata.csv` (P302, verified locally this cycle). Fields: handshape, selected fingers, flexion, major/minor location, path movement, repetition, sign type, contact, wrist twist; **no palm orientation**; codes describe the initial morpheme.
- Phonology probes trained on the official split leak sign identity (every test sign is seen in training; P302/P301), so linear "handshape probe" scores are not clean evidence. Minimal-pair contrasts avoid this (P312 retrains models without the test-pair signs; it found pose models favour handshape contrasts, pixel models location; no SSL model tested).
- Non-manual information: NMFs-CSL (P315) is the only dedicated benchmark; SignRep explicitly excludes non-manual priors (App. A8).

## 5. Nuisance and signer variation

- SignRep adds an adversarial "style" loss on Gram matrices; crop-based methods discard background. No surveyed paper runs a controlled signer-shift or background test of a frozen embedding (P1xx notes). Signer identity is recoverable from pose (P316); per-signer accuracy varies widely (ASL Citizen I3D 43–75%, P301 §6).

## 6. Pose-teacher reliability

- Noise handling is thresholds/masks everywhere (SignRep >50% confidence; MASA 0.4 cut; SignDino validity mask; SHuBERT interpolation; SignBERT+ confidence weighting). Hands are the weakest whole-body part (P318/P319/P320). The local ASL Citizen keypoints are 2-D COCO-WholeBody of unknown provenance with unnormalised scores — a different teacher from SignRep's 3-D lifted MediaPipe (P317 summary; local audit).

## 7. Data provenance

- MS-ASL test overlaps YouTube-ASL (38% of test samples, P307 local check) → any YouTube-pretrained model (SignRep's YT-SL-25 included) is at contamination risk on MS-ASL; WLASL splits are not signer-independent and labels are noisy (P306/P308). SignRep selected checkpoints on **WLASL validation retrieval** (v1 App. A3), so WLASL is a selection-contaminated benchmark for SignRep. ASL Citizen is signer-disjoint and was not used by SignRep for selection (as far as v1 states) → best primary benchmark here; contamination with YT-SL-25 remains UNKNOWN (no ID list).

## Contradictions / tensions

- SignDino's authors say their frozen variant does not exceed SHuBERT, but SHuBERT's ISLR uses LoRA — different regimes (P102).
- V-JEPA shows averaging hurts latent-prediction features, while SignRep's activity-weighted averaging is the anchor's best readout — the cost of averaging is architecture/objective dependent; must be measured on SignRep.

## Blind spots (to close before any novelty claim)

- Forward citations of SignRep beyond SSL-SLR/SignSeek; SignMatch (2609.01886), "Rethinking JEPA with frozen teachers" (2509.24317), PhonSSM (2604.08761), MSLU, OneSign: DISCOVERED only.
- Arrow-of-time / temporal-order pretext tasks in video SSL; VideoMAE v2; DINOv3 feature spreading — not searched/read.
- Several P3xx numbers came from model summaries and must be re-checked against PDFs before entering the claim ledger (flagged in the notes).
