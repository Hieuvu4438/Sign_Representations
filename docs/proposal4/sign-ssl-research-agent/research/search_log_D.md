# Search log D — Closest prior art for H1 (label-free temporal encoder on frozen windows) and H2 (cross-window latent prediction continued pretraining)

Owner: literature agent (family D). All queries were run 2026-10-05 with WebSearch (US index) unless noted.

Primary sources were fetched with `curl -sL` from arxiv.org/abs and arxiv.org/html. They were stripped to text with a local python script that keeps `<math alttext>` so table cells survive. All numbers in the P40x notes were taken from that raw text. No WebFetch summaries were used for numbers.

Tooling caveat discovered during the session: the shell `grep` is ugrep. Patterns with long bounded repeats (e.g. `.\{120\}`) silently return nothing. All negative keyword checks recorded below were therefore redone with a python regex scanner. Scratch copies live in the session scratchpad and are temporary.

Notes written: research/papers/P401–P406. Existing notes reused for the negative checks: P102 (SignDino), P112/P314 (SignSeek), P201 (V-JEPA), P203 (DPC), P206 (TCA), P312 (Phonological Perception).

Scope: (a) temporal aggregation and SSL on frozen clip or frame features; (b) time-direction / arrow-of-time sensitivity of video representations; (c) frozen-teacher and future latent prediction; (d) sign-language SSL or retrieval work with any temporal-order component. Time range: 2018–2026-10.

## Fetches of named papers

| # | Date | Where | Item | Result | Read level |
| --- | --- | --- | --- | --- | --- |
| F1 | 2026-10-05 | arXiv abs + HTML v2 | 2211.17042 SCALE | P401 | FULL_READ |
| F2 | 2026-10-05 | arXiv abs + HTML v2; project page bpiyush.github.io/lift-website (venue NeurIPS 2025, code link) | 2509.08502 Chirality in Action / LiFT | P402 | FULL_READ |
| F3 | 2026-10-05 | arXiv abs + HTML v2 | 2605.15618 Latent Video Prediction for World Modeling (NeurIPS 2026 E&D per arXiv comment) | P403 | FULL_READ main + App. A, B.1–2, B.6, C, J |
| F4 | 2026-10-05 | arXiv abs + HTML v1 | 2509.24317 Rethinking JEPA / SALT | P404 | FULL_READ |
| F5 | 2026-10-05 | arXiv abs + HTML v1 | 2609.01886 SignMatch | P405 (direct threat to H1 framing) | FULL_READ |
| F6 | 2026-10-05 | arXiv abs + HTML v1 | 2609.03695 SignSeek: temporal-order component check | No new note (P112/P314 exist). Temporal fusion is a Conformer plus attentive pooling over pose streams. Supervision is gloss-contrastive with MAP (masked *articulator* latent prediction, not future). A python scan for revers/arrow of time/temporal order/frame order/shuffl/permut/chiral/order-aware returns 0 hits | targeted method read (§3) |
| F7 | 2026-10-05 | arXiv HTML v1 | 2609.06296 SignDino: does it measure time reversal or order sensitivity? | No. The same python scan returns 0 hits. Its ISLR head is "stream-fused, mask-aware time-average + BatchNorm + Linear" (§4.3), which is order-invariant. Its Gram anchoring constrains the frame-to-frame similarity structure. No kNN/retrieval eval (all heads trained). Existing note P102 is confirmed | re-checked raw text |
| F8 | 2026-10-05 | arXiv abs + HTML v1 | 2506.05543 FRAME (surfaced by query 11) | P406 | FULL_READ main paper |

## Queries

| # | Date | Where | Query | Kept | Excluded (reason) |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026-10-05 | WebSearch | sign language time reversal representation embedding | none new | REWIND 2505.20756 (speech speaker time reversal; other modality). 2510.19413 (WMT-SLT 2022 system; abstract checked, no reversal). Progressive Transformers / SLT papers (production/translation). SignDino (already P102) |
| 2 | 2026-10-05 | WebSearch | sign language movement representation self-supervised RGB video arXiv | none new | All hits already noted: SSVT-ISLR (P115), SignRep, SignMAE (P113), SignDino (P102), SHuBERT (P101), SSL-SLR (P103), Spatial-temporal consistency (P104) |
| 3 | 2026-10-05 | WebSearch | hold movement hold model computational sign recognition embedding | DISCOVERED: "Modeling Global Body Configurations in ASL" 2009.01468 (Movement-Hold background; not opened) | Hybrid-transformer ASL (Sci. Rep.), subunit HMM 2009 (Pattern Recognition Letters), signature-embedding papers (no sign-SSL relevance). No hit links a Movement-Hold structure to a learned SSL embedding |
| 4 | 2026-10-05 | WebSearch | sign language phonology movement self-supervised video representation | none new | P115, P102, P104, P113, P101, P321 (all existing) |
| 5 | 2026-10-05 | WebSearch | temporal order aware sign embedding dictionary retrieval isolated sign | SignMatch → F5/P405 | SignSeek (existing), GLE-Net (P315), "Watch, read and lookup" 2010.04002 (dictionary spotting background, cited by SignMatch as [27]), TEMPS (text IR), SIGNER (generation) |
| 6 | 2026-10-05 | WebSearch | frozen clip features temporal transformer sign language self-supervised aggregation | none new (SignDino P102). GitHub junyi2005/signdino exists (not audited) | AdaptSign 2404.08226 (frozen CLIP + supervised CSLR adapters; supervised). "Frozen CLIP models are efficient video learners" 2208.03550 (supervised action). X-MIC, PiSLTRc (supervised) |
| 7 | 2026-10-05 | WebSearch | future latent prediction sign language video JEPA | none in the sign domain. DISCOVERED: "Factorized Latent Dynamics for Video JEPA" 2605.17165 (abstract read: appearance/dynamics subspace auxiliary loss, UCF/SSv2/IN-100; no reversal or sign). TAP-JEPA 2606.00662 (abstract read: frozen V-JEPA 2.1 predictor features for EK-100 anticipation; supervised probes) | VLA-JEPA, JEPA-WAM (robotics). Medium/LinkedIn posts (non-scholarly) |
| 8 | 2026-10-05 | WebSearch | chiral reversible actions sign language video temporally opposite signs | only P402 (and its NeurIPS mirror). DISCOVERED: Re²TAL 2211.14053 (reversible = invertible networks; unrelated meaning) | None in the sign domain |
| 9 | 2026-10-05 | WebSearch | sign language recognition reversed video augmentation temporal flip directional verbs | none | All hits are supervised CSLR/ISLR with standard augmentations (flip, temporal scaling). REVEAL 2602.11244 (VLM stress tests; not sign). SignNet-1M (dataset). No sign paper using time reversal as negative or positive found |
| 10 | 2026-10-05 | WebSearch | arrow of time sensitivity frozen video embeddings retrieval self-supervised temporal head 2026 | DISCOVERED: "Tracing the Arrow of Time" 2605.07568 (Video-LLMs; snippet only). "Made to Order" 2404.16828 (abstract read: shuffled-order prediction to find monotonic change). "Learning and Using the Arrow of Time" (Wei et al., CVPR 2018; via SCALE refs). ArrowGEV 2601.06559 (video grounding). "Reversed in Time" benchmark (video-text retrieval; ResearchGate, not opened) | Temporally Coherent Embeddings 2004.02753 (older SSL; not opened) |
| 11 | 2026-10-05 | WebSearch | predict frozen teacher embedding of future clip from current clip self-supervised video representation continued pretraining | FRAME 2506.05543 → F8/P406 | V-JEPA topic pages (secondary). "From Static to Dynamic" 2603.26597 (image→video transfer; not opened). FROSTER, UMT (frozen CLIP teachers for supervised/VLM). 2412.10925 (JEPA video thesis-style; not opened) |
| 12 | 2026-10-05 | WebSearch | masked clip feature modeling on pretrained clip embeddings long video self-supervised kNN retrieval | DISCOVERED + abstract read: LV-MAE 2504.03501 (masked-embedding autoencoder on frozen segment embeddings of long videos; LVU/COIN/Breakfast; attentive/linear probes). It is the closest H1-like general-domain method besides SCALE | MVD 2212.04500, MaskFeat (in-backbone masked feature modelling). kNN-CLIP (segmentation) |
| 13 | 2026-10-05 | WebSearch | ASL minimal pairs movement direction reversed signs recognition model confusion handshape location movement | P312 already exists (2606.28667 Phonological Perception of Sign Language Models) | lifeprint/pressbooks (teaching material), ERP studies (psycholinguistics) |
| 14 | 2026-10-05 | WebSearch | sign language self-supervised pretext temporal order prediction shuffle verification isolated sign | none new | SignBERT+ (P106), SHuBERT, P104, P115: none uses an order-verification pretext according to the existing notes and this snippet set. Shuffle-and-Learn style order verification has not been found applied to sign SSL in this scope |
| 15 | 2026-10-05 | WebSearch | "time-aware" OR "temporally sensitive" sign language embedding retrieval reversal test 2026 | none | All hits are text temporal IR / RAG (off-domain) |

## Coverage statement and blind spots

- Within queries 1–15 plus the three sign-paper raw-text scans (F5–F7), no sign-language work was found that measures time-reversal or order sensitivity of embeddings, or trains a label-free order-aware sign-level encoder on frozen window features for retrieval. This is a statement about this search scope (WebSearch US index plus arXiv HTML of the named papers, as of 2026-10-05), not a novelty claim.
- The closest general-domain prior art: SCALE (P401) and LV-MAE (abstract only) for masked feature prediction on frozen clip features; LiFT (P402) for a label-free time-aware single vector from frozen frame features; P403 for the reversal measurement; SALT (P404), FRAME (P406), V-JEPA (P201) and DPC (P203) for frozen-teacher, future, or latent prediction.
- The closest sign-domain prior art: SignMatch (P405), an ordered LSTM over SignRep windows that is supervised; SignDino (P102), a label-free temporal encoder on frozen per-frame DINOv3 crops whose ISLR head is order-invariant and which has no retrieval or reversal measurement.
- Not done: forward-citation search of LiFT and SCALE (Semantic Scholar not queried); full read of LV-MAE, TARA (LiFT follow-up), "Made to Order", Retro-Actions (Price & Damen, ICCVW 2019), Ponbagavathi & Roitberg "nearly symmetric actions"; SignMatch supplementary and project page; SignDino code audit; the SignRep paper itself, re-checked for any order or reversal experiment (the existing notes say none, not re-verified here).
