# Search log A — Sign-language-specific SSL / pretraining methods

Date: 2026-10-05. Agent: literature sub-agent (family A). Engines: Claude Code `WebSearch` tool (US web index) for discovery; `curl` of arXiv `abs/` and `html/` pages + local HTML→text conversion for reading (numbers copied from raw table cells); one `WebFetch` pass used early (SHuBERT, SignDino) and then cross-checked against raw text.

Reading-method caveat: the WebFetch summariser produced at least one quote that does not exist in the source (a "temporal grammar of a sign" phrase attributed to SignDino; string not found in the HTML). All notes therefore rely on raw-text extraction, not on WebFetch summaries.

## Queries

| # | Query (verbatim) | Engine | Kept (→ note) | Excluded / deferred (reason) |
| --- | --- | --- | --- | --- |
| 1 | SignBERT pre-training hand-model-aware representation sign language recognition arXiv | WebSearch | 2110.05382 → P105; 2305.04868 → P106 | ResearchGate copy (duplicate); 2408.08544 MSLU (deferred: multimodal pretraining with text, other family); 2501.15187 (→ P111 via #6) |
| 2 | SignBERT+ hand-model-aware self-supervised pre-training sign language understanding TPAMI arXiv | WebSearch | 2305.04868 → P106; USTC publication page (venue check) | Bohrium mirror (duplicate) |
| 3 | BEST: BERT pre-training for sign language recognition with coupling tokenization arXiv | WebSearch | 2302.05075 → P107; AAAI OJS 25470 (venue) | ADS mirror; Awesome-Sign-Language list (index only); PatternRecog combinatorial benchmark (dataset, other family) |
| 4 | MASA motion-aware masked autoencoder with semantic alignment for sign language recognition arXiv | WebSearch | 2405.20666 → P108; GitHub sakura2233565548/MASA (not inspected) | aimodels/themoonlight/pith summaries (secondary); C2RL 2408.09949 (text-supervised SLT/retrieval, deferred); SignMAE 2605.02094 (→ P113 via #18) |
| 5 | SignCLIP connecting text and sign language by contrastive learning arXiv | WebSearch | 2407.01264 → P110; ACL Anthology 2024.emnlp-main.518 | 2509.10266 "SignClip" mouthing for SLT (name clash, different method); 2508.18183 (LLM SLT, other family) |
| 6 | Uni-Sign unified sign language understanding large-scale generative pre-training ICLR 2025 arXiv | WebSearch | 2501.15187 → P111; GitHub ZechengLi19/Uni-Sign | 2411.17799 Signs-as-Tokens (production); 2609.33090 OneSign (DISCOVERED, unified SLU, deferred); 2512.07273 RVLF (SLT) |
| 7 | sign language self-supervised video representation 2026 | WebSearch | SSL-SLR TMLR 2026 listing (venue for P103) | SignLang 2026 workshop page (Irish SL DINOv2 vs MediaPipe annotation support — DISCOVERED, not read); 2607.09611 real-time SLT (other task); 2509.14036 SSL-SSAW (question-based SLT); MDPI 2026 review (survey) |
| 8 | sign language pretraining Hiera masked video | WebSearch | ACL 2024 privacy-aware SLT (SSVP-SLT) → P114; SignRep (anchor, not re-noted) | 2406.06907 SignMusketeers (multi-stream SLT efficiency, deferred); 2606.22494 (survey-like); Hiera ICML paper (generic family) |
| 9 | sign language JEPA self-supervised | WebSearch | — (no sign-specific JEPA paper found) | I-JEPA, V-JEPA 2 pages (generic video SSL family, handled by other agent) |
| 10 | sign language masked video modeling pose targets self-supervised | WebSearch | confirms P101, P106, P108, P113 | 2110.05877 OpenHands (pose-pretrained ISLR, ACL 2022 — DISCOVERED, deferred); SJSU thesis (non-peer-reviewed) |
| 11 | dictionary retrieval sign representation frozen features isolated sign | WebSearch | 2609.03695 SignSeek → P112 | 2609.01886 SignMatch (dictionary→continuous matching, DISCOVERED); 2609.03788 reverse dictionary via captioning (DISCOVERED); Findings EMNLP 2025 "Investigating Dictionary Expansion for Video-based Sign..." (DISCOVERED); 2205.04152 sign spotting (older, other task) |
| 12 | SignRep self-supervised sign representations Wong Camgoz Bowden follow-up 2026 | WebSearch | Confirms SSL-SLR cites SignRep (Table 6) | Surrey repository PDF / Semantic Scholar / mlanthology ICCV entry (anchor metadata only) |
| 13 | "SHuBERT" sign language representation 2026 arXiv isolated sign recognition ASL Citizen | WebSearch | SignDino 2609.06296 → P102 | 2604.27232 Targeted linguistic analysis with minimal translation pairs (DISCOVERED; evaluation of SLMs, deferred); 2604.07606 Bootstrapping annotations with SLMs (DISCOVERED) |
| 14 | sign language self-supervised signer-invariant representation pretraining 2025 | WebSearch | — | M³-SLR MaxFlow MaskFeat multi-view ISLR (ResearchGate only; no arXiv/venue verified → DISCOVERED) |
| 15 | SignSeek transferable representations sign dictionary retrieval | WebSearch | 2609.03695 → P112 | CiCo 2303.12793 (sign–text retrieval, text-supervised); SignRR (production) |
| 16 | sign language foundation model self-supervised pretraining YouTube-SL-25 isolated recognition frozen linear probe | WebSearch | 2309.02450 → P115 | — |
| 17 | sign language pose self-supervised pretraining noisy keypoints confidence 2025 2026 | WebSearch | — (confirms P105/P106/P108/P111 handling) | 2609.14122 SignMimic (production) |
| 18 | SignMAE segmentation-driven self-supervised learning sign language recognition | WebSearch | 2605.02094 → P113 | awesomepapers mirrors |
| 19 | sign language representation learning temporal order shuffled frames isolated sign self-supervised 2026 | WebSearch | confirms SSL-SLR permutation evidence (P103 §5.2) | generic SSL survey 2301.05712 |
| 20 | sign language V-JEPA latent prediction video pretraining sign recognition | WebSearch | — (no sign V-JEPA found) | VL-JEPA, V-JEPA 2, facial-expression V-JEPA 2601.09524 (other family); SignVLM (CLIP-based ISLR, DISCOVERED) |
| 21 | self-supervised sign language pretraining phonological handshape location movement representation arXiv 2026 | WebSearch | confirms phonology probes in P101/P102/P115 | — |

## Direct source fetches (not searches)

- arXiv abs + html: 2411.16765v3, 2609.06296v1, 2509.05188 (latest=v2), 2406.10501, 2509.21223 (latest=v3), 2110.05382, 2305.04868, 2302.05075, 2405.20666, 2407.01264, 2501.15187, 2609.03695, 2605.02094, 2309.02450, 2402.09611, 2503.08529v1 (anchor, only to verify SignRep Table 3/4 numbers quoted by others).
- https://mlanthology.org/tmlr/2026/madjoukeng2026tmlr-sslslr/ (venue of SSL-SLR).

## Coverage notes

- Required items all covered: SHuBERT P101, SignDino P102, SSL-SLR P103, STC P104, SignBERT P105, SignBERT+ P106, BEST P107, MASA P108, Sigma P109, SignCLIP P110, Uni-Sign P111. Added: SignSeek P112 (frozen-retrieval nearest competitor, gloss-supervised), SignMAE P113, SSVP-SLT P114, SSVT-ISLR P115.
- Not found: any sign-language JEPA/V-JEPA paper (queries 9, 20); closest is SignSeek's latent masked prediction (MAP) on labelled pose.
- Works citing SignRep found: SSL-SLR (Table 6), SignSeek (Tables 2–3). SignDino does not cite SignRep.
- Deferred DISCOVERED items listed above need a later pass (OpenHands, MSLU 2408.08544, M³-SLR, SignMusketeers, OneSign, SignMatch, Dictionary Expansion EMNLP Findings 2025, 2604.27232, 2604.07606).
- Not exhaustive; search index is US-only and recency-biased; Google Scholar "cited by" not available to this agent.
