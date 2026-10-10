# Search log C — Evaluation, data, and linguistic structure

Owner: literature agent (family C). All queries run 2026-10-05 via WebSearch (US index) unless noted; fetches via WebFetch (arXiv abs/HTML, ACL Anthology, NeurIPS proceedings, OSF API, GitHub, author pages); PDFs converted locally with PyMuPDF in the session scratchpad (temporary, not part of the project). Notes: research/papers/P301–P322.

Scope: ASL Citizen protocol and follow-ups; WLASL / MS-ASL splits and critiques; ASL-LEX 2.0 phonology and its join to ASL Citizen; probing/analysis of sign models (phonology, signer identity, non-manuals, minimal pairs); pose-estimator reliability. Time range: 2019–2026-10.

## Queries

| # | Date | Where | Query | Kept | Excluded (reason) |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026-10-05 | arXiv / NeurIPS | fetch arXiv 2304.05934 abs + HTML; NeurIPS camera-ready PDF | P301 | — |
| 2 | 2026-10-05 | WebSearch | ASL Citizen community-sourced dataset NeurIPS 2023 Datasets and Benchmarks proceedings | P301 (proceedings URL), P310 (2410.05206 surfaced) | Scribd copy, liner.com review (secondary) |
| 3 | 2026-10-05 | MSR project page, GitHub | microsoft.com/…/asl-citizen ; github.com/microsoft/ASL-citizen-code | P301 license/code facts | — |
| 4 | 2026-10-05 | WebSearch | ASL Citizen phonology ASL-LEX isolated sign recognition Kezar "Improving Sign Recognition with Phonology" | P303, P304; SignDino 2609.06296 (already P102, other family) | awesomepapers.io mirror |
| 5 | 2026-10-05 | WebSearch | "ASL Citizen" benchmark 2024 2025 isolated sign recognition state of the art accuracy signer-independent | P322 (NAACL Findings 389); SignBart 2506.21592 DISCOVERED | SignBart not opened (pose ISLR model, low relevance to evaluation) |
| 6 | 2026-10-05 | arXiv HTML | SignDino 2609.06296 (phonology probe protocol, ASL Citizen results) | cross-ref only → P102 owned by other family | — |
| 7 | 2026-10-05 | arXiv HTML/PDF | SHuBERT 2411.16765 (Table 2, Table 9–10, §4.1 MS-ASL overlap) | cross-ref only → P101 owned by other family | — |
| 8 | 2026-10-05 | ACL Anthology PDF | 2023.eacl-main.200 | P303 | — |
| 9 | 2026-10-05 | WebSearch | ASL-LEX 2.0 database download OSF phonological properties 2,723 signs license | P302 (OUP, PMC, asl-lex.org, OSF) | ASL-LEX 1.0 papers (superseded; background only) |
| 10 | 2026-10-05 | OSF API | api.osf.io/v2/nodes/zpha4 (files, license); streamed signdata.csv and signdataKEY.csv in memory | P302 LOCAL_AUDIT | — |
| 11 | 2026-10-05 | arXiv | 2310.00196 (Sem-Lex) abs + HTML | P304 | — |
| 12 | 2026-10-05 | WebSearch | "Exploring Strategies for Modeling Sign Language Phonology" Kezar | P305 | claplab publications page (index) |
| 13 | 2026-10-05 | arXiv / GitHub | 1910.11006 (WLASL) abs + PDF; github.com/dxli94/WLASL | P306 | emergentmind topic pages (secondary) |
| 14 | 2026-10-05 | WebSearch | Dafnis Chroni Neidle Metaxas WLASL incorrect labels cleaned LREC 2022 isolated sign recognition | P308, P309 | ResearchGate copies, Semantic Scholar figure page |
| 15 | 2026-10-05 | WebSearch | WLASL signer overlap train test split critique signer-independent evaluation isolated sign recognition | DISCOVERED: ChaLearn LAP 2105.05066 (AUTSL signer-independent challenge); "Rethinking Sign Language Translation: The Impact of Signer Dependence on Model Evaluation" 2609.07965 (SLT, not opened) | TransSLR 2608.06407, Bangla SLR 2608.06252 (other languages/tasks; snippet-level claims only) |
| 16 | 2026-10-05 | ACL Anthology | 2022.signlang-1.26 (abs + PDF), 2022.sltat-1.3 (abs) | P308, P309 | — |
| 17 | 2026-10-05 | arXiv | 1812.01053 (MS-ASL) abs + PDF v2 | P307 | — |
| 18 | 2026-10-05 | WebSearch | WLASL MS-ASL duplicate videos overlap test set leakage YouTube-ASL contamination isolated sign recognition | none new | results generic; contamination number came from SHuBERT §4.1 instead |
| 19 | 2026-10-05 | arXiv HTML | YouTube-ASL 2306.15162 (check for overlap statistics) | negative: only "OpenASL is largely a subset of YouTube-ASL" (§2.1); no MS-ASL/WLASL overlap % | — |
| 20 | 2026-10-05 | arXiv | 2604.27232 (SL-Models-Analysis / ASL-MTP) abs + HTML; local README | P313 | — |
| 21 | 2026-10-05 | WebSearch | signer identity information in sign language representations probing signer leakage self-supervised pose features | P316 | SignRep (anchor; other family), Cross-Modal Consistency 2503.12485, SSL spatial-temporal consistency 2406.10501 (SSL family) |
| 22 | 2026-10-05 | WebSearch | signer-invariant sign language recognition adversarial signer removal "signer" embeddings isolated sign recognition | P314 (SignSeek 2609.03695); DISCOVERED: signer removal module for CSLR 2212.13023 (ACM TOMM), DeSIRe (deep signer-invariant representations), adversarial multi-task signer-independent features (Applied Intelligence 2022), "Towards the extraction of robust sign embeddings for low resource SLR" 2306.17558 | Not opened: CSLR or older supervised signer-invariance methods; listed as follow-up for a signer-adversarial branch (which would add supervision) |
| 23 | 2026-10-05 | arXiv HTML + PDF | 2609.03695 SignSeek (Table 2 incl. SignRep frozen retrieval) | P314 | — |
| 24 | 2026-10-05 | arXiv | 2008.10428 NMFs-CSL abs + PDF | P315 | — |
| 25 | 2026-10-05 | WebSearch | MediaPipe hand landmark failure sign language motion blur occlusion comparison RTMPose COCO-WholeBody hands evaluation sign language | P317, P320; RTMW via #27 | forasoft blog (non-scholarly); HMP 2312.16737, robot hand shadowing 2603.11383, SLP conditional collapse 2606.01643 (off-topic) |
| 26 | 2026-10-05 | WebSearch | Moryossef "Evaluating the Immediate Applicability of Pose Estimation for Sign Language Recognition" CVPR workshop 2021 | DISCOVERED only (CVF page returned HTTP 403; not read) | — |
| 27 | 2026-10-05 | WebSearch | RTMW real-time multi-person 2D and 3D whole-body pose estimation arXiv COCO-WholeBody hand AP | P319 | review/blog mirrors |
| 28 | 2026-10-05 | WebSearch | Ivashechkin Mendez Bowden "Improving 3D Pose Estimation for Sign Language" 2023 | P318 (author PDF) | "Meaningful Pose-Based Sign Language Evaluation" 2510.07453 DISCOVERED (SLP evaluation; follow-up) |
| 29 | 2026-10-05 | WebSearch | phonological minimal pairs isolated sign recognition evaluation confusable signs ASL-LEX neighbors model errors | P312 | Journal of Memory and Language 2021 (human lexical-decision neighborhood effects; psycholinguistic background, not model evaluation) |
| 30 | 2026-10-05 | WebSearch | "ASL Citizen" phonological features handshape location probing frozen representations 2025 2026 arXiv | P321; DISCOVERED: PhonSSM 2604.08761 (ICLR 2026 workshop; WLASL2000 72.1% claim), handshape GNN 2509.18309 (Carbo & Nalisnick; 37-class handshape benchmark) | Abstract-only checks of these two; not noted as P-files (model papers, not evaluation) |
| 31 | 2026-10-05 | WebSearch | Atwell Bragg Alikhani "Studying and Mitigating Biases in Sign Language Understanding Models" EMNLP 2024 | P310 venue; P311 (EMNLP Findings 2025 dictionary expansion) | dblp/program pages (index only) |
| 32 | 2026-10-05 | ACL Anthology PDF | 2025.findings-emnlp.1243 | P311 | — |
| 33 | 2026-10-05 | arXiv | 2411.03568 ASLKG (HTML) | P322 | — |
| 34 | 2026-10-05 | arXiv HTML | 2503.08529 SignRep (dictionary retrieval protocol, Table 3/4, pose teacher ref) | cross-ref only (anchor; other family) | — |

## Error in process (recorded, not hidden)

- One WebFetch to a guessed arXiv ID (2310.19066) for Moryossef et al. returned an unrelated paper ("Gauge-optimal approximate learning…"); discarded, not cited. Lesson: do not guess IDs.
- Two WebFetch summaries of SHuBERT Table 9 disagreed (handshape 0.6293 vs 0.7080); resolved by local PDF text: 0.6293 is Sem-Lex, 0.7080 is ASL Citizen.

## Local audits run (read-only)

- ASL Citizen splits/metadata (/home/dongvk/datasets/ASL_Citizen): columns, counts, signer disjointness, gloss coverage, keypoint format, license file (P301).
- ASL-LEX 2.0 ⨝ ASL Citizen join coverage, class counts, majority baselines, coded minimal-pair supply (P302).
- MS-ASL local JSON: signer disjointness, cross-split same-video URLs, and YouTube-ASL video-ID overlap (38.3% of test samples) (P307).
- WLASL local metadata (nslt_*.json, metadata_meaning_frames.json) has no signer IDs or URLs → WLASL signer-overlap count NOT computed (UNKNOWN; needs WLASL_v0.3.json).

## Blind spots / not yet covered

- No FULL_READ yet for P310, P312, P314, P321 (key evaluation papers) — tables taken partly from model-summarized fetches; verify before ledger promotion.
- Not searched: PopSign ASL (NeurIPS 2023 D&B) protocol; ASLLVD; How2Sign isolated subsets; CSL/AUTSL signer-independent protocols beyond NMFs-CSL; forward citations of ASL Citizen beyond those surfaced.
- Backward snowballing from P312 and P311 reference lists not done.
- YouTube-SL-25 video-ID list vs MS-ASL/WLASL not checked (no local list).
