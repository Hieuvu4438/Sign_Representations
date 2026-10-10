# Search log E — novelty refresh for the readout claim (C-R1)

Date 2026-10-06. Engine: WebSearch (US). Purpose: find prior sign-language work that compares clip readouts (averaging vs set/sequence matching) of frozen sign encoders for dictionary retrieval.

| # | Query | Kept | Excluded / note |
| --- | --- | --- | --- |
| E1 | sign language dictionary retrieval pooling frozen features temporal aggregation comparison mean pooling vs sequence matching isolated sign | SignMatch 2609.01886 (already P405: supervised, learned dictionary projection over 32-frame SignRep clips, LSTM; no mean-vs-set ablation), SignSeek 2609.03695 (P112: attentive pooling, gloss-supervised) | Temporal Accumulative Features (2019, classifier features), Temporal Lift Pooling (CSLR, learned pooling inside a supervised network), theses/production papers |
| E2 | "sign retrieval" OR "sign lookup" video "chamfer" OR "dynamic time warping" embeddings self-supervised 2024–2026 | none sign-specific | generic video alignment (LAC, global temporal alignment), late-interaction retrieval (ColBERT-style chamfer), driving-video retrieval |

Conclusion (scope: searches A–E, 2026-10-06): no sign-language paper found that measures how much a frozen sign encoder's averaged readout under-estimates retrieval relative to training-free order-aware / set-matching readouts on the same embeddings. Set matching itself is not new (ViSiL, TCA, DnS, late interaction); the defensible contribution is the measured effect for sign encoders, its dataset dependence, and its implication for how SSL sign encoders are compared. Not exhaustive: forward citations of SignRep beyond SSL-SLR/SignSeek/SignMatch not enumerated.
