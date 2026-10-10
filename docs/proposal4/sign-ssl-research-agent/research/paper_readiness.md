# Paper readiness memo (T06) — 2026-10-06

## Decision

**Evaluation-finding paper: READY_FOR_DRAFT (narrow scope)** — main claim C012 has locked-protocol confirmatory evidence on a held-out test split with signer-cluster uncertainty, a reproduced baseline (C011), controls that rule out the simplest alternatives, and a novelty refresh (search logs A–E).

**Method paper (new SSL objective): NOT READY → PROPOSAL_ONLY.** Three method hypotheses were tested and are negative within scope (H002, H003 = user proposal5, H004). No method claim may be written as a result.

Scope (user steering D012): SignRep-centred; SHuBERT is reference only (val check C014 kept as a side note). The confirmatory evidence is for SignRep on three datasets.

## Evidence-to-claim table

| Claim ID | Statement (as it may appear in the draft) | Evidence | Uncertainty | Status |
| --- | --- | --- | --- | --- |
| C011 | Our pipeline reproduces SignRep's frozen ASL Citizen retrieval (R@1 49.0 vs 49.95 reported) | EX05 R0 | single fixed encoder; sample CI not needed for fidelity | SUPPORTED |
| C012 | On the same frozen SignRep window embeddings, a training-free ordered two-segment readout raises ASL Citizen test R@1 by +5.9 (single vector) and window-set (chamfer) matching by +9.4; on WLASL2000 the gains are +2.0 and +1.2 | EX05-v1 locked; 11-signer cluster bootstrap; Holm | WLASL chamfer CI includes 0 | SUPPORTED_WITHIN_SCOPE |
| C005/C007 | SignRep (and every surveyed sign SSL method) evaluates frozen clips with an order-insensitive pooled readout | paper/code audit; notes P101–P115 | coverage provisional (no full forward-citation sweep) | FACT / INFERENCE |
| Diagnostics | Averaged SignRep clip embeddings are nearly time-reversal invariant (cos 0.959); static-frame windows keep 83% of R@1; confusions share handshape/location far above chance, movement near chance (ASL-LEX) | EXPL (val) | exploratory, val only | to be presented as analysis, not confirmatory |
| C009 | Cross-window latent prediction continuation does not improve over matched continuation | EX02 | 1 seed, val LOSO | negative result |
| C010 | Cross-articulator forecast increments (CIP premise) are absent in ASL Citizen keypoint dynamics | EX03 | 2-D teacher, isolated signs | negative result |
| H004 | Label-free distillation of set similarity into one vector did not beat the training-free readout | EX04 v1/v2 | 1 seed | negative result |
| C013 | Context vs reported supervised systems | reported numbers | different supervision/modality | context only, no ranking claim |
| C015 | Readout effect replicated on MS-ASL test (+1.9 bins2, +4.5 chamfer), locked EX07 | EX07-v1 | contamination risk (YouTube-ASL) | SUPPORTED_WITHIN_SCOPE |
| C016 | Single-vector whitening does not generalise (−2.4 on MS-ASL) | EX07-v1, H005 | — | negative result |

## Strongest remaining reviewer objections

1. "Chamfer/ordered pooling is well known (ViSiL, TCA, DnS)" → the contribution is the measured size of the effect for sign encoders and its implication for how SSL sign encoders are compared, not the readout; must be framed so.
2. "One encoder" → partly answered: SHuBERT residual stream shows the same direction on val (C014); test-set confirmation only for SignRep.
3. "Why does WLASL gain less?" → exploratory: not duration, weakly gallery size; WLASL test signers are 99.9% present in the train gallery (official metadata) vs signer-disjoint ASL Citizen — plausible but confounded.
4. "Is +5.9 practically meaningful?" → equals ≈ 12% relative R@1; brings label-free SignRep above the reported number of a gloss-supervised pose model (context, C013).
5. "Test split previously touched in the repo" → disclosed; this project's decisions never used those results.

## What would upgrade the method track

A new objective must beat the **ordered readout on the original checkpoint**, not the averaged readout — that is the honest baseline the evidence now sets (+5.9 R@1 for free). Candidate directions are in `open_questions.md`.
