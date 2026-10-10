# Paper outline (draft scope: evaluation finding + negative method results)

Working title: **"Averaging Hides What Frozen Sign Encoders Know: Readout Matters for Self-Supervised Sign Representations"** (title is provisional; claims below map to `paper_readiness.md`).

## 1. Problem

Self-supervised sign encoders (SignRep, SHuBERT, SignDino…) are compared with frozen retrieval or probes after pooling short-window or frame features into one clip vector. If the pooling discards information the encoder has, the benchmark measures the readout as much as the representation — and objective changes that seem to help or hurt may be readout artefacts.

## 2. Contributions (each with claim IDs)

1. A faithful re-implementation of SignRep's frozen dictionary-retrieval protocol that reproduces its ASL Citizen number (49.0 vs 49.95 R@1) — C011.
2. Evidence that the released SignRep window embeddings retrieve far better than its averaged readout suggests: +5.9 R@1 with a training-free ordered two-segment vector and +9.4 with window-set matching on ASL Citizen test (11/11 signers positive), smaller on WLASL2000 (+2.0 / +1.2) — C012, locked protocol, single test access.
3. Diagnostics of what averaged sign embeddings discard: time-reversal near-invariance, static-frame robustness, ASL-LEX-based confusion analysis that avoids probe leakage (handshape/location shared far above chance in errors, movement near chance) — exploratory.
4. Negative results that matter for method design, all against matched controls: cross-window latent-prediction continuation (≈ continuation itself), cross-articulator conditional-innovation premise (0/36 forecast gates), label-free set-to-vector distillation (no gain over the training-free readout) — C009, C010, H004.
5. A recommendation: report ordered/set readouts alongside the averaged one, and compare new SSL objectives against "original encoder + ordered readout".

## 3. Method section (evaluation protocol, not a new model)

- Windows: 16 native frames, stride 2; SignRep `features` (z^avg); activity weights γ = max(P_LH,P_RH).
- Readouts R0 (paper), R1 (bins2 over activity span), R2 (chamfer@20 rerank). Class-max cosine; train gallery; ties counted against the query.
- Statistics: paired signer-cluster bootstrap; Holm for two primary comparisons.

## 4. Experiments

- Tables: EX05 test results (ASL Citizen, WLASL); val development table (pooling family incl. max/GeM/normalised mean, bins K = 2/3/4, chamfer, DTW); gallery-size ablation; WLASL signer-overlap analysis (pending).
- Figures: gain by ASL-LEX movement type; gain by active-span length; perturbation table (reverse/shuffle/static).
- Negative-results section with protocols and controls (EX02, EX03, EX04).

## 5. Limitations (must appear)

One encoder for the confirmatory claim (SHuBERT generality pending/inconclusive); ASL only; WLASL selection contamination and signer overlap; ASL Citizen test previously touched in the repository by unrelated projects; ASL-LEX codes describe the initial morpheme and lack palm orientation; no Deaf/expert review of linguistic interpretation; pose teacher used for CIP differs from proposal5's canonical teacher.

## 6. Not to claim

No "new SSL method", no "linguistic understanding", no "signer-invariant", no SOTA ranking across supervision regimes.
