# Hypothesis ranking (cycle C001/C002, 2026-10-05)

Scores 0–5 are qualitative judgements by the same agent (internal review, knew all EXPL numbers in `cycles/C001.md`); they are not evidence, and a fatal objection is not offset by a high total.

| ID | Hypothesis | Significance | Novelty left | Falsifiability | Fairness of baseline | Impl. risk (5 = low) | Info/cost | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H002 | XWP continuation: windows predict ±Δ neighbour teacher embeddings so the averaged clip embedding keeps movement | 4 (targets a measured loss mode of the anchor) | 2–3 (latent prediction is generic; sign-specific mechanism + diagnostics are the delta; pending P401–P404) | 5 (C0/A0 controls, movement-type and reversal predictions) | 4 (matched data/steps/masking/predictor) | 3 (continuation can be destructive — seen in another repo project) | 4 (≈ 4 GPU-h pilot) | **PILOT (primary)** |
| H001 | Label-free learned single-vector aggregation over frozen windows (e.g. distil chamfer similarity into one vector, or masked window prediction) | 3 (efficiency + readout) | 1–2 (SCALE masked clip-feature prediction; Distill-and-Select style similarity distillation in video retrieval — to verify) | 4 | 4 | 5 | 5 (minutes on cached features) | BACKUP — run only if H002 fails or as a readout baseline |
| H-ord | Explicit temporal-order objective (arrow-of-time / order prediction) | 2 (EXPL: position penalty adds nothing beyond chamfer; reversal costs only −2.4 R@1 in this closed vocabulary) | 2 | 4 | 4 | 4 | 2 | DEPRIORITISED — evidence says order is not the main loss in ASL Citizen; keep reversal sensitivity as a diagnostic only |
| H-seed-B | Reliability-aware pose-teacher targets | 2 | 2 (SignBERT+ weighting, β-NLL) | 3 | 2 (local teacher ≠ SignRep's 3-D teacher; cannot reproduce SignRep's prior losses) | 2 | 2 | DEFERRED (weak test bed) |
| H-seed-C | Nuisance (background/appearance) robustness | 3 | 2 (SignRep already has an adversarial style loss; FAME/BE) | 3 | 3 | 3 | 2 | DEFERRED — no project evidence yet that nuisance is the binding error; a signer-ID probe can be added later |
| H-seed-D | Non-manual (face/mouth) targets | 3 | 2 (SHuBERT/SignDino face streams) | 2 (no validated non-manual groups in ASL Citizen; NMFs-CSL would need a new pipeline) | 3 | 2 | 1 | REJECTED for this cycle (out of reach of the primary benchmark's annotations) |

## Why H002 over H001 as primary

H002 changes the encoder (a representation-learning contribution) and makes predictions that a readout change cannot fake: the **same** averaged readout must improve, specifically on path-movement signs, while C0 (same compute, Δ=0) must not. H001 is cheaper but its closest prior art (SCALE; similarity distillation) already covers the mechanism in generic video, so its defensible delta would be only the sign-domain evidence.

## Fatal-objection watch list for H002

1. Domain adaptation alone (continuation on ASL Citizen) explains the gain → C0/A0.
2. Predictor makes the task trivial (copying, cos(T(A),T(B)) ≈ 0.52) → log prediction loss vs copy baseline; mechanism diagnostics.
3. Gain only appears with the LOSO val gallery, not the official train gallery → re-test on official protocol before any claim.
4. Closest prior art (frozen-teacher JEPA, DPC) already reports the same effect in sign or general video → novelty map after P401–P404.
