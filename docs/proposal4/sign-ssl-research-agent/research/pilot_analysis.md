# Pilot analysis (T05) — three method hypotheses, all negative

All numbers are project measurements (REPRODUCED_RESULT), single seed unless stated, ASL Citizen **val** (test untouched for these). Protocols were locked before each pilot.

| Hypothesis / protocol | What changed | Result | Decision |
| --- | --- | --- | --- |
| H002 XWP / EX02-v1 | Continue SignRep 4000×24 steps on unlabeled ASL Citizen train videos; each masked window predicts the frozen teacher's ±8-frame neighbour window embedding (anchor + predictor) | val LOSO R@1: B0 0.4950, **M0 0.5001**, C0 (same compute, Δ=0) 0.4994, A0 (anchor only) 0.4988. M0 − C0 +0.07 [−0.12,+0.38], 3/6 signers | Null mechanism: the +0.4–0.5 over B0 is in-domain continuation, independent of the prediction target (D003). Prior art (SALT, FRAME, V-JEPA) predicted this risk (R001) |
| H003 CIP-Sign (user proposal5) / EX03-v1 | Premise test: do cross-articulator forecasters G1(own+peer) beat own-only G0 and shuffled-peer Gsh on held-out signers? | 0/36 edge–horizon gates; 25/36 edges with negative gain; ridge check agrees | Premise refuted within scope (2-D COCO-WholeBody geometry, isolated signs) → student pilot not run (spec §10.5) (D004) |
| H004 set-to-vector / EX04-v1, v2 | Label-free student trained to reproduce chamfer similarities of frozen windows | v1 transformer 0.6011 (official gallery), wmean-teacher control 0.5576, instance-SSL control 0.3018; v2 bins2-initialised soft segments 0.6435 vs its own training-free init 0.6498 | No learned student beats the training-free ordered readout (D006) |

Failed / invalid runs (kept in the ledger): EX02 attempt 1 (student deep-copied from frozen teacher → no backbone update; caught because M0 features were identical to B0), EX02/EXT001/EX04 OOMs on the shared GPU, EXT002 subset-parsing bug, CIP calibration sample-size bug.

## What the pilots taught

1. The released SignRep window embeddings already contain the information that a better clip readout recovers (+9 R@1); objectives that perturb the encoder (XWP, continuation) move retrieval by ≈ 0.5 R@1 at most at this budget.
2. Cross-articulator predictability — the premise of CIP — is not present in isolated-sign keypoint dynamics beyond each articulator's own history.
3. Learning a single-vector summary label-free is harder than it looks: a randomly initialised aggregator under-fits even a cosine teacher; initialised at the ordered readout it drifts down.
