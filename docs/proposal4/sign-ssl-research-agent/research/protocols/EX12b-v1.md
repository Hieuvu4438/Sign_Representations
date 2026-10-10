# EX12b-v1 — Seed replication of H008 on ASL Citizen

## Status

LOCKED 2026-10-07 before seeds 1 and 2 were trained. Purpose: training-seed variance for the EX12 claim (C019), not selection.

## Design

Identical D2 configuration (`h004_train.py --arm H4 --batch 64 --steps 12000 --tau 0.03 --teacher q2g`), seeds 1 and 2, trained on unlabelled ASL Citizen train windows; each evaluated on val (inside the training script) and on test with `ex12_eval.py` (same locked comparisons). Report all three seeds (0, 1, 2): mean ± sd of test R@1 and of H8 − R1; no seed is dropped or selected.

## Decision

The EX12 claim is kept as SUPPORTED_WITHIN_SCOPE if the 3-seed mean of H8 − R1 is ≥ +1.0 R@1 and every seed is > 0; otherwise the claim is downgraded to "seed-sensitive" and reported as such.
