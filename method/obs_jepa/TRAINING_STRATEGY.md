# Training strategy (proposal11 A/B/C vs SignRep reproduction, same protocol)

## Fixed protocol (all methods, including the baselines)

| Item | Value |
| --- | --- |
| Pretraining data | 3,000 unlabelled ASL Citizen train videos (ASLC002 corpus) — identical for SignRep-repro and A/B/C |
| Budget | 4,000 optimizer steps × 32 clip views (= 128k views, the SR3D budget of EX16); B sees 64-frame windows (2.56 s) — longer physical support, reported as a factor |
| Init | SignRep-repro: Hiera-B Kinetics-MAE; A/C: V-JEPA 2.1 ViT-B (distilled); B: scratch |
| Seeds | 1 seed for screening; finalists × 3 seeds |
| Evaluation | `oj/evaluate.py` on ASL Citizen **val only**: ASLC300 (gallery 3,000 train videos, 1,500 val queries) true linear probe + retrieval, LOSO-val whitened retrieval, body-part/signer probes |
| Primary metric (declared now) | ASLC300 **true linear probe macro Top-1** (proposal §12.2); co-primary LOSO-val whitened R@1 (continuity with EX15/EX16) |
| "Better than SignRep-repro" | both primary metrics ≥ +1.0 point with signer-cluster bootstrap CI > 0, consistent over 3 seeds; per-signer results shown (only 5–6 val signers) |

Baselines evaluated with the same code: **SR3D** (EX16 SignRep-repro, batch 32 × 4k), **ASLC002** (Codex SignRep-repro, batch 4 × 5k), **A00** (frozen V-JEPA 2.1, A preprocessing), **JV3D** (EX16: V-JEPA + SignRep priors), random-init B (floor).

## Phases and gates

| Phase | Runs | Question | Gate to continue |
| --- | --- | --- | --- |
| P0 baselines | extract + evaluate SR3D, ASLC002, JV3D, A00 | reference numbers in this protocol | pipeline reproducible |
| P1a A recipe | A-R generic vs A-D generic | A-R or A-D background | pick on val (primary metric), lock |
| P1b A factorial | A10 generic/uniform (= P1a), A11 generic/balanced, A12 partial/uniform, A13 partial/balanced, A10c generic_cov/uniform | weighting vs masking vs "more hand left visible" (§13.2, §13.5) | A12/A13 beat A10c → mask arrangement matters; else report allocation effect |
| P2 B | B12 latent, B13 latent+anchors, B11 centred-CE, B10 raw | does geometry SSL learn more than raw targets? | B13 teacher audit: probes (shape/location/movement) ≥ B12 and direct descriptors |
| P3 C | C1 = A13 + C-observed (`--c-reserve`), control A13r = A13 with the same reserved masks | does visible geometry distillation help RGB beyond the same masks? | only if P2 gate passed |
| P4 confirm | best A, best B, C if passed × 3 seeds; best A at 384 px (A5); A14 (whole) and A15 (jitter) controls for the best A | stability, resolution, metadata specificity | decision rule above vs SR3D |

## Cost (measured on the shared GPU, other jobs running)

A 224 px: ≈ 0.8 s/step at batch 32 → ≈ 55 min per 4k-step run; B: ≈ 1.1 s/step → ≈ 75 min; 384 px ≈ 3–4× A.
P1 ≈ 6 runs ≈ 5.5 h; P2 ≈ 5 h; P3 ≈ 2 h; P4 ≈ 10 h. Feature extraction is currently the slowest part (≈ 45 min per model on the 10k-clip val set + ≈ 20 min ASLC300): parallel decoding in extraction is the first engineering fix before P1.

## Order on the shared server (≤ 2 jobs, one GPU job at a time)

1. Finish EX16 (running) → its SR3D/JV3D/ASLC002 models become P0 baselines.
2. P0 extraction/evaluation; P1a; P1b — sequential runner with GPU-memory wait and OOM retry (as `jepa_targets/runtime/run_ex16.sh`).
3. P2 in parallel with P1 only if GPU memory allows (B is small and CPU-bound); otherwise after P1.
4. P3 after the B gate; P4 last.

Known risks: at 224 px hands occupy few patches (partial events: ≈ 19 structured targets, 13% fallback), so A's region effects may only appear at 384 px; 3,000 videos is small for a 86M-parameter ViT (fast overfitting possible — watch the held-out losses and collapse statistics); COCO-WholeBody quality is uncalibrated; ASLC300 gallery = pretraining videos (transductive for every method alike).
