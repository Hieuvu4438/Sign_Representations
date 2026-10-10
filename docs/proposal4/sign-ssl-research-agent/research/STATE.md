# Research state

| Trường | Giá trị |
| --- | --- |
| state_version | 9 (2026-10-07 ~11:00, after EX12b, H010, H012 diagnostics; ASC-JEPA set aside per user) |
| terminal_status | CHECKPOINT — draft v0.5: readout finding (C012/C015) + H008 label-free set-distilled vector SUPPORTED_WITHIN_SCOPE on ASL Citizen test, seed-stable (C019, EX12b); novelty of H008 mechanism is low (DnS/FGD), so the method track continues on signer-confounding of label-free targets (H010–H012b) |
| current_cycle | C005 complete — method track NEED_RESOURCE (D009); evaluation paper draft v0.3 |
| active_hypothesis | none running. Confirmed: C012 (readout gap, SignRep, test), C014 (same direction, SHuBERT residual stream, val). Closed negative: H002 XWP, H003 CIP (user proposal5; EX03 isolated, EX08 continuous, EX09 appearance+3-D teacher — all 0/36), H004 set-to-vector, H005 single-vector whitening, H006 mined positives (diagnostic), H007 32-frame ordered student (EX11). Also confirmed: C015 (MS-ASL test) |
| active_protocol | none open (EX05, EX06/b, EX07 completed) |
| active_run/job | EX16-v1 (SignRep-repro vs JEPA, same priors) runner `method/jepa_targets/runtime/run_ex16.sh` (log `runtime/logs/ex16_chain.log`; waits for free GPU memory — shared GPU), then EX15 JHOG/JPRM retries. EX14 PAUSED |
| G0 | PASSED (58 notes, search logs A–E incl. novelty refresh 2026-10-06) |
| G1 | EVAL_REPRODUCED on ASL Citizen test (0.4902 vs 0.4995 R@1) |
| G2 | PASSED for the evaluation claim; method hypotheses rejected |
| G3/G4 | passed for readout; method pilots failed G4 |
| G5 | PASSED for C012 (locked protocol, single test access, Holm) |
| G6 | evaluation-finding draft ready (narrow scope); no method claim |
| GPU-hours used (approx.) | ≈ 53 before EX14; cap extended to ≤ 85 for the method track (D019) |
| Disk | ≈ 24 GB free; project runtime ≈ 14 GB (features of ASL Citizen train/val/test, WLASL, SHuBERT val ×2) |
| Test accessed by this project | ASL Citizen + WLASL test once (EX05-v1); MS-ASL test once (EX07-v1); all logged in `method/p4_signssl/runtime/logs/TEST_ACCESS.log`. No untouched benchmark test split remains locally for ASL except NMFs-CSL/other languages |

## Key artifacts

`paper/draft.md` (v0.3), `paper_readiness.md`, `paper_outline.md`, `confirmatory_analysis.md`, `pilot_analysis.md`, `open_questions.md`, `reproducibility_checklist.md`, cycles C001–C002 (+ this state for C003), ledgers.

## Next actions (if research continues; run sequentially, ≤ 2 processes)

1. Method track (needs a new hypothesis that beats "original SignRep + ordered readout", not the averaged readout): candidates in `open_questions.md` #4 (partial/set-matching clip objective with continuous-signing pseudo-pairs) and #7 (label-free signer-nuisance removal measured by the same-signer distractor test — note: gallery whitening failed, H005). Write a hypothesis card + closest-prior-art pass before any code.
2. Optional strengthening of the paper: SHuBERT on test (needs ≈ 9 h train + 7 h test extraction with 2 workers) or a third encoder; signer-disjoint WLASL re-split as a robustness table.
3. CIP-Sign (proposal5) only if a continuous-signing corpus with a 3-D hand teacher becomes available locally; first step = `scripts/cip_gate.py`.

## Queue (user request 2026-10-06, after current task)

1. **Current:** H008 dev (D1–D3 on val) → lock EX12 → confirmatory on ASL Citizen / WLASL / MS-ASL test (disclosed reuse; H008 never tuned on test).
2. **Next (user request):** implement and test **ASC-JEPA** (`docs/proposal8/ASC_JEPA_End_to_End_AI_Agent_Implementation_Spec_vi.md`) in a NEW folder `method/asc_jepa/`. Plan: SignRep warm-start branch (spec §2.4 allows it as a separate branch) on unlabelled ASL Citizen train videos; ROI from local COCO-WholeBody keypoints (declared adaptation of the MediaPipe extractor); chain B0 → B1 → B4 (+ C4a random-gate control if B4 shows signal); evaluate frozen retrieval on val (SignRep readout + two-segment readout). MAE-K400 controlled chain only if budget allows (requires Hiera checkpoint download). If not better → record negative and continue the research program.

## Queue (2026-10-07, method track: signer confounding of label-free targets)

Findings so far (val, exploratory unless stated): label-free neighbours are signer-dominated (H010); temporal-difference chamfer T4 is the best training-free readout (0.726 on 3k val queries); training-free moment vector M2D2 0.653 full val vs H008 0.675 (+2.2 CI>0) — M2D2 is a required baseline; rest-segment pseudo-signer exclusion raises cross-signer same-gloss mining precision +5.6 (oracle +19.4) (H012).

1. DONE: H011 dev +1.67 [0.38, 2.91] → go (marginal). 
2. DONE: H012b X1 +0.36, X2 +0.25 → NOT met; mining precision did not transfer to the student. XO reference pending.
2b. EX14-v1 LOCKED (H11 only) and queued; XO oracle reference 0.6838 (+0.86) → candidate composition not the bottleneck.
2c. H013 (distillation generalisation gap) queued after EX14: `runtime/h008/run_h013.sh` (A1 window dropout, A2 extra unlabelled pools; A3 conditional), log `logs/H013_chain.log`; decision rule in `hypotheses/H013.md`. Uses `scripts/h013_train.py` (copy; `h004_train.py` frozen by EX14 lock).
3. If either passes: write EX14 (locked) — best label-free arm(s) + M2D2 baseline + T4 readout on ASL Citizen / WLASL / MS-ASL test (clusters recomputed per dataset, same K rule); disclose prior test use.
4. If none passes: write the signer-confounding measurement (H010/H012 tables, oracle headroom) into the draft as analysis; next idea = better label-free nuisance estimate (e.g. per-clip rest descriptors from face/body ROI rather than full-frame windows).

## Current direction (user request 2026-10-07 ~15:30): H014 / EX15-v1

Paused: EX14 (H011 seeds 1, 2 trained; val 0.6971 / 0.6864; WLASL/MS-ASL + test eval not run — resume with `runtime/h008/run_ex14.sh` after removing nothing; it skips existing models), H013 (not started).
EX15: V-JEPA 2.1 ViT-B continued pretraining on unlabelled ASL Citizen train, arms A0/SR refs, M, J, M+P, J+P, then J+REL/NM/MOT/HOG. Literature: `target_priors_review.md`. Protocol: `protocols/EX15-v1.md`.
