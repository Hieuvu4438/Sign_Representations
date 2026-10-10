# ASLC002 pose-prior audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run (read-only implementation and data audit)
- Verification Status: ANALYZED; numeric checks performed; no independent complete training rerun
- Version Label: ASLC002_pose_priors_audit_v1

Nine pose-prior branches pass the audited geometry/mask/gradient checks. Activity code is connected to the loss and can receive gradients, but its current supervision is sparse and has no negative labels. This limits the interpretation of activity-weighted retrieval as learned active/inactive segmentation.

Corpus: 3000 real training videos, 268069 decoded frames, 222933 signer-present consecutive 16-frame windows. All normalized-cache hashes and provenance match prepared manifest. Pose/angle/distance numeric checks cover all decoded frames of all 3000 videos. Activity audit covers all eligible 16-frame training windows, not just each video's first window.

| Prior | Expected valid fraction under uniform-video/uniform-window sampling | Observed valid fraction in training snapshot | Updates with nonzero prior loss |
|---|---:|---:|---:|
| Body angles (22 × sin/cos) | 100.0000% | 100.0000% | 542/542 |
| Right-hand angles (41 × sin/cos) | 45.1756% | 45.1943% | 529/542 |
| Left-hand angles (41 × sin/cos) | 35.4013% | 35.2341% | 486/542 |
| Full pose (61 × xyz) | 47.4010% | 47.4605% | 542/542 |
| Right hand relative to wrist | 45.1756% | 45.1943% | 529/542 |
| Left hand relative to wrist | 35.4013% | 35.2341% | 486/542 |
| Hand/body interaction (12 × 22 × xyz) | 27.6328% | 27.7164% | 534/542 |
| Right-hand distances (5 × 11 × xyz) | 45.1756% | 45.1943% | 529/542 |
| Left-hand distances (5 × 11 × xyz) | 35.4013% | 35.2341% | 486/542 |
| Hand activity (LH, RH) | 0.1588% | 0.1153% | 2/542 |

The expected fraction above is averaged per video before averaging across the cohort; the full-window-weighted fraction is separately recorded in JSON. Repeated observed training views count each time. A valid fraction describes supervision coverage, not pose accuracy.

## Correctness evidence

- Full 61-point layout, 19 body/face + 21 RH + 21 LH; wrist-centred local hands, correct pair endpoints and signed per-axis differences match independent NumPy arithmetic.
- Teacher Euler targets are 22 body / 41 each hand, encoded as sin/cos. Teacher observation masks are explicit; missing keypoints and nonfinite targets do not contribute loss.
- Released decoder architecture and loss produce nonzero prediction gradients for every branch on a real eligible clip; masked entries produce zero gradients. Latent backward gradient is finite.
- 11 relevant existing prior/loss tests pass, including translation invariance, angle periodicity, missing wrists, global-vs-local masks and activity hand order.
- Three fixed ASLC002 videos were checked against native teacher FK: standardized pose reconstruction error 0; shared bone-length error <= 4.48e-8; hand roots match the body wrists.
- Training snapshot: updates 1..542; all observed total losses and gradient norms finite. Mean reconstruction over first 50 updates 8.033103, over last 50 1.095359. This is a progress check, not convergence or downstream-performance evidence.

## Activity limitation

Only 2522 of 222933 windows have both shoulders and both hips above confidence 0.5 throughout (1.1313%). Shoulder visibility is ~99.95%; hip visibility over decoded frames is 3.9176% / 1.9935%. Hand detection and body-wrist confidence add further exclusions.
Eligible activity windows: LH 563, RH 675. All 1238 eligible hand/window targets are 1 (active); inactive targets=0. Across all eligible windows the valid fraction is 0.2777%; expectation under actual uniform-video/window sampling is 0.1588%.

Thus the branch is not disconnected or always zero because of an autograd bug, but it almost never sees supervision and cannot learn active-versus-inactive discrimination from this corpus/mask rule. The current run should report mean pooling as the reliable reference and disclose this limitation with any activity-weighted results. Do not claim that more epochs alone creates the missing inactive labels.

## Source agreement and remaining uncertainty

Checked local ICCV main paper §3/§5.1 and supplementary A1/Table9 (P01.pdf/P02.pdf), the pinned original skeleton teacher, and the reproduced targets/loss wiring. PDFs parse without encryption or parser repair. Shape/layout/heuristic thresholds/loss weights/hand scales agree with the stated contract.

Distance direction convention, angle-confidence aggregation and complete-window activity eligibility are documented reproduction choices. The paper specifies visible-hand motion and low position but does not publish exact missingness aggregation. This audit does not establish historical equality with unavailable author pretraining code or the original YT-SL bone table, nor absolute accuracy of the 3-D teacher.

## Next action

No active training implementation, configuration, checkpoint, cache or membership was modified by this audit. Preserve ASLC002 as the current documented run. If activity-aware baseline reproduction is required, use a separately versioned run to revise eligibility (for example, an explicit teacher-inferred torso policy and sufficient observed hand frames), audit coverage and both active/inactive labels before training, and compare under the same cohort/budget. Missing hands must not automatically become inactive labels. Those alternatives need validation and should not be represented as exact author defaults.

## Evidence

pose_priors_audit.json; pose_priors_FK_audit.json; pose_priors_training_snapshot.json; pose_priors_audit.log; scripts/audit_pose_priors.py.
