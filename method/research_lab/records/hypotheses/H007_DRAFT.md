# H007 — Gallery-relative rejection under signer/domain changes

- Status: **draft**, NOT registered/piloting/promoted.
- Owner: Codex; created2026-10-05; same-agent sequential review.
- Direction slug: frozen_probe; contribution candidate: analysis/evaluation.
- Nearest works: L01/S46/S47/S48/S49; S45 abstract-only calibration lead.
- Protocol/run IDs: none. GPU allocation: none; no new score computed.
- Intake/decision: [intake](../literature/OPENSET_INTAKE_20261005_r1.md),[D073](../decisions/D073.md).

## Research question

Ngưỡng rejection của frozen SignRep có giữ operating point khi membership và
kích thước one-shot dictionary thay đổi qua signer/domain trên corpus ASL?
Use case dự kiến: một dictionary hữu hạn có thể không chứa lớp query;
usability thực tế chưa khảo sát. Literature support S47/S48; local rejection
failure **unknown**, D072 chỉ đo closed-gallery ranking.

## Prediction và falsifier

Hypothesis: threshold fitted on source calibration can produce material
unknown FAR drift when gallery/signer/domain changes, even if known-only ranking
is adequate. Candidate screen: ≥0.05 absolute FAR drift is a proposed practical
failure threshold, chosen before rejection scores; metadata feasibility and
uncertainty plan must still be locked in a new version before running.
Đây là dự đoán empirical, không phải hệ quả tất yếu của domain shift.

No new correction mechanism selected. First discriminate baseline failure from
gallery composition, label/cohort selection, source/signer leakage and changes
to feature inputs. If fixed standard scores/calibration controls retain their
operating points, or effects disappear under paired/common-query controls,
kill the proposed failure premise before designing a method.

## Prospective experiment constraints

Train-only calibration, label/source/signer joins and native feature provenance
must pass metadata audit. Existing ASLCitizen dev/MS-ASLval reuse is exploratory;
fresh confirmation unknown and test outputs locked. No class selection from
known D072 performance or rescue of H006. Frozen raw cosine baseline required.
NAC/margin/maximum-softmax/appropriately defined energy scoring are controls,
not inventions. More labels/negative-query access reported for every baseline.

Proposed primary diagnostic: unknown FAR change after freezing source threshold;
report known CCR/coverage/error and joint curve, all galleries, paired signer
uncertainty. Exact estimand, calibration target, class/signer split, seeds,
evaluator reference/parity, cap and selection policy **pending registration**.
No HPO allowed by this draft. Initial census CPU metadata only; no GPU request
or allocation. Pilot cap estimate pending input layout/count audit.

## Feasibility scorecard (0–5; not promotion)

| Criterion | Score | Basis |
| --- | ---: | --- |
| Problem value | 3 | Finite dictionary use case; local rejection failure unknown |
| Novelty | 1 | S47/S48 direct mechanisms; no unique contribution yet |
| Protocol | 3 | Class membership metadata possible; calibration/groups/evaluator unvalidated |
| Compute | 4 | Existing clip vectors likely reusable; no empirical cost registration |
| Identifiability | 3 | Coupled gallery/query controls possible; domain/preprocessing confounds remain |

Weighted2.65/5. Hard blockers to measurement: exact calibration cohort/provenance,
registered protocol/runs and evaluator validation. Hard blockers to scale:
failure evidence, mechanism novelty, fresh confirmation and S45 final methods
if pursuing conformal routing. H007 adds a different task question, not reopened
whitening or a successful method. Original H001–H006 budgets/results preserved.
