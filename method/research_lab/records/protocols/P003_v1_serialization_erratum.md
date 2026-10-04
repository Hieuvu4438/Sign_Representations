# P003_v1 — strict JSON boolean serialization incident, after completed inference

2026-10-04local, incident CONFAN001 failed CPU-only at22:12:42UTC, wall.151s,
GPU0seconds. Original prediction/integrity runs remain complete/valid/immutable.
All1217query inputs and15rankings already verified exactly before scoring.

Root cause: actual_cost_screen used `intcomparison and numpy.median(...)<=.8`,
which can return numpy.bool_ rather than Pythonbool. The strict JSON writer
correctly refused this unsupported type. Unit tests checked the truth value but
did not previously serialize the entire analysis result. json.dumps failed before
writing analysis.json or metrics.csv; no task numbers were persisted/shown to the
agent. The failed process computed scores internally, so this is **after a scoring
attempt**, not a new claim that no computation happened. No outcome-driven method,
metric, threshold or policy selection is justified by this failure.

Fix: wrap the unchanged boolean expression in Pythonbool. No comparison, margin,
bootstrap, grouping, weighting, source, seed, metric or search changes. Add strict
full-result JSON roundtrip/oracle fixture. Only confirmatory.py gate-verdict
acceptance and the output boolean type change; new supplementary engineering
gate links immutable original pre-test registration/config hashes and exact diff.
Do not edit original freeze/technical gate or overwrite CONFAN001 failed artifacts.

Retry1: new CONFAN002 output/run ID, same scientific CONFAN001config SHA and
original method/scorer. Bounded CPU120s, no new inference/GPU costs. Resume semantics
unchanged. Any further failure requires diagnosis, max2engineering retries. This
incident refutes no scientific hypothesis, and does not create another policy trial.
Core nativeadapter/pooling/scorer and completed vectors/ranks/timing unchanged.

The corrected outputs may be interpreted under the original scientific design
with both disclosed engineering corrections; never describe a silently pristine
code freeze. Additional source/recipe/rank verification is post-execution QA,
not a test-dependent changed endpoint. Same-agent roles, not independent review.
