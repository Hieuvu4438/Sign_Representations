# OSCRREFTEST002 — unchanged manual fixtures, NumPy execution repair v2

Inherits all immutable fixture/input/manual values/float1e−12/sourceSHA/license/
CPU60s/GPU0/HPO0/complete2MiB limits from OSCRREFTEST001_manual_oracles_v1.md.
Fresh run identity/code/config; original failed run remains unchanged.
Only delta: namespace builtin __import__ added for NumPy internal lazy imports.
Still extract/execute only the exact audited sourceAST function with no module
execution, import statements/decorators/annotations. Referencefunction unchanged.
No sandbox guarantee; fixed sourcehash/manual inspection provide provenance.

Expected audit: pinned areas.25/1/.25 for all_known_wrong/perfect_separated/
wrong_tied_scores, respectively; manual area0/1/0. Expected two disagreements
give reference REJECTED for H007; successful audit is not successful parity.
No public-paper score inference, numeric sign-feature/task/test quality access.
Bind v1protocol/D077/originalfailure status/config/log/error alongside newsource.
