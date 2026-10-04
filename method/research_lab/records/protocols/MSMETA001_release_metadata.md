# MSMETA001 — Bounded release and source/split feasibility audit

Locked before launch, 2026-10-04; D041; hypothesis H006 independent-corpus feasibility.
Diagnostic metadata audit, no quality metric or cohort selection. Full local
MS-ASL train/val/test and classes/synonym JSON; all rows retained. Exact bytes
and parsed JSON compared to publisher archive, without field normalization.
Release differences must remain explicit; audit completion does not imply
official-comparable model evaluation.

One bounded public archive GET, 4MiB maximum, 25s network timeout and 32MiB
declared decompressed-member bound. Read selected five JSON members into memory;
retain downloaded archive, no path extraction or media download. Header/redirect/
archive SHA recorded. Roots/identities bound in registered config. Compare exact
accepted YouTube parentIDs to LINEAGE003's headerless39197-ID public corpus list;
pool membership is not actual checkpoint training membership.

Primary audit outputs: release equality, row counts/contracts, unique signer_id
and legacy signer (nonnegative, missing -1 separate), pairwise signer/source/
exact(parent,start_time,end_time) intersections, source candidate existence
video/<accepted URL ID>.mp4, immediate raw/processed file counts/bytes, full
class support and unavailable parentIDs. No interval-overlap completeness,
decoded source identity, native frame/crop coverage, physical signer identity,
near-duplicate proof or numeric processed-filename join. Metadata intervals
below16 frames are a hint only, with no dropping or native eligibility claim.

Success: complete bounded auditable outputs; unexpected contracts and overlap
are findings, not suppressed errors. Technical failure if input hash, root
schema/class count, archive/network/bounds or source CSV contract fails. Keep
all failure artifacts; at most two reasoned fresh-version retries. No tuning,
model inference, feature extraction, test-quality access or GPU.

CPU deadline120s, seed1701 solely runtime identity, HPO0/GPU0. Complete per-file
caps plus source/config/env/log/status/error/manifest and atomic-replacement
reserve calculated before exclusive output creation. Durable ceiling12MiB;
256MiB RAM planning plus existing scratch reserve check (not measured peakRSS).
Small-record exception to failed root bulk20% reserve; no bulk extraction.
Inspect counts/release/source/signer evidence before any fresh GPU protocol.

Precondition MSMETATEST001: two fixed software test groups (URL identity and
annotation schema corner cases), CPU120s/GPU0/durable2MiB. Same bounded wrapper,
declared sources and outputs; no network or performance claim. Audit config
registered only after terminal passed tests. Same-agent inline roles; no external
scientific independence claim. Test/report/status/config SHA retained.
