# Participant-count erratum — P001_v4 first dev pilot

Recorded2026-10-04 after first prediction analysis. The fixed COHORT001 dev
subset has **five**, not six, participants: P21=112, P39=112, P26=109,
P12=25, P5=20 clips; total378. Whole ASL Citizen validation metadata has six
participants. Earlier descriptions incorrectly carried that whole-split count
into the100-class subset. No clip or participant was removed from this pilot.

Authoritative sources: COHORT001_cohort.json sample IDs/participant fields,
BASE002 prediction_schema.json and ANALYZE002 analysis.json. The predeclared
bootstrap implementation derives groups from actual participant IDs; it used
all five groups and2000 replicates, seed1701. Paired accuracy differences,
support subsets, metric denominators, source/window coverage and ranks are
unchanged. There is no scientific rerun or result invalidation for this
descriptive error. The small-group limitation is stronger than previously stated.

P001_v3, D003, selected earlier session/novelty descriptions and ANALYZE002
registration retain their historical wording with this explicit correction.
The issued P001_v4_launch_gate.json also retains its original descriptive
limitation and hash because BASE002 records that exact gate hash. Its phrase
"six dev participants" is invalidated as a cohort-count statement; the gate's
cache/config/integrity bindings remain correct. Current reports must use five.
P001_v1/global dataset metadata and source S10 describing the whole split do
not require correction. No source, scorer, test access or statistical recipe
changes; no silent edits of completed runtime artifacts.
