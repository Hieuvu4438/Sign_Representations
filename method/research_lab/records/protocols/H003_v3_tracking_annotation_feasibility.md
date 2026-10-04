# H003_v3 — Auxiliary tracking annotation metadata feasibility

[D020](../decisions/D020.md) prelaunch; TRACKREL001 config
86c265a8cef0b3837d3afa4f0da2a9a9a6c943b588cc7cd92a09f473db65d4fe.
Question: can official older RWTH manual annotation schema/IDs join local
PHOENIX2014T train/dev without invented offset or language equivalence?
Primary software outputs: exact identifier intersection counts/schema/provenance.
Predicted direction unknown; zero intersection is a valid applicability finding.
Baseline: literal officialsource plus exact localCSV identities. Changedvariable:
annotation source availability; source/frame identity is controlled before any
value-based use. No qualitymetric/natural-failure labels/scientificpilot authorized.

Cohort is metadata only: all current localtrain/dev names, no filtering on outcomes.
Seed1701 for run record, no stochastic search/HPO. CPU<=180s,0GPU; download<=2MiB,
expanded<=24MiB/RAMreserve; no full image archive or unrestricted crawl. Retain
config/source/env/actualargv/status/error, officialresponse+hashes, schema/counters,
matchedIDs/frame-dir availability. Stop on cap/XMLentity/hash/time failure; any
subsequent technicalparser correction gets newconfig/run and failedattempt retained.
