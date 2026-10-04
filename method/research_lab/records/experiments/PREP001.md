# PREP001 — Resource and ASL Citizen metadata audit

- Registered: 2026-10-03, before inventory script execution.
- Type: PREPARE audit, no H ID, no scientific metric or test predictions.
- Question: Can existing assets support a train/dev frozen probe without
  missing videos or metadata leakage?
- Inputs: nine documented dataset roots, ASL Citizen CSVs/video directory,
  two primary encoder checkpoints; upstream git/interface identity.
- Scope: split metadata only, including test IDs for split separation audit;
  no test feature extraction, scoring, class selection or model selection.
- Budget: single CPU process, bounded metadata/checkpoint reads; no GPU;
  no corpus download, <10 MiB new small records; no video content hashing.
- Script: `method/research_lab/scripts/inventory.py` (to be created).
- Output: new exclusive run directory under `method/research_lab/runtime/`;
  lightweight summary in `records/inventory/`.
- Required evidence: timestamps, host, script/config hashes, git SHAs,
  checkpoint SHA-256, annotation counts/hashes, video availability,
  missing IDs, split video/participant overlap, ontology coverage,
  disk reserve and GPU allocation visibility.
- Validity: existence and metadata evidence only; filename checks cannot prove
  decode quality, near duplicates, independent signers or pretraining disjointness.
- Status: completed, exit0; [status](PREP001_status.json),
  [inventory](../inventory/PREP001_inventory.json). Findings F001/F002;
  coverage/split checks pass, resource reserve fails; no GPU used.
