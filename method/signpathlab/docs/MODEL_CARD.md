# Model card — diagnostics and a small development pilot

Phase3 also provides12small-pilot SSL checkpoints (E01–E04 × seeds17/29/43),100updates
each, D96/2encoder layers and1predictor layer, plus E00 frozen visual probe artifacts.
Validation uses16clips/8curated classes from official signer-disjoint metadata. Mean PC Top1
is27.08% for each E02/E03/E04; signature and motion controls tie in all seeds. These are
limited pilot results, not a full ASL Citizen benchmark or a signature-superiority claim.
The actual learning source is archived/hash-linked in `evidence/phase3_pilot_source.json`.

Initial phase2 artifacts are CPU synthetic smoke checkpoints and a small real-data integration
checkpoint after 20 successful updates on two ASL Citizen train clips. They verify the
software path and are not a research model or deployment recommendation. There is no
held-out performance or reproduction claim from that integration gate. The separate
phase3pilot comparison above has its own limited scope.

Diagnostic temporal model: dimension48, two encoder layers, four heads, one predictor
layer. Inputs retain four 384D DINOv2 visual streams plus 28/84/84D static pose streams.
Geometry uses deterministic degree2+anchors+timing targets projected to32D. Inference
uses unmasked online projectors/encoder and still requires static pose extraction.
The model is bidirectional/offline; it is not a causal streaming system.

DINOv2 is loaded strictly from pinned official source and explicit local pretrained
weights; its code and weight SHA-256 are recorded. MediaPipe detector priors are
external pretrained assets. SSL sign labels are absent from loss/sampling; official
train labels are consumed separately by the frozen downstream diagnostic classifier.

The checkpoint includes online/predictor/teacher, optimizer/scheduler/scaler, successful
and attempted counters, config/data/target/implementation identity, rank-local RNG and
sampler state. Resume rejects incompatible identities. Export/deployment packaging and
parity now work for linear, attentive and frozen-visual graphs. The final E02 export
and actual RGB-to-embedding run are recorded in `IMPLEMENTATION_PHASE4_VI.md`;
raw `.pt` checkpoints remain trusted-local artifacts.

Principal limitations: a small curated training/validation pool, sparse hand/geometry
availability, unknown mirror metadata, no final test or full benchmark, no robustness/
phonological study, no real continuous evaluation, and no evidence that signature beats
the matched motion control in the measured pilot.


The final locked-environment confirmation repeats E00–E04 × 3 seeds with the same
registered pool/hyperparameters under `artifacts/verified_pilot`. Each run carries
source/environment/config/cache provenance. Use the phase4 report for its actual
results and artifact IDs rather than mixing historical and final source identities.
