# PARTREMOVE001 — H005_v1 held-gloss task diagnostic

Prelaunch registration retained; **completed**, H005/H005_v1/D025/D026. Question: does one known participant-
associated removal improve unseen-participant lookup beyond generic geometry?
Five fixed modes, rank8, SHA50/50gloss split, all378dev sources, three fixed
one-example galleries, full native pooled vectors; no parameter selection.
Primary metric/gates/uncertainty/supervision in immutable protocol/hypothesis.
CPU180s/0GPU, seed1701 (random basis1701+fold),2000 group-bootstrap repetitions,
64MiB RAM planned/<20MiB durable evidence, exclusive source snapshot.
PARTREMTEST001 four oracle groups and CLI --help pass; old scorer unchanged.
Actual config/run/argv/environment/predictions/bases/hash/status/cost and all
outcomes follow; no illustrative science numbers.

## Actual attempt and validity

RunPARTREMOVE001_20261004_H005_v1_1fab462b_s1701, session52963 terminal exit0,
PID2857892/hostptitiec,2026-10-04T08:28:19.160737→08:28:20.403979UTC.
CPU wall1.2432625030633062s/GPU0; NumPy2.5.3, one BLAS thread; no CUDA,
new encoder call, test access, install or source/shared-env change. No retry.
Peak RSS not instrumented;64MiB is planned workspace, not measured peak.
Config SHA1fab462ba128a0e2c22d1c1849882a7e767d59de4da7321ee21843dd58dfbebf.

Actual argv (OMP/OPENBLAS/MKL_NUM_THREADS=1):

```
/home/haipd/Sign_Representations/.venv-signrep/bin/python method/frozen_probe/scripts/audit_participant_removal.py --config method/frozen_probe/configs/PARTREMOVE001.json --seed 1701 --run-id PARTREMOVE001_20261004_H005_v1_1fab462b_s1701 --output-dir method/frozen_probe/runtime/PARTREMOVE001_20261004_H005_v1_1fab462b_s1701
```

Runtime `method/frozen_probe/runtime/PARTREMOVE001_20261004_H005_v1_1fab462b_s1701`:
original source_snapshot/config/environment/status, four outputs and sibling.log.
Report SHA1b17a98edb0cd46e1b1e21022de26329b7a2c6774ef4fcc5ddd650de7c723537;
all5670rank rows SHAde44f5846aa994c071f605bf6d58fa161753efc5436b1f57fa45df6c9663d8ed;
fold metadataa7c5ced2a4f3434f0f49b0e4746032a3b3c335542e3a42141381b89a7c82a0a4;
fitted basis2639c8228c3a73f5d8649ef402d94d41c41187aa51a7b0c9069a2df2fcb1114b.
`PARTREMOVE001_verification.json` independently recounts all full rankings/DCG,
class/split/support/source/fitted basis orthogonality, input/output digests and
gate. Four oracle groups passed. Raw ID metric exactly equals v0. All378queries
per mode/gallery,50class ranks, no exclusions. Valid exploratory diagnostic.

## Registered comparison results

| Mode | DCG s1701 | DCG s1702 | DCG s1703 | Mean DCG | Secondary ID balanced recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | .8241431288 | .7900824174 | .7682808765 | .7941688076 | .1938131684 |
| center | .8206345489 | .7850379175 | .7653500135 | .7903408266 | .1698144189 |
| random | .8216357815 | .7853924440 | .7664068558 | .7911450271 | .1696691995 |
| residual PCA | .8612005643 | .8355012561 | .8140355704 | .8369124636 | .1050606010 |
| participant | .8394824196 | .8065227617 | .7870317559 | .8110123124 | .1007320272 |

Participant-minus-raw+.0168435048,center+.0206714857,random+.0198672853,
but residual-PCA−.0259001512. ID balanced center-minus-participant .0690823917;
all gallery raw deltas nonnegative. **Overall gate fails**, because the fixed
PCA control wins; no rank/gate/seed relaxation. Conditional five-participant95%
bootstrap for participant-minus-PCA [-4.2822691741,-1.7510057411] in100×DCG units,
not population/significance or causal proof. Other four comparisons, full fold
metrics and Top1/5/MRR retained in original report; no seed cherry-picking.

Residual PCA was a preregistered control. Its descriptive mean gain over raw
.0427436560 is a useful next-question signal; it was not the primary success
endpoint and does not rescue H005_v1 or establish a new method. Bases learned
from650/663 fully labeled base-class sources plus50gallery supports per direction,
so no globally one-shot-only gain claim. Reused data/five val participants and
different50class dictionaries prevent comparison to original100class H002.
D026 kills scaling of this participant-specific recipe and continues generic
within-gloss geometry/transfer investigation. No harmful-signer causality,
linguistic preservation, privacy, cross-domain or conference novelty conclusion.
