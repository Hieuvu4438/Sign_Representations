# ASLC001 — fixed ASL-Citizen pilot

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-07
- Verification Status: UNVERIFIED
- Protocol Status: frozen before retrieval metrics
- Version Label: ASLC001_v1

User chose a pilot before full-data training. Membership is frozen by seed20261007:
64 eligible official Gloss classes,2 training videos/class (prefer different training
signers),1 validation video/class.128 gallery/pretrain videos and64 validation queries.
No test evaluation. Official CSVs contain40154/10304/32941 train/val/test examples;
35/6/11 signer IDs are disjoint. CSV and membership SHA256 are recorded in inventory.

Pretraining uses only RGB-derived teacher targets; gloss/class/signer metadata does
not enter any training objective. Training labels only define the balanced pilot
selection and later retrieval truth. Existing COCO WholeBody133 keypoints are not
used: the original MediaPipe/pinned3-D teacher pipeline is executed on RGB videos.
Corpus bone medians are calculated on these128 training videos only.

Initialization: official Kinetics MAE encoder, random SignRep projection/decoder/
activity heads, seed20261007. Public YT-SL-trained SignRep weights are not loaded.
Train1000 updates,100 warmup updates,4 views/2 distinct videos per batch,80% masking,
AdamW/Adam,loss weights,layer decay and clipping preserved. BF16 CUDA AMP is an explicit
precision choice. Checkpoints include optimizer/scheduler/EMA/RNG/sampler and hashes.

Step0 and step1000 retrieve from the identical128-video train dictionary, using
64 real validation queries with signers absent from train. Original inference
preprocessing on full RGB frames; every16-frame window at stride2;BF16 compute,
float32 feature transport; video pooling thenL2 and cosine nearest individual video.
Report both mean pooling and raw-activity-weighted pooling; class ranking uses the
best individual gallery video per class. Report Top1/Top5,macro metrics,DCG,MRR,
correct counts and rank statistics, plus per-query predictions.

The initial activity head is untrained, so step0 weighted results have that limitation.
The comparison is descriptive,single seed,and short training. No convergence or
full2731-class benchmark claim. A64-class dictionary is easier than the full task;
these pilot results must not be compared directly with paper/full-dataset scores.

GPU is shared; each own process is capped to.042 of total VRAM (~2.06GiB allocator).
Do not kill or modify other jobs. CPU teacher extraction uses4 persistent worker
processes. Stage hard timeouts are4hours; monitor process status/progress and report
failures. Dataset files remain read-only; outputs stay under runtime/ASLC001.

Commands (workspace root):

```bash
.venv-signrep/bin/python method/jepa_baseline/scripts/asl_citizen_data.py inventory --output method/jepa_baseline/runtime/ASLC001
.venv-shubert-native/bin/python method/jepa_baseline/scripts/asl_citizen_data.py poses --output method/jepa_baseline/runtime/ASLC001 --workers 4
.venv-signrep/bin/python method/jepa_baseline/scripts/run_asl_citizen_pilot.py baseline --output method/jepa_baseline/runtime/ASLC001
PYTHONPATH=method/jepa_baseline/src .venv-signrep/bin/python -m signrep_baseline prepare-corpus --manifest method/jepa_baseline/runtime/ASLC001/source.jsonl --held-out-manifest method/jepa_baseline/runtime/ASLC001/official_held_out.jsonl --training-reference-manifest method/jepa_baseline/runtime/ASLC001/official_train.jsonl --output method/jepa_baseline/runtime/ASLC001/corpus --pose-python .venv-shubert-native/bin/python --weights method/jepa_baseline/runtime/teacher/models
.venv-signrep/bin/python method/jepa_baseline/scripts/run_asl_citizen_pilot.py train --output method/jepa_baseline/runtime/ASLC001
# Interrupted training: add --resume method/jepa_baseline/runtime/ASLC001/train/last.pt
```
