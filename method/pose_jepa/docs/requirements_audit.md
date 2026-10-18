# Requirement-by-requirement implementation audit

The scope is the proposal14 baseline repository, executable pipeline, correctness
gates and qualified experiments. Conditional performance claims remain qualified.

| Guide requirement | Authoritative implementation/evidence |
|---|---|
| §§1–3 method/source/supervision boundaries | `method_provenance.md`, `codebase_review.md`, `sources.lock.json`; scratch defaults and export objective names distinguish controls |
| §4 isolated environment/layout | Python 3.10.19 / torch 2.5.1+cu121 in `runtime/venv`; exact lock/environment report; every named source/config/test family present |
| §5 manifest/splits/no SSL labels | `data/schema.py`, `splits.py`, `pose_dataset.py`; actual data audits; label mutation yields bitwise identical SSL losses/update state |
| §6 raw/cache/tracking/mapping/timing | `import_unisign.py`, `extract_pose.py`, `person_tracking.py`; mapping arrays, crossing/disappearance, variable PTS/missing-frame/raw-score tests |
| §7 normalization/local-global/validity/sampling | Full-source fixed shoulder reference, local roots plus anchors, quality/missingness; contiguous SSL and whole isolated subsampling; reference/padding/support tests |
| §8 GCN/tokenizer | Fixed graph topology, spatial-only messages, channel LN, shared hand module, ordered flatten/project, 2-frame validity flags; full-backbone pilots and leakage tests |
| §9 teacher/student/predictor/output | Encoder initialized once, teacher deepcopy/FP32/eval, packed original C indices, full-lattice mask queries, invalid rows skipped/zeroed; raw 4-part mean concat and sequence 1024 |
| §10 mask/no leakage/attention | M/C exact partition, per-time/hand/context constraints, deterministic RNG; private coordinate perturbation keeps visible output invariant; temporal-mixing negative control fails invariant |
| §11 CE/physical target/zero support | FP32 log-softmax/center, explicit numerator/denominator, local-root exclusion, global motion with joint endpoints/PTS/continuity, known arithmetic tests and all-zero skip |
| §12 EMA/optim/DDP/resume | Deduplicated alias EMA, no-decay groups, unscaled clipping, unified overflow, whole-effective-batch denominators/center; actual Gloo references, CUDA AMP, atomic exact resume/prefetch cursor |
| §13 defaults/profiles | `ssl_pose_jepa.yaml`, config schema; smoke and full-backbone 5,000-update pilot; explicit runtime overrides/effective batch recorded |
| §14 metric stage | Namespaced verified labels, distinct-clip P*K and signer preference, autograd gather/local anchors; head freeze/warmup, joint replay/control, online encoder+head parity; 3 equal short controls |
| §15 ISLR | Raw frozen linear probe and full fine-tune; train-only vocab, Top1/5 micro/macro/failures, validation-only selection; real overfit/frozen workflows |
| §16 retrieval | Exact gloss-max and video multi-positive protocols, deterministic ties, invalid/absent-gloss failures, chunked scores, fixed denominators; test fixtures and raw test reports |
| §17 sequence | Unique canonical segment stitching, explicit time/index/validity caches; full utterance CTC/mT5; repeated-label feasibility/blank decode; differentiated long stitching and cache disabled during CSLR encoder updates |
| §18 inference/checkpoint | `SignRepresentationModel` CPU API, separate stripped state bundle, no random projector/fusion, matched Stage 2 online pair, malformed/no-observation status, objective/hash metadata; actual source-load parity |
| §19 controls/selection | All named same-encoder loss/scratch configs, priority ablations; 5 completed pilots, label/no-replay controls; selected geometry by validation, test retained independently |
| §20 CLI | All required subcommands implemented; fixture/converter/audit/protocol/SSL/extract/retrieval/ISLR/metric/CSLR/SLT/export exercised by parser, workflow tests and real runs |
| §21 correctness/learning/deliverables | JUnit/JSON/logs; actual multi-process/AMP tests, private-support negative control, no label/depth leakage, weighted references, overfit 16 real train clips, 25,000 pilot updates, shuffled collapse diagnostic |
| §22 M0–M8 | `IMPLEMENTATION_STATUS.md` distinguishes code/test/train/evaluation and data limitations |
| §§23–24 extensions/kernels | Optional extensions remain disabled; required EMA/CE/physical/DDP/retrieval math implemented and checked against independent references |
| §§25–26 source/hand-off fidelity | Primary pinned sources, actual checksum/load states, full method provenance, configs/locks/logs/checkpoints/bundles/CPU example/completion report |

Named artifact details: `HybridCriterion` is in `losses/hybrid.py`; shared sequence
workflow is in `training/sequence_tasks.py`; scientific provenance is shared by all
trainers in `training/provenance.py`; persistent feature mapping is in
`evaluation/cache.py`. These support the specified module APIs without changing
the recipe. Package discovery includes `src/signrepr/data` (runtime data ignore
patterns are anchored at project root).

Conditional/unexecuted research deliverables: full-scale corpus metrics, full-budget
Stage 2, NMF/minimal pairs, new RTMW real-weight benchmark and real continuous
CSLR/SLT metrics. Data/annotation/extractor provenance and independent budgets are
needed. The code/adapters and fixture paths are implemented/tested; no numbers are
filled in for these missing experiments. The main recipe retains its auxiliary
losses; no minimal-pair superiority/removal conclusion is drawn.
