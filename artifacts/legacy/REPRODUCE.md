# Reproduce the current execution stage

After the archive move, run relative commands in this document from
`artifacts/legacy`. First run `python3 artifacts/legacy/scripts/setup_runtime.py`
from the repository root. See the root `README.md` for current upstream inference
commands and fresh output directories. Historical locked manifests and result
snapshots are preserved; their absolute paths may refer to the pre-move layout.

This project executes `docs/proposal1/Sign_Representation_AI_Agent_Execution_Plan_VI.md`.
Research is **IN_PROGRESS**. A verified novel method has not yet been established.
Do not call the common lexical cohort a full ASL Citizen reproduction or a novelty result.

All original datasets remain read-only. Generated manifests, caches and logs live here.
GPU 0 was explicitly allocated by the user for one worker on 2026-10-01.

## Inventory and adapters

```bash
python3 bootstrap.py --dry-run
python3 bootstrap.py
python3 scripts/build_manifests.py --dry-run
python3 scripts/build_manifests.py
python3 scripts/validate_manifest.py --manifest data/manifests/all.jsonl --report reports/manifest_validation.json
python3 scripts/audit_splits.py --manifest data/manifests/all.jsonl --output reports/split_audit.json
```

The combined split audit deliberately returns nonzero for cross-split recording groups
in MS-ASL and recording-name candidates in PHOENIX. This is recorded evidence, not
a reason to overwrite official splits. ASL Citizen has a separate clean metadata audit.
Missing files are recorded in `data/manifests/exclusions.jsonl`; never redownload them.
Unresolved How2Sign/WLASL source groups remain unresolved.

## Pinned sources and environment

See `provenance/source_commits.json` and `provenance/assets.json`. Clone source repos
at those exact commits; do not run dataset download wrappers or upstream Slurm scripts.
SignRep checkpoint W01 must match its registered SHA256 before loading.

The verified local environment uses Python 3.13.5, torch 2.11.0+cu128 and torchvision
0.26.0+cu128 inherited read-only from the existing interpreter. Its project venv contains
timm 1.0.20 and albumentations 2.0.8. No packages were installed into conda base.
The full observed dependency list is `provenance/env_signrep.pip-freeze.txt`.
This is a tested compatibility environment, not a bitwise reproduction of the author's environment.

```bash
python3 -m venv --system-site-packages .venv-signrep
.venv-signrep/bin/pip install --no-deps timm==1.0.20
.venv-signrep/bin/pip install albumentations==2.0.8
python3 scripts/fetch_asset.py --asset W01 --dry-run
python3 scripts/fetch_asset.py --asset W01
.venv-signrep/bin/python -m unittest discover -s tests -v
```

On a clean machine, first install the recorded torch/torchvision versions in an isolated
environment rather than assuming they already exist. Exact CUDA/hardware compatibility
must be smoke-tested there. Do not blindly install the full inherited package list.

## Train-only smoke and immutable lexical cohort

```bash
python3 scripts/select_pilot.py
.venv-signrep/bin/python scripts/extract_features.py --manifest data/manifests/pilot.jsonl --backbone signrep --config configs/signrep.yaml --output features/signrep_pilot --resume --determinism-check
python3 scripts/lock_common_protocol.py --dry-run
```

The protocol is already locked in this worktree. Do not rerun its mutation command
here: it correctly refuses to overwrite an existing lock. In a new empty worktree:

```bash
python3 scripts/lock_common_protocol.py
```

The locked cohort has 200 train-selected classes and 2,400/726/1,600 train/val/test clips.
The class vocabulary is fit from train only. Clips remain in official signer-disjoint splits.
Short clips requiring padding are excluded from probe evaluation with an explicit reason.

```bash
.venv-signrep/bin/python scripts/extract_features.py --manifest data/manifests/common_v1.jsonl --backbone signrep --config configs/signrep.yaml --output features/signrep_common_v1 --resume --dry-run
```

Inspect the live extraction PID before launching another GPU job. The running process
and its queue are recorded in `state/common_suite.json`, the feature manifest state,
`runs/experiment_registry.csv`, and `logs/signrep_common_v1_extraction.log`.
For a fresh execution with no live process:

```bash
.venv-signrep/bin/python scripts/extract_features.py --manifest data/manifests/common_v1.jsonl --backbone signrep --config configs/signrep.yaml --output features/signrep_common_v1 --resume
.venv-signrep/bin/python scripts/run_probe.py --config configs/protocol_common_v1.yaml --baseline B00 --seed 0
.venv-signrep/bin/python scripts/run_probe.py --config configs/protocol_common_v1.yaml --baseline B01 --seed 0
.venv-signrep/bin/python scripts/run_probe.py --config configs/protocol_common_v1.yaml --baseline B02 --seed 0
.venv-signrep/bin/python scripts/run_probe.py --config configs/protocol_common_v1.yaml --baseline B03 --seed 0
python3 scripts/update_run_registry.py
```

Learned heads also require seeds 1 and 2. B00/B01 are deterministic and run once.
`run_common_suite.py` waits on an explicitly supplied live extraction PID and runs all
eight jobs serially. It stops on failure; it never retries a crashed run automatically.
Probe commands reject incomplete caches and refuse to overwrite run directories.
Validation selects learning rate, temporal width and epoch; test selects none of them.
No representation adaptation is enabled by this lexical protocol.

## ASL-MTP and the grammar gate

```bash
python3 scripts/resolve_asl_mtp.py --dry-run
python3 scripts/resolve_asl_mtp.py --limit-pages 3
```

ASL-MTP is reserved for external test. Raw video IDs are not canonical utterance IDs.
The resolver requires an exact translation match, recording-prefix agreement and an
explicit frontal view. It never chooses the first ambiguous candidate or downloads
login-gated media. Resolved metadata still requires linguistic/video review.
Published NLL/BLEURT tables are not gold training labels or scope annotations.

Track A now has historical recorded-function train/validation gold and intervals
for a limited internal diagnostic. Scope learning still needs a locked target-domain
and evaluation protocol; source-empty tiers do not establish absence. Track B needs
verified same-language query mappings, gold occurrence boundaries and verified absence.
See `reports/annotation_request.csv` and `reports/gate_decisions.json`.

## Historical NCSLGR internal diagnostic

No additional gold was supplied by the user. Public historical NCSLGR XML research
mirrors and the official 20120129 video index were separately obtained and pinned.
The metadata audit covers all1,887utterances; the locked diagnostic uses222elicited
utterances with exactly one positively recorded NEG/WH/YN function. It does not
infer an absent or affirmative class. See the source commits and registered
NCSLGR assets; original listed dataset roots are never downloaded or modified.

```bash
python scripts/audit_ncslgr.py
python scripts/fetch_ncslgr_cohort.py --reuse-heads --download
python scripts/audit_ncslgr_media.py
```

In a fresh reproduction, also fetch the27registered `NCSLGR_FRONTAL_*` assets
using `scripts/fetch_asset.py --asset <registered_id>`. Their exact official URLs,
SHA256 and sequence metadata are pinned. `evidence/ncslgr/lana_frontal_heads.json`
is the correction list: Lana's camera1 is frontal, camera0 side. The preparation
script applies it and verifies frame clocks/sequence identity. Session view sheets
record the metadata-only visual audit; no expert linguistic review is implied.

```bash
python scripts/prepare_ncslgr_diagnostic.py
.venv-signrep/bin/python scripts/extract_features.py --manifest data/external/ncslgr/diagnostic_v1/manifest.jsonl --backbone signrep --config configs/signrep.yaml --output features/signrep_ncslgr_diagnostic_v1 --resume
python scripts/run_ncslgr_probes.py
python scripts/audit_ncslgr_diagnostic.py
```

Completed protocol/prepared inputs, probes and audit refuse overwriting. Use a fresh
worktree for the complete sequence. Feature extraction supports verified resume.
All444footer-cropped FFV1 inputs passed ROI pixel equality and extraction. Three
768→3linear heads use the same100epochs/3LRchoices and seeds0/1/2; train-only
standardization and validation selection precede test. Pooling uses utterance
support, not gold functional-core boundaries.

The primary split separates source signer and official video sequence. Four XML
archive groups still cross splits. `reports/ncslgr_diagnostic_audit.json` records
post-hoc XML-group bootstrap and27utterance subset sensitivity without refitting.
Small validation classes and one test signer limit inference. Current grammar
results are an internal diagnostic, not verified method novelty or external
semantic comprehension.

## Registered statistics and fixed error sample

`configs/statistics_common_v1.json` was saved before any common test prediction
files existed. It specifies paired recording resampling, all three training seeds,
and a fixed SHA256 sample of 30 IDs from the union of B03 errors.

```bash
.venv-signrep/bin/python scripts/bootstrap_metrics.py --predictions runs/common_v1_B03_signrep_seed0/predictions.csv runs/common_v1_B03_signrep_seed1/predictions.csv runs/common_v1_B03_signrep_seed2/predictions.csv --reference runs/common_v1_B02_signrep_seed0/predictions.csv runs/common_v1_B02_signrep_seed1/predictions.csv runs/common_v1_B02_signrep_seed2/predictions.csv --output reports/common_v1_primary_bootstrap.json
.venv-signrep/bin/python scripts/audit_errors.py --predictions runs/common_v1_B03_signrep_seed0/predictions.csv runs/common_v1_B03_signrep_seed1/predictions.csv runs/common_v1_B03_signrep_seed2/predictions.csv
```

These commands already completed; existing outputs are immutable. A fresh
reproduction must use fresh output paths. Macro recall averages classes with
positive resampled support, matching the probe estimator. Class-absence draws
are counted. Intervals condition on the observed cohort and fixed trained heads;
they do not include training uncertainty or independent signer-population
uncertainty. The report lists every seed, its mean and its training SD.

The failed first error-audit consumer expected a different timestamp key; its
state/log remains archived. Recovery used the actual `window_start_sec` and
`window_end_sec` schema and reran only the failed audit. Predictions and bootstrap
results were preserved. Expert linguistic and visual-identity reviews remain pending.

## Native SHuBERT GPU smoke

The separate Python 3.10 environment borrows the installed h4wpp runtime through
`--system-site-packages`, without installing into that environment. Its observed
dependency lock is `provenance/env_shubert_native.pip-freeze.txt`. Do not blindly
install the entire inherited freeze into conda base.

```bash
/home/haipd/miniconda3/envs/h4wpp/bin/python -m venv --system-site-packages .venv-shubert-native
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv-shubert-native/bin/python scripts/smoke_shubert_encoder.py --output features/shubert_encoder_smoke_fresh --dino
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv-shubert-native/bin/python scripts/prepare_shubert_native.py --manifest data/manifests/pilot.jsonl --output features/shubert_native_preprocessing_fresh
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 .venv-shubert-native/bin/python -X faulthandler scripts/extract_shubert_pilot.py --preprocessing-report features/shubert_native_preprocessing_fresh/report.json --output features/shubert_native_pilot_gpu_fresh --device cuda:0
```

Install the small pinned dependencies shown in
`logs/shubert_dependency_install*.log` into the project venv before these commands.
W02–W07 must match their registered hashes. The DINO architecture is pinned to
`7764ea0f912e53c92e82eb78a2a1631e92725fc8` with `pretrained=False`, then strict-loaded
from the author's teacher checkpoints. No generic DINO weights are substituted.

The final verified pilot is `features/shubert_native_pilot_gpu_attempt4/report.json`.
It uses eight train videos, CPU author preprocessing and one GPU feature worker.
It retains all 12 FFN branch layers as `[L,T,D]`, along with last-layer and fixed
layer-average features. Missing streams have explicit observation masks and are
excluded from pooling/loss; author imputation remains in encoder context.

CUDA must initialize before importing decord in this runtime. Two failed GPU
attempts and isolated import-order checks are preserved in
`reports/shubert_cuda_import_audit.json`. This is a tested compatibility workaround,
not an upstream model change. The public raw-video scripts retain input FPS while
the paper pretraining removes every other frame; downstream preparation is TODO
upstream, so these features do not constitute exact downstream paper reproduction.

## Resumable native common cohort

The native extraction protocol is locked in `configs/protocol_native_common_v1.json`.
Sixty fixed source headers were sampled before the full run; native processing
checks every decoded clip. Preprocessing uses one CPU worker; extraction uses the
sole GPU0worker and can wait for the CPU producer. The locked4,726clip cohort is
unchanged. Generated per-clip audits and feature sidecars include input, code,
checkpoint and stream hashes. Resume rejects changed or corrupt artifacts and
never automatically reruns a terminal failed clip.

```bash
.venv-shubert-native/bin/python scripts/prepare_shubert_cohort.py --manifest data/manifests/common_v1.jsonl --output features/shubert_native_common_v1_preprocessing --dry-run
.venv-shubert-native/bin/python scripts/prepare_shubert_cohort.py --manifest data/manifests/common_v1.jsonl --output features/shubert_native_common_v1_preprocessing --resume
CUDA_VISIBLE_DEVICES=0 .venv-shubert-native/bin/python -X faulthandler scripts/extract_shubert_cohort.py --device cuda:0 --resume --wait-for-preprocessing
```

These two workers are already live in the current worktree: check the exact sessions
in `state/live_handles.json` and process commands before resuming. A fresh initial
preprocessing/extraction omits `--resume`. `--limit-new-clips` bounds integration
validation without changing the manifest. The first2preprocessing clips, then
1resumed clip and3GPU clips completed successfully before the full launch.

Feature frame clocks come from original decoded-frame presentation timestamps,
including variable gaps and initial offsets. Only the final end time may use a
reported packet duration or an explicitly recorded fallback. Native cue crops
preserve frame indices. The cache stores all12FFN branches, last-layer/fixed-average
features and four-stream observation masks. Full extraction PASS will still require
reporting zero-valid clips and matched surviving groups before cross-backbone probes.

## Public-demo11625 translator local CPU smoke

C04source is pinned to69d3d77aa4a4fec89048f5417d44fa717e36f6f8;
W08to578a0233e770c8ce4dc75d859b91fdea7c34f5aa, SHA256
cb605058abf7fd50b80f055fe8a6e271ed64f04ef7b1b450c7a364f88495526a.
All source/config assets are listed in provenance/assets.json. No example videos,
W09base weights, UI imports or external uploads are used. The initial urllibTLS
failure is preserved; standard verified curl transport retrieved the exact same
public asset withHTTP200, expected size and SHA256.

The separate translation venv borrows h4wpp/native dependencies read-only, then
installs transformers4.30.2/tokenizers0.13.3/accelerate1.0.1/datasets3.1.0/
huggingface-hub0.28.1 locally. The observed freeze is
provenance/env_shubert_translation.pip-freeze.txt; no conda base packages changed.

```bash
CUDA_VISIBLE_DEVICES='' .venv-shubert-translation/bin/python -X faulthandler scripts/smoke_shubert_translation.py --output features/shubert_translation_smoke_fresh
```

The completed attempt is features/shubert_translation_smoke_attempt1/report.json.
It uses one104frame frontal prepared NCSLGR clip and its source English caption,
C04cue extraction/stride1 and strict author-DINO weights through the pinned local
architecture. Token NLL includesEOS and ignores padding=-100; manual score agrees
with native loss. Bounded greedy generation is saved verbatim. This verifies a
numerical local adapter, not semantic minimal-pair accuracy or the paper checkpoint.
The embedded demo encoder differs fromW02at213/226tensor values.

## Remaining required work

Full native SHuBERT cohort/cache integration, fair cross-backbone controls,
independent translator evaluation, expert error review, conditional main-track
diagnostics/adaptation, external evaluation, full literature review and the
requirement-by-requirement completion audit remain outstanding. `state/tasks.json`
is the persistent task state. No completion claim is supported at this stage.

## Recorded functional-core interval diagnostic

The immutable protocol is `configs/protocol_ncslgr_intervals_v1.json`; metadata
targets are `data/external/ncslgr/intervals_v1/targets.jsonl`. This is an exploratory
extension locked before interval metrics, after classification test results on
the same cohort had already been observed. The interval target is the positively
recorded source functional core, distinct from fully adjudicated grammatical scope.
Classification uses all222utterances; only179single-event utterances supervise
endpoints (89train/25val/65test). Other43utterances have multiple source events,
retain classification loss, and are never treated as absent.

```bash
CUDA_VISIBLE_DEVICES='' .venv-signrep/bin/python scripts/run_ncslgr_intervals.py --lock-only
CUDA_VISIBLE_DEVICES='' .venv-signrep/bin/python scripts/run_ncslgr_intervals.py
CUDA_VISIBLE_DEVICES='' .venv-signrep/bin/python scripts/audit_ncslgr_intervals.py
```

These commands refuse to replace existing protocols/runs/audits. Completed results
are `runs/ncslgr_intervals_v1/summary.json` and `reports/ncslgr_interval_audit.json`.
A fresh execution needs a separately reviewed version/run namespace or fresh
workspace, preserving existing attempts. Four body/face global/temporal readouts
use6915activeparameters and identical3LR/100epoch/3seed budgets. Gold intervals and
classes are supervision only, never forward inputs. Boundary position distributions
have original timestamp nodes and are not calibrated presence/absence probabilities.
The audit verifies all528boundary files and12prediction tables. Its train-only
class-duration prior comparison is explicitly post-hoc, with fixed selected class
predictions; it cannot replace the locked primary or select another head.

## Equal-grid common-backbone controls

`configs/protocol_fair_common_v1.json` locks SignRep, SHuBERT last FFN branch and
fixed12-layer average. Both learned heads have3LR candidates,100epochs and3seeds;
the temporal width64 is fixed, with parameter matching within0.1%. All controls
use the same eligible source IDs, exclude zero-valid clips explicitly, preserve
missing internal token positions and stop if any locked class loses all support.
Source token rates/receptive fields/crops still differ; this is not a causal
matched-context backbone comparison.

```bash
CUDA_VISIBLE_DEVICES='' .venv-signrep/bin/python scripts/run_fair_common.py --lock-only
CUDA_VISIBLE_DEVICES='' .venv-signrep/bin/python scripts/run_fair_common.py --wait-for-native
```

The original CPU-only waiter has terminated before fitting; inspect `state/live_handles.json` and
`runs/fair_common_v1/status.json` before another launch. It waits for native terminal
PASS, never starts extraction, never evaluates partial native test coverage and
never retries a failed extraction. Tools currently use an isolated PID namespace:
host PID absence or an unavailable old tool session does not prove job exit.
Observe native status/log progress and exclusive worker locks before recovery.
# Native NCSLGR CPU extension and equal-budget controls

The full222body-video cohort resumes its bounded successful pilot. Its input
manifest contains source video IDs/clocks, without functional classes/core/text.
The feature adapter hides CUDA and holds a separate CPU feature-worker lock.
The existing lexical GPU0worker remains the sole project GPU worker.

```bash
.venv-shubert-native/bin/python scripts/prepare_shubert_cohort.py --manifest data/manifests/ncslgr_native_body_v1.jsonl --output features/shubert_native_ncslgr_v1_preprocessing --resume
CUDA_VISIBLE_DEVICES='' .venv-shubert-native/bin/python -X faulthandler scripts/extract_shubert_cpu_cohort.py --protocol configs/protocol_native_ncslgr_v1.json --device cpu --resume --wait-for-preprocessing
CUDA_VISIBLE_DEVICES='' .venv-signrep/bin/python scripts/run_fair_ncslgr.py --wait-for-native
```

These NCSLGR jobs are complete; inspect state/live_handles.json and worker locks
before using the commands. Do not duplicate live workers. The equal-budget suite
waits for full terminal PASS, records all zero-valid exclusions and stops if any
fixed class loses support. All18frozen heads have6915parameters and identical
seeds/LR/epochs. Both native last FFN and fixed all12average are reported, alongside
SignRep; no best test layer is selected. Duration priors use train single-positive
intervals only. Different token rates/context/crops and earlier same-cohort test
observations remain explicit limits: exploratory controls, not encoder adaptation
or established novelty. Protocol lock: configs/protocol_fair_ncslgr_v1.json.
Bounded pilot snapshots: evidence/native_ncslgr_cpu_pilot_v1. The inherited
native-protocol locked_utc timestamp is documented without modifying its hash in
reports/native_ncslgr_cpu_pilot_audit.json.

## Categorical validation-temperature sensitivity

Completed: `.venv-signrep/bin/python scripts/run_ncslgr_calibration.py`. Do not rerun over the existing suite. The protocol was locked with `--lock-only` before its metrics. Nine immutable original heads and their train normalization are reused; temperature selection uses validation NLL only. All nine selected the upper grid boundary8. Test predictions and raw probabilities match the original saved tables, and argmax is unchanged. Reports include NLL, multiclass Brier, five fixed reliability bins/counts and conditional paired-source-group intervals. This is exploratory categorical sensitivity, not established calibrated uncertainty or boundary/presence calibration. Artifacts: `runs/ncslgr_calibration_v1`, `reports/ncslgr_calibration_audit.json`.

## Completed native controls and audited disk recovery

Native NCSLGR222cache,18strict function heads and9author-imputed classification
heads are complete. Original commands above document execution; reruns refuse
existing outputs. New sensitivity was locked using `scripts/run_ncslgr_author_pool.py
--lock-only`, then run with CUDA hidden. Source-context tokens include author
carry-forward/black crops; observation masks remain separate. Strict result files
already existed when this sensitivity was locked; it is exploratory.

```bash
CUDA_VISIBLE_DEVICES='' .venv-signrep/bin/python scripts/audit_native_ncslgr_controls.py
```

This audit is already complete and refuses overwriting its artifact. It checks
222source clocks/caches, reconstructs27savedheads and verifies747boundaryfiles.
See reports/native_ncslgr_controls_audit.json; it is machine integrity, not expert
review. The concrete30case expert packet is reports/ncslgr_expert_review_v1.

Original lexical native extraction ended at the disk guard, not full PASS.
Its terminal status, logs and attempts are preserved; recovery cache verification
is reports/native_common_disk_recovery_audit.json. A single reviewed wrapper is
already running CPU preprocessing after free space exceeded 60 GiB:

```bash
.venv-signrep/bin/python scripts/recover_native_common_disk_stop.py
```

Do not duplicate it. Session89084 and state/native_common_disk_recovery_v1/status.json
record its live handle/phase. CPU preprocessing completes before soleGPU0extraction;
all original hashed code/configs are unchanged and disk reserve remains enforced.
The locked fair suite then writes a separate runs/fair_common_v1_recovery1 using
the original module with only OUTPUT redirected. Original failedwaiter is retained.
Recovery failure is terminal with no automatic retry.

## External semantic-pair CLI readiness

`scripts/run_semantic_pairs.py` is implemented for offline CPU W08author-demo
scoring on independently reviewed, checksummed C04cue caches and external gold.
Current configs/experiments/A01.yaml is BLOCKED_ACCESS, with missing gold/gate/cache
hashes intentionally null. It is not a locked evaluation protocol. The completed
readiness attempt reports BLOCKED_GOLD_OR_ACCESS and exits nonzero before loading
a model. Do not rerun over reports/semantic_pairs_readiness_v1.

```bash
.venv-signrep/bin/python scripts/run_semantic_pairs.py --config configs/experiments/A01.yaml --output reports/semantic_pairs_readiness_v1 --dry-run
```

Unit tests check deterministic candidate swaps/ties, canonical gold/mapping guards,
wrong-recording donor mapping and utterance aggregation. No actual external semantic
accuracy has been measured. Numerical backend integration subsequently passed on
one 104-frame training-signer clip: `features/semantic_pair_backend_smoke_attempt1/report.json`.
Two identical source captions reproduce mean NLL 1.474203467 and tie credit 0.5;
blank streams execute with unchanged source length. This does not verify distinct
semantic pairs or external manifest/gold integration. CandidateEOS is scored, truncation
is forbidden, and half-credit ties are saved. Whole-source video/wrong-video/blank
controls and a fixed shorter-text prior are reported together; no test selection.
Bootstrap first averages pair rows withinutterance/phenomenon, then resamples recipient
recordings. Wrong-video donor reuse adds dependence not covered by that interval.

## Q02 grounding CLI readiness

`scripts/run_grounding.py` implements the standard CPU subsequence-DTW baseline.
The current `configs/experiments/Q02.yaml` is BLOCKED_DATA and deliberately leaves
gold hashes, validation-selected search parameters and registered IoU thresholds
unset. It cannot run inference until all three grounding gates and immutable
manifests are verified. The existing attempt below exited nonzero with
BLOCKED_GOLD_OR_MAPPING before search; preserve its output.

```bash
.venv-signrep/bin/python scripts/run_grounding.py --config configs/experiments/Q02.yaml --output reports/grounding_readiness_v1 --dry-run
```

Eight synthetic unit/integration tests verify middle-subsequence matching, full
query coverage, original-clock gaps, NMS, duplicate detections as false positives,
verified-negative denominators, boundary errors, complete recording resampling,
unreviewed mapping/absence rejection, validation-only parameter selection, and a
complete numerical CLI run retaining an empty-valid-token negative target. They
do not supply linguistic gold or real grounding accuracy. Full integrity checks
now include 59 tests.

A future verified run writes `coverage.json`, `predictions.jsonl`,
`predictions.csv`, `metrics.json`, `summary.json` and `status.json` into a new
project-local output. Metric/search code, source/cache/metadata, protocol and
manifest hashes are recorded. It reports primary macro and pooled AP, registered
secondary IoU metrics, recall, false positives/minute and matched onset/offset
error. Bootstrap retains entire recordings, including negatives; all-negative
draws have undefined AP and are counted explicitly. Scores minimize summed
cosine distance before path-length normalization. Duration filtering uses the
best path per endpoint and does not search alternative constrained paths.

No representation adaptation or method novelty is established by this baseline.

## A03 same-source cue intervention diagnostic

`configs/protocol_ncslgr_branch_ablation_v3.json` fixes eight conditions on the
previously audited 175-utterance common cohort: all streams, without face,
without hands, without body posture, face only, hands only, body posture only,
and zero streams. Source videos, crop files, clocks and observation masks are
shared. Each condition uses the frozen W02 encoder and the same 2,307-parameter
classification head, three seeds, three learning rates and 100 epochs. All 24
heads are selected and saved using validation before any test inference.

V1 failed an eager XML import before model loading; the import was moved to the
readout stage without installing anything in the native environment. V2's
first training smoke passed, but full extraction failed a locked absolute
old-cache difference guard on the second training clip. Both failures, logs,
protocols and code snapshots remain under `evidence/ncslgr_branch_ablation_*`.

The read-only reproduction diagnosis found exact DINO/encoder repeats within
its own run, but did not certify equivalence with old cached features. V3 is a
separately locked experiment: it recomputes its all-stream comparison baseline
in the same run as every intervention. Original-cache differences are reported
separately; the failed old-cache guard is not changed to PASS. The within-run
repeat threshold remains 1e-5. This choice used training-only numerical
evidence before any A03 classification metrics, with earlier NCSLGR results
known; it is exploratory, not independent confirmation.

```bash
.venv-signrep/bin/python scripts/run_ncslgr_branch_ablation.py --lock-only
.venv-shubert-native/bin/python -X faulthandler scripts/extract_ncslgr_branch_ablation.py --limit-new-clips 2
.venv-signrep/bin/python scripts/continue_ncslgr_branch_ablation.py
```

These commands have already been run; do not duplicate live pipeline session
5437. Two fixed training cases passed numerical review; their report, index and
metadata are preserved separately in `evidence/ncslgr_branch_ablation_smoke_v3/`.
The one reviewed continuation completes CPU extraction, checks full 175-clip
PASS, then runs the CPU heads. Incomplete extraction cannot trigger fitting and
a child failure is terminal with no automatic retry. CPU execution uses no GPU;
the existing lexical recovery retains the sole allocated GPU 0 consumer.

The zero-stream condition can retain position/duration and observation-mask
selection signals. Zeroing is an out-of-distribution diagnostic and cannot
identify causal branch importance. There is no A03 classification result until
full extraction and head evaluation finish. Whole-project integrity checks
currently pass 63 tests, including four cue-intervention/continuation tests.
