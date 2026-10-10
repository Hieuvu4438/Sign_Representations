# SignRel-JEPA

Independent implementation of
[`SignRel_JEPA_End_to_End_Implementation_Spec.md`](../../docs/proposal6/SignRel_JEPA_End_to_End_Implementation_Spec.md).
Everything is isolated in this folder. Other methods, existing datasets, third-party
checkouts and existing environments are read-only inputs.

The research question is whether **inter-articulator visual relation prediction**
improves frozen sign representations over generic or individual-articulator latent
prediction. An SSL loss decrease is not evidence of that improvement.

The engineering MVP and fixed three-seed A/B/C/D pilot are complete. The pilot
does **not** support a consistent relation benefit, so multi-scale training and
scale-up were not started. Read [REPORT.md](REPORT.md) for results and diagnosis,
and [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for the exact evidence.

## Model and data contract

- Teacher sees the full RGB clip; student removes hidden patch tokens **before
  contextual attention**, rather than blanking them after encoding.
- Four articulators: anatomical left/right hand, face, upper body. Deterministic
  region pooling uses tubelet unions and either token centers or patch intersection.
- Five initial edges: LH–RH, LH–Face, RH–Face, LH–Body, RH–Body.
- Teacher relation: `MLP([a,b,a-b,a*b])`, with both encoder and relation projector
  frozen and EMA-updated after successful optimizer steps.
- Cross-attention queries contain edge identity, continuous temporal position/span
  and scale (represented by kernel span). The decoder predicts two latent endpoints
  and passes them through the **student relation projector**. Thus that projector
  receives gradients and provides a trained source for its teacher EMA.
- L1, smooth L1 or cosine latent regression; optional articulator, generic patch
  and variance/covariance objectives. Multi-scale targets are deterministic averages
  with complete-window endpoint validity.
- Generic patch prediction is a **local JEPA baseline**, not a claim to reproduce
  the entire official V-JEPA training recipe.
- RGB-only `encode_video`/frozen extraction uses mean visual tokens, LayerNorm and
  L2 normalization. No landmark detector or relation graph is required at inference.
- SSL dataset, collation and forward reject semantic keys recursively. Dataset and
  language bookkeeping is removed before collation. Evaluation annotations stay in
  separate manifests and never define SSL pairs or losses.

MediaPipe Tasks caches store exact raw-frame indices, source/model hashes,
dimensions, confidence semantics, normalized coordinates and version. The face
search uses a non-destructive shoulder-derived head crop, then maps landmarks back
to original-frame coordinates. Ambiguous anatomical hand assignments remain invalid.
Unknown heatmap score scales from existing local keypoints are not treated as probabilities.

Spatial augmentation shares the actual rounded crop with landmarks/boxes, flips
coordinates and swaps hand identities. Missing intervals longer than three frames
remain invalid; short intervals may interpolate. Structural preflight reports all
excluded clips explicitly.

## Environment

The current machine already has a working interpreter:

```bash
cd /home/haipd/Sign_Representations/method/signrel_jepa
PY=../../.venv-signrep/bin/python
$PY -m pytest -q
```

No package has been installed into that environment by this method. For a fresh
environment, install this project and its optional preprocessing/test dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[preprocess,test,diagnostics]'
```

`requirements.lock.txt` is the observed full package freeze from the working
environment, rather than a claim of cross-platform compatibility. Each run captures
its own freeze, executable, environment, CUDA information, resolved config, source
snapshot and content hashes.

## Prepare data and assets

```bash
$PY scripts/fetch_assets.py
$PY scripts/prepare_manifest.py --dataset how2sign \
  --root /home/shared_data/sign_language/How2Sign/train/raw_videos \
  --limit 100 --seed 42 --probe --out manifests/how2sign_pilot100_ssl.jsonl
$PY scripts/prepare_manifest.py --dataset wlasl --purpose eval --split all \
  --metadata assets/WLASL_v0.3.json --out manifests/wlasl_all.jsonl
$PY scripts/prepare_eval_pilot.py --classes 40
$PY scripts/audit_leakage.py --pretrain manifests/how2sign_pilot100_ssl.jsonl \
  --eval manifests/wlasl_all.jsonl manifests/eval_pilot/asl_citizen_train.jsonl \
  manifests/eval_pilot/asl_citizen_val.jsonl manifests/eval_pilot/asl_citizen_test.jsonl \
  --out reports/leakage_sources.json --clean-out manifests/how2sign_pilot100_clean.jsonl
$PY scripts/extract_landmarks.py --manifest manifests/how2sign_pilot100_clean.jsonl \
  --out-root cache/landmarks_v2 --out-manifest manifests/how2sign_pilot100_landmarks_v2.jsonl
$PY scripts/preflight_dataset.py --manifest manifests/how2sign_pilot100_landmarks_v2.jsonl \
  --config configs/pretrain/relation_mvp.yaml --out-manifest manifests/how2sign_mvp_ready_v2.jsonl \
  --out-report reports/data_preflight_mvp_v2.json --draws 4
$PY scripts/visualize_geometry.py --manifest manifests/how2sign_pilot100_landmarks_v2.jsonl \
  --config configs/pretrain/relation_mvp.yaml --out-root reports/geometry_qa100_v2
```

The initial v1 cache required a face-detector correction. `repair_face_cache.py`
upgrades owned v1 caches efficiently without changing original videos, hand/body
localization or old cache files. New extraction produces v2 directly. Inspect all
four 25-clip contact sheets before research training. See [DATASETS.md](DATASETS.md)
and [PROVENANCE.md](third_party/PROVENANCE.md) for source terms and exclusions.

## Train and compare

```bash
# Tiny engineering verification on RGB videos with the ready debug manifest
$PY scripts/train.py --config configs/pretrain/debug_local.yaml --run-dir runs/my_debug

# Pretrained ViT-B, BF16, four-step research smoke test
$PY scripts/train.py --config configs/pretrain/research_smoke.yaml

# Audit the completed fixed target comparison
$PY scripts/summarize_controlled_pilot.py --split val --out reports/controlled_pilot/summary_val.json
$PY scripts/summarize_controlled_pilot.py --split test --out reports/controlled_pilot/summary_test.json

# Fresh comparison uses its own checkpoint, report and derived feature destinations
$PY scripts/run_controlled_pilot.py --steps 200 --seeds 42 43 44 --splits val \
  --run-root runs/new_pilot --report-root reports/new_pilot

# Resume keeps the original total-step LR/EMA schedule
$PY scripts/train.py --config runs/my_debug/config.resolved.yaml \
  --resume runs/my_debug/checkpoints/latest.pt
```

`--stop-after N` stops at global step N without changing the total training horizon.
Checkpoint state includes all student/teacher/predictor heads, optimizer, scheduler,
AMP scaler, rank-specific RNG, step/epoch, data asset hashes, manifests and source
fingerprint. Resume rejects config, data, code or world-size mismatches. Batch order
and augmentation are keyed by the draw, so worker scheduling cannot alter resumed
samples. Distributed losses are normalized by the global valid-query count.

The completed pilot used source fingerprint `0fe5684…` and its original hybrid
block/tube/random corruption. Its source is preserved in every run's
`source.tar.gz`. Final engineering source `8a29bcb…` fixes complete tube masking,
empty local DDP query handling, feature-study isolation and extraction provenance.
Resume an old checkpoint only with its matching source/config; current source
intentionally rejects that mismatch. Feature extraction still restores its encoder
strictly. New studies cannot overwrite existing feature artifacts without a
matching prior run record. `scripts/train.py --seed N` sets the resolved run seed.

Periodic validation is optional and isolated from SSL batches. Add this block to
a resolved training config when explicitly running a new experiment:

```yaml
trainer:
  eval_every: 2000
evaluation:
  gallery_manifest: manifests/eval_pilot/asl_citizen_train.jsonl
  query_manifest: manifests/eval_pilot/asl_citizen_val.jsonl
  config: configs/eval/retrieval_asl_citizen.yaml
  selection_metric: retrieval.Recall@1
  selection_mode: max
```

The hook freezes the encoder, preserves all training RNG streams, records feature
and encoder hashes, and retains `best.pt` by validation score. Test/train query
splits are rejected for checkpoint selection. DDP broadcasts evaluation failures
to every rank. Best scores and collapse-alarm state survive resume.

The controlled family is:

| Variant | Generic patch | Articulator | Relation | Mask |
|---|---:|---:|---:|---|
| A | 1.0 | 0 | 0 | random |
| B | 1.0 | 0.25 | 0 | random |
| C | 1.0 | 0 | 1.0 | random |
| D | 1.0 | 0.25 | 1.0 | random |

The separate `relation_mvp.yaml` uses structured mask mixture and relation plus
articulator loss. Standalone articulator/relation configs are also provided. Keep
masking fixed when attributing gains to targets; separately compare structured masks.
Each objective, graph, projector formulation, temporal scale and regularizer is
configurable. The multi-scale recipe is **phase 2**, contingent on representation
evidence from the single-scale experiment.

For DDP, use the interpreter's launcher module:

```bash
$PY -m torch.distributed.run --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/pretrain/debug_local.yaml --run-dir runs/ddp_cpu
```

This host has one GPU. Two-rank CPU/Gloo FP32/BF16 and single-GPU BF16 can be verified
here; multi-GPU NCCL needs a machine with multiple GPUs. CPU BF16 disables oneDNN
because its convolution backward is unsupported on this host; reference kernels
still perform the BF16 test. Production checkpointing uses atomic `latest.pt`
replacement, bounding disk usage.

## Frozen evaluation

```bash
$PY scripts/extract_features.py --checkpoint runs/research_smoke/checkpoints/latest.pt \
  --manifest manifests/eval_pilot/asl_citizen_train.jsonl --out features/signrel_train.npz
$PY scripts/extract_features.py --checkpoint runs/research_smoke/checkpoints/latest.pt \
  --manifest manifests/eval_pilot/asl_citizen_val.jsonl --out features/signrel_val.npz
$PY scripts/evaluate_all.py --features features/signrel_val.npz --gallery features/signrel_train.npz \
  --config configs/eval/retrieval_asl_citizen.yaml --out reports/signrel_val.json

# Generic frozen initialization and read-only released SignRep baseline
$PY scripts/extract_features.py --config configs/pretrain/relation_mvp.yaml \
  --manifest manifests/eval_pilot/asl_citizen_train.jsonl --out features/vjepa_train.npz
$PY scripts/extract_features.py --baseline signrep \
  --manifest manifests/eval_pilot/asl_citizen_train.jsonl --out features/signrep_train.npz
```

Retrieval uses class-max cosine for Recall and DCG, plus **instance mAP** over all
eligible gallery videos. It removes self/source overlap, optionally same signers,
and reports queries without positives explicitly. Probes fit standardization only
on training features. Phonology is reported for official signer-independent splits
and a stricter gloss-disjoint partition. Signer leakage uses known signers with
gloss-disjoint training/evaluation; it is not an unseen-signer recognition score.
Minimal-pair distances are exploratory with explicit property coverage. The default
local evaluation points to the existing ASL-LEX CSV in another method as read-only data.
Change that config path for another machine; reference videos are never needed.

`evaluate_all.py` and the periodic hook share the same evaluator. New reports also
include embedding geometry and matched-property minimal-pair retrieval/confusion.
Five-property matching is the default when the evaluation config declares those
properties; the dedicated diagnostic can explicitly select another matching set.

```bash
$PY scripts/diagnose_features.py --features features/controlled_pilot/C_relation_seed42_val.npz \
  --gallery features/controlled_pilot/C_relation_seed42_train.npz \
  --metadata ../../method/p4_signssl/runtime/external/asllex/signdata.csv \
  --out reports/my_feature_diagnostic.json
$PY scripts/visualize_augmentation.py --manifest manifests/how2sign_pilot100_landmarks_v2.jsonl \
  --config configs/pretrain/relation_mvp.yaml --out-root reports/my_augmentation_qa
$PY scripts/diagnose_localization.py --checkpoint runs/controlled_pilot/C_relation_seed42/checkpoints/latest.pt \
  --manifest manifests/how2sign_mvp_ready_v2.jsonl --out reports/my_localization_diagnostic.json
$PY scripts/evaluate_occlusion.py --checkpoint runs/controlled_pilot/C_relation_seed42/checkpoints/latest.pt \
  --manifest manifests/eval_pilot/asl_citizen_val.jsonl \
  --gallery features/controlled_pilot/C_relation_seed42_train.npz \
  --clean-features features/controlled_pilot/C_relation_seed42_val.npz --out reports/my_occlusion.json
```

`diagnose_relation_targets.py` tests leave-one-clip-out constant prediction and
cross-clip endpoint mismatch. `plot_pilot.py` exports signer/linguistic tradeoffs.
The stronger `relation_hide_both: true` mask is available for future failure-mode
experiments; its geometry passed 380 augmented draws, but it has no training result.
See [ABLATIONS.md](ABLATIONS.md) for 27 unrun, explicitly gated recipes.

`run_controlled_pilot.py` writes a live `status.json`, per-run logs, resolved configs,
features and evaluations. On explicit restart it verifies and skips completed runs,
and resumes checkpoints for interrupted training. Poll a confirmed live process
before restarting it. Frozen SignRep is an external baseline, not an equal-pretraining
control. SHuBERT requires separate stream/clock compatibility validation and is not
claimed as reproduced by the RGB extractor.

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for verified gates, remaining
work and evidence. A 40-gloss pilot cannot establish SOTA or broad phonological,
signer-invariant or cross-language claims. Report negative relation results directly;
do not use lower SSL loss or extra objectives to hide them.
