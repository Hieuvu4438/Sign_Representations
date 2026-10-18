# Pose-JEPA for sign representations

**Archived on 2026-10-15 at the user's request.** Full results, validation history,
official test/probe reports and verified source/provenance evidence are retained in
[the results archive](../../docs/experiments/pose_jepa_results_20261015/REPORT_VI.md).
The entire `runtime/` directory, including weights, caches, features and its Python
environment, has been removed; both background services have been disabled and
removed. Old runs cannot resume without their deleted weights. Source, configs,
tests and documentation remain. Runtime paths and launch instructions below are
historical setup examples; recreate an environment and data before using them.

Independent implementation of [proposal14](../../docs/proposal14/Pose_JEPA_End_to_End_Implementation_Guide.md).
The baseline implementation and downstream adapters are complete. See
[the completion report](docs/COMPLETION_REPORT_VI.md),
[implementation status](IMPLEMENTATION_STATUS.md) and [model card](docs/model_card.md).

The archived ASLC3000 background job is documented in
[ASLC3000_BACKGROUND_JOB_VI.md](docs/ASLC3000_BACKGROUND_JOB_VI.md).
It uses a lingering user systemd service, periodic/graceful checkpoints, automatic
resume, frozen code and a queued final evaluation. From this directory:

```bash
runtime/venv/bin/python scripts/aslc3000_job.py status
runtime/venv/bin/python scripts/aslc3000_job.py stop
runtime/venv/bin/python scripts/aslc3000_job.py resume
```

Full official ASL-Citizen training has its own service and job directory. See
[FULL_ASLC_BACKGROUND_JOB_VI.md](docs/FULL_ASLC_BACKGROUND_JOB_VI.md).
It trains from scratch on 40,154 clips for at most 125,482 updates, selects on
official validation, and queues official test retrieval and a frozen linear probe.
Use `scripts/full_aslc_job.py status|stop|resume` in the same Python environment.

From the workspace root:

```bash
export PYTHONPATH="$PWD/method/pose_jepa/src"
python -m signrepr.cli --help
python -m pytest method/pose_jepa/tests -q -W error
python -m signrepr.cli make_fixture --output method/pose_jepa/runtime/fixture --samples 16 --seed 42
python -m signrepr.cli validate_data --manifest method/pose_jepa/runtime/fixture/manifest.jsonl --report method/pose_jepa/runtime/fixture/audit.json
python -m signrepr.cli pretrain --config method/pose_jepa/configs/ssl_pose_jepa.yaml --profile smoke --device cpu
python -m signrepr.cli export --checkpoint method/pose_jepa/runtime/ssl/last.pt --output method/pose_jepa/runtime/bundle
```

The YAML defaults point to a synthetic fixture for the initial smoke. Replace data
and validation protocols before real training. Paths are relative to the YAML
directory. Synthetic metrics never establish sign-language quality. Stage 1 has
no metric projector; requesting `embedding=metric` without a trained head fails.

The torch training core does not import RGB/MMCV/rtmlib/mT5. Pose extraction and SLT
load their optional dependencies only when invoked. Uni-Sign pickle inputs and
training checkpoints must come from trusted sources. Inference bundles are loaded
as state dictionaries with strict shape/config and checksum validation.

The tested isolated environment is `runtime/venv` (Python 3.10.19, torch 2.5.1+cu121).
Install the exact dependency set on another compatible machine:

```bash
cd method/pose_jepa
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r requirements-tested.lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -e .
```

The lock includes test/SLT/PyAV dependencies. RTMW detection additionally needs the
optional extraction environment and actual weight hashes; injected-detector/PyAV
streaming is tested. External mT5-base config/revision is pinned, weights are not
bundled. Dataset, weights, environment and run directories are excluded from Git.

Use configurations relative to their YAML directory; make a separate output
directory for each run. The guide profiles are `smoke` (20 updates, two encoder
layers/one predictor layer), `pilot` (5,000 updates, full backbone) and `full`.
Resume with the same profile/config/source hash. The completed pilot source
snapshots are available under `runtime/source_snapshots/<code_sha256>/signrepr`.

```bash
python -m signrepr.cli import_unisign --input /trusted/pose_pickles \
  --metadata /metadata/source.jsonl --output /cache/pose --manifest /manifests/imported.jsonl
python -m signrepr.cli validate_data --manifest /manifests/imported.jsonl --report /reports/audit.json
python -m signrepr.cli build_protocol --config configs/data_protocol.yaml --output /manifests/protocol
python -m signrepr.cli pretrain --config configs/ssl_pose_jepa.yaml --profile pilot --device cuda
python -m signrepr.cli pretrain --config configs/ssl_pose_jepa.yaml --profile pilot --resume /runs/last.pt
python -m signrepr.cli extract --checkpoint /runs/best_validation.pt \
  --manifest /manifests/gallery.jsonl --output /features/gallery --embedding raw
python -m signrepr.cli extract --checkpoint /runs/best_validation.pt \
  --manifest /manifests/query.jsonl --output /features/query --embedding raw
python -m signrepr.cli retrieval --gallery /features/gallery --queries /features/query \
  --protocol gloss_max_single_relevant --output /reports/retrieval.json
python -m signrepr.cli train_islr --config configs/islr_frozen.yaml --init /runs/best_validation.pt
python -m signrepr.cli train_islr --config configs/islr_finetune.yaml --init /runs/best_validation.pt
python -m signrepr.cli eval_islr --checkpoint /runs/islr/best_validation.pt \
  --manifest /manifests/islr_test.jsonl --output /reports/islr.json
python -m signrepr.cli train_metric --config configs/metric_gloss.yaml --init /runs/best_validation.pt
python -m signrepr.cli train_cslr --config configs/cslr_frozen.yaml --encoder /runs/best_validation.pt
python -m signrepr.cli train_cslr --config configs/cslr_finetune.yaml --encoder /runs/best_validation.pt
python -m signrepr.cli train_slt --config configs/slt_frozen.yaml --encoder /runs/best_validation.pt
python -m signrepr.cli export --checkpoint /runs/best_validation.pt --output /artifacts/pose_encoder
```

These `/...` paths stand for resources you supply; edit task configs to train on
the audited manifests. Classification/sequence training requires separate held-out
validation. Full continuous CSLR and SLT need their actual transcripts/translations.
Default fixture labels are arbitrary and are flagged synthetic.

The real local sanity artifacts use 16/8/8 official-split ASL-Citizen clips with
an explicitly audited legacy score adaptation. Five full-backbone pilots ran
5,000 updates each; geometry won validation on this tiny subset, so both the
selected geometry and hybrid bundles are retained. Test results do not establish
full-benchmark superiority. No inference-only random fusion/projector is added.

```bash
OMP_NUM_THREADS=2 runtime/venv/bin/python examples/inference_cpu.py \
  --bundle runtime/hybrid_pose_encoder \
  --pose 'runtime/aslc_sanity/poses/train/16135350953831118-APRIL.npz'
```

Source audit, real subset conversion, overlay audit, pilot orchestration, source
update parity, bundle parity/latency and report helpers are under `scripts/`.
