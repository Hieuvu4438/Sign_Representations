# SignCoord-SSL (proposal9)

Implementation isolated from existing method folders. Current state is
**engineering in progress; research efficacy unproven**. See
[implementation plan](docs/implementation_plan.md),
[data card](docs/data_card.md) and [evaluation protocol](docs/evaluation_protocol.md).

Run the local validated environment from the repository root:

```bash
export PYTHONPATH="$PWD/method/signcoord_ssl/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv-signrep/bin/python -m pytest -q method/signcoord_ssl/tests
.venv-signrep/bin/python -m signcoord.cli doctor --config method/signcoord_ssl/configs/base.yaml
```

The base doctor intentionally fails until real inputs/checkpoint/backend are
resolved. It does not download anything. For a separate new environment, install
`pip install -e '.[dev,hiera]'` inside this folder, selecting PyTorch compatible
with the actual CUDA driver. `requirements.lock` records the observed local
environment and is not a portable CUDA installer. Wheel construction has been
tested without modifying the shared venv.

Executable fixture preparation and training:

```bash
.venv-signrep/bin/python method/signcoord_ssl/tools/generate_synthetic_fixture.py \
  --output method/signcoord_ssl/data/fixture_new
.venv-signrep/bin/python -m signcoord.cli fit-target-stats \
  --config method/signcoord_ssl/data/fixture_new/fixture_smoke.yaml \
  --output method/signcoord_ssl/data/fixture_new/stats/target_stats.json \
  --phase-output method/signcoord_ssl/data/fixture_new/stats/phase_calibration.json
.venv-signrep/bin/python -m signcoord.cli train \
  --config method/signcoord_ssl/data/fixture_new/fixture_smoke.yaml
```

Use `--arm A1_inspired|A2|A3|A5|A6|A7|A8|A9`, `--seed` and repeated typed
`--set dotted.key=value`. Unknown config keys fail. Use a unique `run_dir` for
each arm/seed; an existing run requires explicit `--resume`. `train --steps N`
stops at a checkpoint boundary while preserving the original total LR schedule;
`smoke --steps N` deliberately changes the total schedule for smoke testing.
Exact resume requires unchanged implementation, scientific/input hashes and
world size; checkpoints contain RNG state for every rank.

Currently implemented commands: `doctor`, `manifest`, `audit-data`, `fit-target-stats`,
`audit-targets`, `smoke`, `train`, `profile`, `extract-features`, `eval-retrieval`,
`export`, `init-reference`, `extract-signrep-reference`. Retrieval takes `--gallery`, `--queries`, `--labels`, `--output`;
features take `--checkpoint`, `--manifest`, `--output`; export takes
`--checkpoint`, `--output`. Exported `infer.py` accepts NPZ RGB/valid/times.
`profile` accepts warmup/measure steps; its fixed predecoded-batch timing includes
optimizer work but excludes data loading/initialization and forces full relation
weight for cost measurement. A short CPU profile is not a GPU budget estimate.

Production pose extractor sharding, exact original SignRep prior reproduction,
probes/translation/CTC CLIs, dev checkpoint selection, compare validation and real
matched experiments remain unfinished. DDP training now supports global validity
weighting, differentiable statistics gather, accumulation, rank RNG and atomic
rank0 checkpoints. Two-rank Gloo tests cover exact resume, missing targets on
one rank, decode/prediction failures and A9 target exchange across ranks.
CUDA/NCCL/AMP execution remains to verify on available hardware.

After generating and fitting a new fixture, launch the CPU distributed check:

```bash
.venv-signrep/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module signcoord.cli train \
  --config method/signcoord_ssl/data/fixture_new/fixture_smoke.yaml \
  --set run_dir=method/signcoord_ssl/runs/A7_fixture_ddp_new
```

`manifest --adapter asl_citizen|wlasl2000|how2sign --root ... --split ... --output ...`
freezes fully decoded source PTS/media hashes, separates labels, and writes an
availability/failure ledger. WLASL requires `--annotation` with pinned upstream
metadata; use `--exclude-groups-from` to exclude known shared source groups.
`--max-classes`/`--max-per-class` select a disclosed metadata-only screen.
`audit-data --perceptual --review-decisions ...` adds a bounded pHash candidate
screen with explicit decisions; it does not establish absence of all reuse.

Generic upstream Hiera uses the pinned K400 video MAE weights and a separate
strict loader. Fetch assets into this folder and materialize the frozen A0:

```bash
.venv-signrep/bin/python method/signcoord_ssl/tools/fetch_generic_hiera.py
.venv-signrep/bin/python -m signcoord.cli init-reference \
  --config method/signcoord_ssl/configs/generic_reference.yaml \
  --output method/signcoord_ssl/runs/A0_new/reference.pt
```

Pass the A0 checkpoint to `extract-features`/`export`; it has zero sign SSL updates
and contains only encoder weights. A0 cannot be trained. Official `A1_ref` uses
`extract-signrep-reference --config configs/signrep_reference.yaml --manifest ...
--output ...`, T16/native-frame stride2/direct resize, and the actual official
`fc`. Video averaging of post-fc windows is a disclosed downstream extension.
Install the `signrep-reference` extra when the official transformation dependency
is absent. It is an external reference with different pretraining/readout, not a matched
objective comparison. A1-inspired predicts the shared 2D geometry sequence from
one pooled vector. A5 needs newly fitted magnitude statistics. A9 deranges only
exactly compatible anchor validity/timing groups and logs skipped samples.
Training A1-inspired/A2/A3/A6 does not require phase calibration.

Native exports include encoder-only deployment code, pinned vendor source and
licenses, relative paths, dependency versions and source hashes. The shipped
`infer.py` runs after relocating the complete bundle without the source repo
on PYTHONPATH. Actual generic T32 FP32 outputs match exactly in a fresh process
that forbids training/pose imports; see `docs/evidence/portable_export_report.json`.

Real local cache provenance is traced in the data card. The bounded converter
uses verified image-coordinate scaling and a disclosed train-only quality
heuristic; original estimator and signer IDs remain unknown. This supports
engineering smoke, not a fully verified strict-inductive research claim.
Current canonical train statistics also bind NPZ contents and the source-gap
policy. Existing `bounds_v4`/`controls_v5`/`source_gaps_v6`/`pose_bound_v7` statistics require
refitting; the current local pilot uses `*_recipe_bound_v8.json`. Interpolation is
disabled for unknown tracks and only fills at most two verified-track gaps.

Selected real manifests now exist for ASL-Citizen (200 train/100 dev, 50 classes)
and WLASL (197 source-filtered train/86 dev, 50 classes). They are subset screens.
A0 ASL-Citizen RGB-only extraction is running from a frozen source capsule under
`runs/A0_ASLCitizen_screen50_capsule_v1`; check `live_step.json` and
`screen_report.json` for execution state. No matched candidate efficacy claim is
available yet.

Runs, tensors, videos, caches and wheel output are ignored by Git. Small execution
reports live in `docs/evidence`; neither existing data/checkpoints nor other
methods are modified.
