# SIGNREL temporal runbook

Run from this directory. Use these own environments to avoid importing another method's `signrel` namespace. Core and perception share preinstalled packages read-only; editable installs belong only to this folder.

```bash
.venv-core/bin/python -m signrel doctor --output artifacts/environment.json
.venv-core/bin/python -m pytest tests -q
.venv-core/bin/python -m signrel make-synthetic --output data/synthetic --seed 17
.venv-core/bin/python -m signrel train --config configs/smoke_a1.yaml
.venv-core/bin/python -m signrel train --config configs/smoke_a4.yaml
```

Existing run directories require explicit resume or a new output directory. CPU smoke is engineering verification. The frozen-source environment receipts are requirements-core.lock.txt and requirements-perception.lock.txt; system-site-packages reuse is disclosed in docs/DECISIONS.md.

```bash
.venv-core/bin/python -m signrel resolve-sources --config configs/sources.yaml
.venv-core/bin/python -m signrel import-data --config configs/data_local.yaml
.venv-core/bin/python -m signrel audit-splits --config configs/data_local.yaml
.venv-perception/bin/python -m signrel extract-cache --config configs/pilot_a4.yaml --limit-videos 20
.venv-core/bin/python -m signrel audit-cache --config configs/pilot_a4.yaml
.venv-core/bin/python -c 'from signrel.data.suitability import suitability; suitability("data/cache/cache_manifest.jsonl","artifacts/teacher_suitability.json")'
.venv-core/bin/python -m signrel profile --config configs/pilot_a4.yaml
```

Review artifacts/cache_qa before expanding. Preserve preprocessing and sources for all controls. Generic DINO is frozen; no lexical fields enter SSL manifests, caches, or student input. Natural missing observations are excluded, SSL masks retain valid attention positions. Global artifacts/budget_ledger.json applies to all sequential jobs; caps do not reset per run.

```bash
.venv-perception/bin/python -m signrel extract-cache --config configs/pilot_a4.yaml
.venv-perception/bin/python -m signrel extract-cache --config configs/extract_ssl_dev.yaml
.venv-perception/bin/python -m signrel extract-cache --config configs/extract_eval_dev.yaml
.venv-core/bin/python -m signrel train --config configs/pilot_a1.yaml
.venv-core/bin/python -m signrel train --config configs/pilot_a4.yaml
.venv-core/bin/python -m signrel train --config configs/pilot_a4.yaml --resume runs/pilot_a4_seed17/checkpoints/last.pt
```

A completed checkpoint returns already_complete. Resume validates exact code/config/source/cache/split/protocol hashes and restores RNG/optimizer state. Fixed update budget selects the last completed checkpoint. Changing the budget or method requires a new config/run. Default bf16 falls back to fp32 on unsupported CUDA; CPU smoke requires fp32. DDP is intentionally disabled.

```bash
.venv-core/bin/python -m signrel export-embeddings --config configs/eval_dev_a0.yaml
.venv-core/bin/python -m signrel export-embeddings --config configs/eval_dev_a1.yaml
.venv-core/bin/python -m signrel export-embeddings --config configs/eval_dev_a4.yaml
.venv-core/bin/python -m signrel eval-retrieval --config configs/eval_dev_a0.yaml
.venv-core/bin/python -m signrel eval-retrieval --config configs/eval_dev_a1.yaml
.venv-core/bin/python -m signrel eval-retrieval --config configs/eval_dev_a4.yaml
.venv-core/bin/python -m signrel aggregate --config configs/aggregate_pilot.yaml
.venv-core/bin/python -m signrel eval-minimal-pairs --config configs/minimal_pairs_real.yaml
```

Export `.npy` vectors and separate `.jsonl` indices; retrieval joins labels only at evaluation. Whole/half physical-time means follow per-canonical-frame aggregation of overlapping windows. Learned embedding dimension is 2304; A0 is 2346. Rank unique glosses by maximum exemplar cosine with fixed K=1. Coverage and exact duplicates/signers are audited. Missing natural minimal-pair annotations return BLOCKED_EXTERNAL with no numeric score; synthetic annotations are explicitly synthetic.

Use the inference API with a trusted local checkpoint:

```python
from signrel.eval.embed import embed_cached
result = embed_cached("path/to/audited_recording.npz", "results/pilot_a4_seed17_dev/encoder.pt")
# result: status, embedding, video_id, encoder_sha256, preprocess_fingerprint, coverage
```

The dedicated encoder artifact drops state decoders. It accepts encoder-state keys with only decoder keys absent. Full train checkpoints include decoders, optimizer and RNG. Raw video inference additionally requires the pinned MediaPipe and DINO pipeline.

Locked-test evaluation refuses to run without artifacts/confirmatory_freeze.json binding every test eval config hash and protocol hash. Freeze all configs and checkpoint selection before test scores. Phase7 matrix must have a measured allocation plan and three paired A1/A4 seeds; optional controls do not automatically launch from aggregate. Build-report reads existing artifacts only. `--dry-run` parses a non-training config without side effects; train dry-run validates a sample/model forward without checkpoint creation.

Dependencies or annotations unavailable to an optional upstream adapter are recorded explicitly. Do not pass generic DINO cache to pretrained SHuBERT or treat SignRep weights as a causal control. Do not compare the current K=1 subset scores directly to published or previous-method scores with different galleries.
