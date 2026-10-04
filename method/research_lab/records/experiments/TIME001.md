# TIME001 — registered actual selected-window cost diagnostic

Registered before implementation/quality feedback, H002/P002_v1 seed1701,
TIME001.json,≤900s/remaining H0021h/program4h, shared GPU0≥8GiB free/8GiB cap.
Five source IDs frozen from native-window-count quantiles before QUAL001 results.
Full activity plus all6 reduced-window policies,3 paired repeats, counterbalanced
order1701; actual pre-encoder selection. No cached feature calls inside timed
inference. Same source hashes/PTS/features/weights; selected output/cache parity
atol1e-5/rtol1e-4. All raw decode/preprocess/encoder/pool/search/e2e timings,
counts, p50/p95, paired median ratios, model/setup cost and peak VRAM retained.
Median ratio≤.8 and actual sample window-count reduction≥2× cost floor; only
joint quality/count/cost eligibility can support a pilot continuation decision.
No dedicated/streaming/population throughput claim under shared load.
Run only after software, completed cache quality/parity and hash-bound timing
gate; same six search policies, not extra trials. No new bulk payload or installs.
Exclusive output method/frozen_probe/runtime/TIME001_20261004_P002v1_s1701.
Exact command after CLI/preflight exists, before execution. No automatic retries.

Pre-quality estimate: selected cost sources have native window counts[18, 24, 30, 39, 64];
all timed forwards=1722 windows across repeats (plus4warmup). Max GPU004
per-window rate0.048035s, max per-frame preprocessing0.002761s;
60s model/decoding/setup allowance and1.5safety factor estimate250.86s,
within900s smoke/H0023600s. Shared estimate provisional; deadline remains strict.
Selection frozen in TIME001.json before quality feedback.

Quality/cache/software gate satisfied, all policies retained for paired timing.
Actual timed mixed train/dev examples are cost/parity observations only; no
classification quality is scored on them. Source SHA verification and gallery
setup outside latency; class-score ranking inside measured search.
Exact command registered before GPU execution:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-signrep/bin/python method/frozen_probe/scripts/window_timing.py --config method/frozen_probe/configs/TIME001.json --seed 1701 --run-id TIME001_20261004_P002v1_s1701 --output-dir method/frozen_probe/runtime/TIME001_20261004_P002v1_s1701 --cache-manifest /dev/shm/signrep_frozen_probe_8bfd9b55453cd54f1d7b/manifest.json --quality-run method/frozen_probe/runtime/QUAL001_20261004_P002v1_s1701 --launch-gate method/research_lab/records/protocols/P002_v1_timing_gate.json
```

Observed completed exit0, wall59.397070s, GPU0.016028904h.
Original outputs/status/snapshot immutable; durable copies retained.
