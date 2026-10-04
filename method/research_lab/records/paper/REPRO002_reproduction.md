# Reproduction index

Generated from completed statuses/configs/output digests/source snapshots; no task scores read.

| Run | Protocol | Seed | GPU seconds | Wall seconds | Status and exact command |
| --- | --- | ---: | ---: | ---: | --- |
| CACHE005_20261004_P001v4_s1701 | P001_v4 | 1701 | 2476.598727 | 2477.870906 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/CACHE005_20261004_P001v4_s1701/status.json) |
| VERIFY001_20261004_P001v4_s1701 | P001_v4 | 1701 | 0.000000 | 0.470857 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/VERIFY001_20261004_P001v4_s1701/status.json) |
| BASE002_20261004_P001v4_s1337 | P001_v4 | 1337 | 0.000000 | 5.493169 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/BASE002_20261004_P001v4_s1337/status.json) |
| ANALYZE002_20261004_P001v4_s1701 | P001_v4 | 1701 | 0.000000 | 0.169760 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/ANALYZE002_20261004_P001v4_s1701/status.json) |
| QUAL001_20261004_P002v1_s1701 | P002_v1 | 1701 | 0.000000 | 2.317680 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/QUAL001_20261004_P002v1_s1701/status.json) |
| TIME001_20261004_P002v1_s1701 | P002_v1 | 1701 | 57.704055 | 59.397070 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/TIME001_20261004_P002v1_s1701/status.json) |
| REPORT001_20261004_P002v1_s1701 | P002_v1 | 1701 | 0.000000 | 0.016999 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/REPORT001_20261004_P002v1_s1701/status.json) |
| CONFTEST001_20261004_P003v1_s1337 | P003_v1 | 1337 | 0.000000 | 0.533554 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/CONFTEST001_20261004_P003v1_s1337/status.json) |
| THR001_20261004_P003v1_s1701 | P003_v1 | 1701 | 0.000000 | 0.422089 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/THR001_20261004_P003v1_s1701/status.json) |
| PRETEST001_20261004_H002_P003v1_741b0823_s1701 | P003_v1 | 1701 | 0.000000 | 147.330352 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/PRETEST001_20261004_H002_P003v1_741b0823_s1701/status.json) |
| CONFTEST002_20261004_P003v1_s1337 | P003_v1 | 1337 | 0.000000 | 0.474809 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/CONFTEST002_20261004_P003v1_s1337/status.json) |
| REPRO001_20261004_P003v1_s1701 | P003_v1_evidence_index | 1701 | 0.000000 | 0.036570 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/REPRO001_20261004_P003v1_s1701/status.json) |
| THRREPRO001_20261004_P003v1_s1701 | P003_v1_registration_rebuild | 1701 | 0.000000 | 0.548757 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/THRREPRO001_20261004_P003v1_s1701/status.json) |
| COVTEST001_20261004_A001_s1337 | A001_input_coverage | 1337 | 0.000000 | 0.035419 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/COVTEST001_20261004_A001_s1337/status.json) |
| COVER001_20261004_A001_s1701 | A001_input_coverage | 1701 | 0.000000 | 0.381107 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/COVER001_20261004_A001_s1701/status.json) |
| COVFIG001_20261004_A001_s1701 | A001_input_coverage_figure | 1701 | 0.000000 | 0.816389 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/COVFIG001_20261004_A001_s1701/status.json) |
| COVFIG002_20261004_A001_s1701 | A001_input_coverage_figure | 1701 | 0.000000 | 0.647956 | [status](/home/haipd/Sign_Representations/method/frozen_probe/runtime/COVFIG002_20261004_A001_s1701/status.json) |

Actual commands below are historical evidence; completed output directories are immutable.
Replay needs the recorded source snapshot/config and a fresh registered output/run ID.
Thread environment and scheduler CUDA visibility are stored in each environment record.

**CACHE005_20261004_P001v4_s1701**

```bash
/home/haipd/Sign_Representations/.venv-signrep/bin/python method/frozen_probe/scripts/extract_cache.py --config method/frozen_probe/configs/CACHE005.json --seed 1701 --run-id CACHE005_20261004_P001v4_s1701 --output-dir method/frozen_probe/runtime/CACHE005_20261004_P001v4_s1701 --smoke-run method/frozen_probe/runtime/GPU004_20261004_P001v3_s1701
```

**VERIFY001_20261004_P001v4_s1701**

```bash
/home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/verify_cache.py --config method/frozen_probe/configs/VERIFY001.json --seed 1701 --run-id VERIFY001_20261004_P001v4_s1701 --output-dir method/frozen_probe/runtime/VERIFY001_20261004_P001v4_s1701 --extraction-run method/frozen_probe/runtime/CACHE005_20261004_P001v4_s1701
```

**BASE002_20261004_P001v4_s1337**

```bash
/home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/run_baseline.py --config method/frozen_probe/configs/BASE002.json --seed 1337 --run-id BASE002_20261004_P001v4_s1337 --output-dir method/frozen_probe/runtime/BASE002_20261004_P001v4_s1337 --cache-manifest /dev/shm/signrep_frozen_probe_8bfd9b55453cd54f1d7b/manifest.json --launch-gate method/research_lab/records/protocols/P001_v4_launch_gate.json
```

**ANALYZE002_20261004_P001v4_s1701**

```bash
/home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/analyze_baseline.py --config method/frozen_probe/configs/ANALYZE002.json --seed 1701 --run-id ANALYZE002_20261004_P001v4_s1701 --output-dir method/frozen_probe/runtime/ANALYZE002_20261004_P001v4_s1701 --prediction-run method/frozen_probe/runtime/BASE002_20261004_P001v4_s1337
```

**QUAL001_20261004_P002v1_s1701**

```bash
/home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/window_quality.py --config method/frozen_probe/configs/QUAL001.json --seed 1701 --run-id QUAL001_20261004_P002v1_s1701 --output-dir method/frozen_probe/runtime/QUAL001_20261004_P002v1_s1701 --cache-manifest /dev/shm/signrep_frozen_probe_8bfd9b55453cd54f1d7b/manifest.json --launch-gate method/research_lab/records/protocols/P002_v1_quality_gate.json
```

**TIME001_20261004_P002v1_s1701**

```bash
/home/haipd/Sign_Representations/.venv-signrep/bin/python method/frozen_probe/scripts/window_timing.py --config method/frozen_probe/configs/TIME001.json --seed 1701 --run-id TIME001_20261004_P002v1_s1701 --output-dir method/frozen_probe/runtime/TIME001_20261004_P002v1_s1701 --cache-manifest /dev/shm/signrep_frozen_probe_8bfd9b55453cd54f1d7b/manifest.json --quality-run method/frozen_probe/runtime/QUAL001_20261004_P002v1_s1701 --launch-gate method/research_lab/records/protocols/P002_v1_timing_gate.json
```

**REPORT001_20261004_P002v1_s1701**

```bash
/home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/summarize_window_budget.py --config method/frozen_probe/configs/REPORT001.json --seed 1701 --run-id REPORT001_20261004_P002v1_s1701 --output-dir method/frozen_probe/runtime/REPORT001_20261004_P002v1_s1701 --quality-run method/frozen_probe/runtime/QUAL001_20261004_P002v1_s1701 --timing-run method/frozen_probe/runtime/TIME001_20261004_P002v1_s1701
```

**CONFTEST001_20261004_P003v1_s1337**

```bash
/home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/CONFTEST001.json --seed 1337 --run-id CONFTEST001_20261004_P003v1_s1337 --output-dir method/frozen_probe/runtime/CONFTEST001_20261004_P003v1_s1337
```

**THR001_20261004_P003v1_s1701**

```bash
/home/haipd/miniconda3/bin/python3 -
```

**PRETEST001_20261004_H002_P003v1_741b0823_s1701**

```bash
/home/haipd/Sign_Representations/.venv-signrep/bin/python method/frozen_probe/scripts/prepare_confirmatory.py --config method/frozen_probe/configs/PRETEST001.json --freeze method/research_lab/records/protocols/P003_v1_freeze.json --seed 1701 --run-id PRETEST001_20261004_H002_P003v1_741b0823_s1701 --output-dir method/frozen_probe/runtime/PRETEST001_20261004_H002_P003v1_741b0823_s1701
```

**CONFTEST002_20261004_P003v1_s1337**

```bash
/home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/CONFTEST002.json --seed 1337 --run-id CONFTEST002_20261004_P003v1_s1337 --output-dir method/frozen_probe/runtime/CONFTEST002_20261004_P003v1_s1337
```

**REPRO001_20261004_P003v1_s1701**

```bash
/home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/build_reproduction_index.py --config method/frozen_probe/configs/REPRO001.json --seed 1701 --run-id REPRO001_20261004_P003v1_s1701 --output-dir method/frozen_probe/runtime/REPRO001_20261004_P003v1_s1701
```

**THRREPRO001_20261004_P003v1_s1701**

```bash
/home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/rebuild_registration.py --config method/frozen_probe/configs/THRREPRO001.json --seed 1701 --run-id THRREPRO001_20261004_P003v1_s1701 --output-dir method/frozen_probe/runtime/THRREPRO001_20261004_P003v1_s1701
```

**COVTEST001_20261004_A001_s1337**

```bash
/home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/software_smoke.py --config method/frozen_probe/configs/COVTEST001.json --seed 1337 --run-id COVTEST001_20261004_A001_s1337 --output-dir method/frozen_probe/runtime/COVTEST001_20261004_A001_s1337
```

**COVER001_20261004_A001_s1701**

```bash
/home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/analyze_input_coverage.py --config method/frozen_probe/configs/COVER001.json --seed 1701 --run-id COVER001_20261004_A001_s1701 --output-dir method/frozen_probe/runtime/COVER001_20261004_A001_s1701
```

**COVFIG001_20261004_A001_s1701**

```bash
/home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/plot_input_coverage.py --config method/frozen_probe/configs/COVFIG001.json --seed 1701 --run-id COVFIG001_20261004_A001_s1701 --output-dir method/frozen_probe/runtime/COVFIG001_20261004_A001_s1701
```

**COVFIG002_20261004_A001_s1701**

```bash
/home/haipd/miniconda3/bin/python3 method/frozen_probe/scripts/plot_input_coverage.py --config method/frozen_probe/configs/COVFIG002.json --seed 1701 --run-id COVFIG002_20261004_A001_s1701 --output-dir method/frozen_probe/runtime/COVFIG002_20261004_A001_s1701
```

Terminal owned GPU cost: 2545.048626s; live attempts: 1 (cost unreconciled).

This index verifies digests and snapshots. It does not establish scientific validity, test completion, population generalization or PAPER_READY.
