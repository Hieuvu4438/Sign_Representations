# QUAL001 — registered H002 cache-based quality screen

Registered before implementation/execution, H002/P002_v1, seed1701.
Full locked1691-source cache,378 dev/100 classes/three800-clip train galleries.
Two dense pooling references plus six fixed reduced-window policies,24 complete
rankings. Primary DCG loss≤.02 and aggregate encoded-window count≤half61339;
no reduction claim from caching. All6 policy trials counted,0GPU; CPU≤120s.
Must pass WIN001, hash-bound protocol gate, CACHE005 integrity and exact dense
mean/activity n8 ranking parity to completed BASE002. No adaptive/test access.
Retain full ranks/schema/source selection hashes/counts/raw seed metrics and
paired five-participant bootstrap intervals, no best-seed headline.
Config QUAL001.json; output method/frozen_probe/runtime/QUAL001_20261004_P002v1_s1701.
Exact command registered once CLI exists, before execution.

WIN00144 checks pass; P002_v1_quality_gate.json validated for this cache/policy/config.
Exact command registered before execution:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 method/frozen_probe/scripts/window_quality.py --config method/frozen_probe/configs/QUAL001.json --seed 1701 --run-id QUAL001_20261004_P002v1_s1701 --output-dir method/frozen_probe/runtime/QUAL001_20261004_P002v1_s1701 --cache-manifest /dev/shm/signrep_frozen_probe_8bfd9b55453cd54f1d7b/manifest.json --launch-gate method/research_lab/records/protocols/P002_v1_quality_gate.json
```

Completed exit0, wall2.317680s CPU,0GPU, all6 policy trials.
Dense reference exact parity passes. Uniform8 DCG.933838893 vs full.934445501,
loss.000606608,15976/61339windows (3.839447× fewer); passes quality/count floor.
Both random8 seeds also pass mean quality/count; uniform4/random4 strict2× floor
fails by rounding (1.972886×). Cached selection still proves no latency saving.
