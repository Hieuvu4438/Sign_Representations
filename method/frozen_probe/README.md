# Frozen probe prerequisites

Namespace đầu tiên cho H001/H002, chưa phải phương pháp paper được chốt.
Đọc [STATE](../research_lab/records/STATE.md) và
[P001_v4](../research_lab/records/protocols/P001_v4.md) trước measurement.

Code hiện có: strict class-ranking scorer, checkpoint audit, native-PTS frozen
SignRep adapter, CPU preprocessing/batched-feature parity, full cohort decode,
duplicate heuristic, content-verified bounded cache, pooling và deterministic
ridge/class-max controls, participant bootstrap analysis.
[P001_v4](../research_lab/records/protocols/P001_v4.md) validated cho first adapted pilot;
GPU parity/extraction và baseline/prediction/analysis CLIs đã triển khai.
40 CPU/synthetic checks pass; GPU004 numerical parity passes. CACHE005/VERIFY001 full cache pass; BASE002/ANALYZE002 first dev pilot complete.
Ordered67.20% versus activity-nearest78.40% primary; registered utility screen
fails, D007 stops scaling that recipe. H002 registration next.
Scorer không nhận gallery instances, chỉ unique class ranking.

Current configs GPU004/CACHE005/BASE002/ANALYZE002 bind the fixed cohort and
feature recipe. GPU preflight requires granted GPU0≥8GiB free; user confirms shared GPU available (utilization≤100%),
preserves scheduler visibility. GPU smoke gates extraction cost; bounded RAM
cache requires20% tmpfs/hostRAM reserve; prediction CLI requires Protocol Auditor
launch gate plus complete hash-bound cache. Completed outputs are immutable;
partial extraction resume is explicitly unsupported. See experiment cards for
budget and exact dependencies; no scientific launch before gate completion.

Các lệnh đã thực sự chạy từ repository root:

```bash
python3 method/research_lab/scripts/inventory.py --run-id PREP001_20261003 --output-dir method/research_lab/runtime/PREP001_20261003
python3 method/frozen_probe/scripts/validate_scorer.py --run-id VAL001_20261003 --output-dir method/frozen_probe/runtime/VAL001_20261003
PYTHONPATH=method/frozen_probe python3 -m unittest discover -s method/frozen_probe/tests -v
```

Run directories dùng create-exclusive: lệnh inventory/parity trên là lịch sử,
không rerun với cùng output dir. Khi thực sự cần validation mới, cấp run ID
và output path mới; giữ completed/failed runs. Runtime bị Git ignore.
Download parity chỉ 7.081 bytes, không dùng GPU. Mỗi lần đổi scorer/config/split
phải version protocol; tie rule local có disclosure trong card.

Actual cohort has five dev participants; explicit erratum in P001_v4.
No confirmatory/test/official benchmark or novelty claim from this first pilot.

H002/P002_v1 dev quality and actual cost pilot completed (WIN001/QUAL001/
TIME001/REPORT001). See D008 and rebuildable quality-cost table. Uniform8
passes joint floor; stride8 already native ablations, no novel sampler claim.
Confirmatory/test plan must be frozen next; six policy candidates exhausted.


Bounded artifact software (D037): `bounded_artifacts.py` enforces complete
registered logical byte/file limits and atomic replacement scratch; compact
geometry records live in `compact_geometry_artifacts.py`. New CPU-only wrappers
can use `bounded_cpu_context.py` with explicit bound source_snapshots and
artifact_file_caps. It captures all owned run records through the store; complete
file coverage and hashes precede finalized success. Original RunContext and
historical completed runs remain unchanged. New run configs/IDs/paths must be
registered before measurement; do not rerun historical software fixture IDs.
ARTIFACTTEST00114component checks and ARTIFACTCTX0015context checks actually
pass. RAM synthetic records are ephemeral and referenced by their manifests;
these checks do not repair old H006 storage compliance or prove scientific utility.
