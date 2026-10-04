# H006_v0 — Base-only geometry, class-disjoint dev dictionary

Registered2026-10-04 before outcomes; H006/D027, Codex. **VALIDATED for diagnostic**
measurement by GEOMTEST002 four oracle groups plus unchanged VAL001 metric.
Adapted same-corpus exploratory setting; no official benchmark or confirmation.

## Native data/inference and strict access

BASE002/COHORT001/VID002/CACHE005 supplies base1313train sources/100gloss classes;
REPQUAL001/COHORT002/VID003/CACHE006 supplies target383val queries and three fixed
100instance/100class one-example train galleries. Input configs/manifests/report
and helper SHA are bound in GEOMTRANSFER001 config. Exact metadata audit is
H005_GEOMETRY_TRANSFER_FEASIBILITY.json (no class/source/ID intersections;
new278train/383val; all five val participants reused). No selected-source removal,
padding, H002 policy trial, test input/feature/label/prediction or P004 test rescue.

CacheReader validates full source/timeline/dtype/hash coverage. Base/target
checkpointf8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b,
native inference code/packages/preprocessing/window recipe are exact equal.
Only contract descriptor difference: complete native PTS/FPS lives in separately
hash-bound VID002 vs VID003 audit; both exact strings are whitelisted, every
other field equal. No numerical/timing/stride difference inferred from wording.
Cache manifests respectively4b2217203fb25af9110226e06cb56f7171ae4b46b111b9bf77c049c33ab9e737
and89b0a1b6cd4c115f74cc5204199e6646dd0471662cb86e7432d7189e25f0c70d.

Offline RGB224/native16frames/stride2, saved768FP32 features plus2raw activities,
existing unchanged full_activity FP64 weighted mean/L2 pool; no encoder call.
Fit arrays load base train only; base val hash-verifies only, never fits/loads.
Target arrays load selected-support union+383queries only; unused train shards
hash-verify, not load/fit. Fitting cannot access any target feature/label. Query
vectors transform pointwise with fixed base parameters; inductive, not joint
query normalization/transductive fitting. Official train/val participant IDs
disjoint across base fit and target queries and between gallery/query.

Supervision:1313 fully gloss-labeled base examples and100target support labels
per gallery, same allowed access for all modes. Raw/center/random/total PCA
ignore base labels. Residual and shuffled PCA consume base labels/multisets.
Not globally one-shot-only, no label-free adaptation or pretraining-disjoint claim.
Language ASL, local gloss strings are labels, no semantic/phonological equivalence
or lexical ontology claim. No source/data/feature upload; release terms retained.

## Six fixed geometry modes

Rank8 unchanged from prior control, no search. X base unit clip vectors;
mu=clip-weighted base mean. R=x-minus-its-base-gloss-mean. Leading eight right
singular vectors of R yield residual PCA; X-mu yields total PCA. Shuffled control
randomly permutes all base labels (exact class counts preserved) via default_rng
1701 and computes the same residual SVD. Random control independently uses
default_rng1701 Gaussian768×8 QR. Distinct operations/streams fixed, no seed choice.
Rank8singular<=1e-12, nonorthogonality>1e-10 or transformed norm<=1e-12 aborts
technically, no lower-rank fallback or source exclusion.

- raw: normalize x.
- center: normalize(x-mu).
- random/total/shuffled/residual PCA: normalize((x-mu)-(x-mu)UU^T).

All base IDs/true+shuffled labels/means/bases/singular values saved. No participants
needed to fit, so this does not inherit H005's identity-removal interpretation.
Weighted within-class pair-covariance identity documented in H006_projection_prior.md;
known NAP-style family, no novel projection claim. S34 centering+L2 is precedent,
not reproduced image benchmark/trained SimpleShot encoder.

## Metric, coverage, falsifier and uncertainty

Exact100class cosine rankings against each selected one-example gallery,
descending score/ascending original target class-ID ties. All383 unique queries
per mode/seed:6894complete ranking rows. Original class ordering/label joins and
one-per-class supports asserted, no cherry-picked gallery or query labels in fitting.
Unchanged `metrics.py` SHA43db516fa029b20986130cae6b98e95e500cefa372d15ae4c020130b9f56c79c:
DCG=1/log2(true_rank+1), binary relevance/no nDCG. Official upstream scorer commit
17f0148b00ef87a5957ec103da8d10bd8eb1fa23,eval_metrics, frozenVAL001 oracle/parity
P001_v1_parity.json; no new metric implementation. Gallery/task are diagnostic.

Primary mean DCG over383queries and three seeds. Pass only if residual-minus-raw
>=.01DCG, residual-minus-each(center/random/total/shuffled)>=.005, and no seed's
residual-minus-raw aggregate negative. Fixed thresholds are screening choices,
not selection-adjusted/population significance. Top1/5/10/20/MRR, every seed,
all deltas and failures retained. No best mode/rank/seed selection or sweep.
Conditional paired participant bootstrap2000reps/1701 of gallery-average query
differences, resampling five target val IDs with micro clip weighting; intervals
100×DCG units. Shared supports/prior dev reuse/small five groups prevent broad
population or causal inference. Report all five fixed controls; no p-values.

GEOMTEST001 initial three groups passed; metadata preflight then identifies only
different audit descriptor wording, strict exact whitelist/helper added before
registration. GEOMTEST002 four groups pass: independent weighted-pair covariance,
known residual-vs-total axes/projection, shuffled-label count preservation,
zero-vector failure, class/source/ID exclusivity and forbidden recipe/timeline
drift. Old native/scorer hashes unchanged; CLI smoke passed. No scientific run
failure/retry exists from this prelaunch code refinement.

CPU180s/single BLAS thread/GPU0,64MiB planned RAM +20%reserve/2GiB workspace;
<20MiB durable files, root bulk reserve fails. Exclusive RunContext snapshots,
timeout, actual argv/status/environment/logs/output digests. No overwrite/resume
of terminal runs. Change creates version/run; no gate relaxation after outcomes.
Pass continues contribution review only; no automatic GPU/PAPER_READY promotion.
