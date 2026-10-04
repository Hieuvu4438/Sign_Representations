# ARTIFACTTEST001 — Compact artifact software validation

Registered before software outcomes, D035. No hypothesis, task measurement,
training, model inference, HPO, original output conversion or score access.
New BoundedArtifactStore requires all artifact file caps plus manifest and peak
atomic-replacement scratch <= ceiling before exclusive directory creation. All
writes declared/capped, immutable except declared status replacements; JSON
finite, NPYv2 exact shape/dtype/header/payload size, no pickle/object arrays.
Unexpected/missing/modified/symlink files and digest drift reject finalization.
Complete caller integration is required before another scientific run: this
component cannot constrain callers that bypass it or other processes. Logical
bytes differ from filesystem allocation/RAM. Partial I/O retains evidence and
requires a new run; this is not an OS quota or arbitrary-process sandbox.

Compact schema stores three covariance upper triangles, mean/eigenvalues,
int16 zero-based fullclass ranks and six FP64 row metrics; shared class/mode/
support/query/label/participant metadata fixes Cartesian order. Exact symmetry
including signed zero; reject asymmetric covariance instead of averaging. ID
and per-row metric values preserved; whitening is derived and omitted, so
future eigen-reconstruction has no cross-environment bitwise/rank guarantee.
No inference that bytes imply correct scientific scores/split/provenance.

Fixed tests: prewrite rejection of six dense768x768FP64 matrices under20MiB;
complete-byte boundary and status replacement scratch; unsafe/colliding paths;
unknown/oversized/immutable writes leave original bytes unchanged; all required
config/environment/source/log/report/status/manifest fields; unknown/missing
files; same-size digest corruption and symlink detection; dtype/shape/object/
nonfinite/JSON rejection; C/Fortran NPY exact-size roundtrip; rank permutations/
metric bounds/int16 range; Cartesian identity/duplicate/missing/reference/extra
field/type checks and noncontiguous class IDs; full768D/9192x100 synthetic shape
serialization plus independent saved-array/covariance reconstruction. No measured
score, original feature access or speed claim. Success all tests and saved full
shape manifest verifier; failure technical/software, no scientific negative.

CPU120s/GPU0; durable run records<2MiB, test log<=1MiB. Full-size synthetic arrays
only in a fresh named /dev/shm fixture directory, aggregate<=32MiB retained, no
deletions. Full shape writer ceiling20MiB includes5MiB other file bounds plus
manifest and actual NPY headers; old H00620MiB incident unchanged. CPU workspace
planned128MiB, peak uninstrumented. CLI smoke and exclusive registration before
run. Existing GPU allocation preserved but unnecessary. Root bulk reserve fails,
lightweight records exception; /dev/shm current32.87GBfree exceeds policy reserve.
