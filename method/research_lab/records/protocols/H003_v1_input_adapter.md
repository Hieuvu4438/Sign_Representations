# H003 v1 — Adapted aligned input adapter, diagnostic only

Registered before INPUTPARITY001, after six INPUTTEST001 alignment/state oracles.
No H003 scientific pilot, labels/task metrics or HPO authorized. Same four fixed
train/dev clips and immutable LANDMARK001 observations, no new detector/test access.

Policy: native pose0/face-nearest/native independent hand-nearest0.1 threshold;
native eye/mouth gray face helper, hand bbox scale1.5/clipping/resizeINTER_AREA224;
keep native duplicate assignment and record shared detection explicitly. RGB native
frame; OpenCV writer receives BGR permutation/mp4v at native reader averageFPS.
No direct-crop replacement of encoded native inputs. Record source-native PTS and
frame index separately, verify every encoded stream has the same full frame count.

Missingness: explicit raw API fields; crop observation requires pose anchor plus
selected detector/crop. After first observed crop/body, repeat prior value with
observedFalse/fill_kind/source_frame_index; initially black224RGB/body14D−9999.
Whole-rowNone retains one frame, unlike native writer skip; body sentinel flattened
14D consistently. These are disclosed adaptations, not exact parity on unsafe
native branches. Collapsed/nonfinite body, malformed face/empty-hand list and
observed crop errors stop before state commit; no exception-to-zero substitution.
No anatomical/true-failure claims from detector categories or coordinate proximity.

State/mask fixtures: retain firstmissing frame/four arrays/14D sentinel; preserve
last observed pixels and source index while marking absent; preserve and flag native
duplicate hand assignment; reject framegap/bodycollapse/malformed face without
advancing state or silently filling. Six fixtures already pass on CPU.

INPUTPARITY001: run native crop/body functions only with fresh own paths/explicit
parameters (no upstream mains/default outputs), and own adapter on all396 native
frames. Compare decoded encoded crops exact pixel-wise and native body arrays
exactly, plus every current duplicate flag to independently retained CROP002 geometry.
Current four clips have no missing face/body/invalid crop, so native unsafe branches
are not exercised by this real-data parity; covered only by diagnosed fixtures.

CPU deadline180s, RAMstore ≤256MiB/reserve20%+2GiBworkspace, no GPU/model/detector
inference or downloads. Root lightweight manifest/status only; payload stays in
bounded /dev/shm with hashes. All source/config/native helper hashes bound before
execution. Pixel/body mismatch, source/timeline/coverage failure stops; no tolerance
widening/exclusion/truncation/codec swap. Native code/data remain read-only.

Success permits preparing a separately registered2–5clip real four-stream GPU
software smoke using fixed published DINO/SHuBERT interfaces, with compute gate.
It does not authorize scientific retrieval/recognition or natural-failure pilot.
