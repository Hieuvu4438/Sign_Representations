# MSNATIVESCAN001 — prospective streaming native eligibility and interval reference

Diagnostic prerequisite for H006 MS-ASL100 adapted dictionary retrieval. No
encoder/task-quality/official score/newmethod access. Previous D053 census fixes
all3425candidate IDs before native pixels/gallery selection. New code is separate
from immutable msasl_native.py and retains the same source-origin, half-open
annotation-second, yxyx-on-actual-dimensions, RGB/OpenCV linear224 crop semantics.
One CPU decoder thread; no FPS resampling/input padding or frame substitution.

Streaming implementation validates every decoded native frame's PTS/timebase/
corruption, stable actual crop geometry, then hashes every selected resized RGB
frame without retaining a full video array. Select max1024frames per<=8s interval,
max4096decoded from keyframe, max3840×2160, max512MiB read-only compressed parent.
These are prospective resource gates, not scientific exclusions. Selected tick
sequence is hashed as raw little-endian int64; source SHA256/bytes/mtime bound once
per unique parent, size/mtime rechecked around each scan. No copying/downloading.

Independent ffprobe interval reference reports integer PTS, stream timebase/
start_pts. Seek request uses absolute origin+annotation start and absolute
origin+annotation end+1s flush margin; returned keyframe-prefix/extra frames are
filtered using exact rational half-open bounds. Every selected tick must match
PyAV, integer-exact; no post-outcome tolerance widening. Command and response-byte
count retained. Documentation: https://ffmpeg.org/ffprobe.html#Main-options
`read_intervals` explicitly warns seek may differ from requested start; use an
absolute end, not a duration from actual seek start. Web read2026-10-04 section
read_intervals, not full-document/version parity. Actual installed ffprobe and
PyAV versions/libraries recorded at runtime. Reference per-child hard timeout30s,
1MiB stdout/stderr size gate after PIPE capture (not OS memory quota). Original
four-source full-source ffprobe/OpenCV pixel checks remain authoritative historical
parity; this new reference is interval-only timestamp parity, not independently
decoded full-cohort RGB parity or historical official-cut reproduction.

Stages: software tests first (full lossless decode/manual boundaries/crop hashing/
short/no-padding/resource stops); then metadata-fixed cost/reference subset,
including all four prior technical sources and extrema of duration/source size/
offset, median duration per split. No outcome-based replacement. Every old source's
native count/first-last PTS/first16RGB hash must match immutable MSNATIVE002.
Only after those stages pass, register whole3425audit deadline/complete byte plan
using actual representative cost. A timeout/parity/resource/decoder issue retains
perrow status and blocks promotion; do not silently treat it as native short.

Native-short (<16 frames) is a preregistered availability result, explicitly
retained perrow; no zero feature, padding or invisible denominator reduction.
Full audit must account for all3425IDs. Eligible candidates may define a fresh
adapted protocol before task quality only if all100class IDs retain train and val
support; retain original counts and missingness. Any incomplete row or missing
class stops scientific cohort promotion pending a prospective documented fix.
No prior P004 shorttest rescue, no official benchmark substitution or class
renumbering. Geometry/source-pool/pretraining limitations unchanged.

Primary engineering gates: software oracles pass, old four-source parity pass,
all selected rows accounted, every independent native tick exact, no incomplete
rows. Cost subset cannot prove fullclass/native coverage or task utility. Artifacts:
full source identities, every row's status/reason/count/timestamp/hash/reference
command/captured wall, summary, code/config/env/status/log/error/full manifests.
No HPO/seed sweep. Software cap2MiB/time120s; cost cap2MiB/time1800s (CPU only).
Root reserve exception limited to bounded lightweight records; RAM256MiB+
2GiBworkspace reserve mandatory. Python run deadlines not OS hard native-call
watchdogs; no indefinite continuation without live-process monitoring.
