# MSPTSDIAG001 — native timestamp failures, exact location and independent stream

Diagnostic only, before any fix/cohort/gallery/modelquality. ALL38 incomplete
rows from terminal MSNATIVECOVER001 have reason native_pts_not_increasing and
are trainonly; preserve frozen fullscan/3425denominator/3short/3384eligible.
Collect complete bounded native seek-prefix ticks/timebase/origin until source
end or first presentation time>=annotation end, matching scanner traversal.
Record every nonincreasing pair and whether preceding/current frames lie inside
the halfopen annotation interval or outside. Never sort/deduplicate/interpolate/
drop frames or change timestamp tolerance. Independently run same ffprobe absolute
interval+1s flush reference, retain raw integerpkt_pts and best_effort_timestamp,
filter selectedticks mathematically; report count/order/header equality and exact
native selectedtick agreement, not silently fixing mismatches.

Primary information: issue occurs within selectedclip vs keyframepreroll; duplicate
vs backwardsjump; actualheader/timebase/streamorigin validity and originalsourceSHA.
No success criterion that promotes quality. Anyfix requires fresh prospective
adapter/softwarefixtures and failedrow revalidation, preserving all old artifacts;
no need to reencode/doubleextract all3384passing clips blindly. Native-short rows
already have independent clockcounts13/15/10, no padding/rescue in thisdiagnostic.

Bind terminal fullconfig/status/sourceparents/records/summary/code and38exactIDs/
officialtrainmetadata/sixunique parent SHAs. CPU300s/GPU0; max4096native/ref ticks
perrow,512MiBparent/1MiBreference stdout/stderr/30schild timeout; stream single
decodedframe, no RGB resize/model/feature calls. Recordcap12MiB/whole20MiB plus
complete snapshot/context/status/log/summary/manifest bytebound. One decoderthread;
256MiB+2GiBreserve checked. Reference PIPEpostcapture bounds not OSquota; Python
deadline not guaranteed native-hardwatchdog. Actual argv/env/status/errors retained.
No novelty/truecrop/human/sourceauthentication/encoder-unseen claim. Goalactive.
