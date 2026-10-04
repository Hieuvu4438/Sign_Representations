# MSCLKDIAG001 — all38 raw/estimated clocks and RGB correspondence

Pre-quality diagnostic. Fixed ALL38 train failures from terminal MSPTSDIAG001,
in original order with source SHA/annotation index/class; no new subset or gallery.
Bind original diagnosis/fullscan via its bound parent artifacts, official train
metadata, six source parents, executable hashes, code snapshots and completed
MSCLKTEST001. Registration exclusive; no overwrite completed runs.

For each row: repeat old absolute ffprobe interval+1s flush, retain every raw
pkt_pts and estimated best_effort integer tick, available DTS/packet-position/
picture type and header. Require old raw reference ticks identical. Re-decode
the SAME number/order of frame ordinals through PyAV with original keyframe seek,
save raw integer ticks and RGB24 SHA256 per frame, no video/RGB arrays stored.
ffmpeg input seek_timestamp1/noaccurate_seek/copyts, vsync0, frames:v exact count,
explicit rawvideo RGB24/stream enc_time_base/framehash SHA256; retain actual text,
stderr, argv, returncode, parsed packet timestamps/timebase/sizes/hash. Compare
exact hashes/order/count and best clock rational times. Negative joins retained,
never sort/shift/drop duplicate frames to make them agree. All38 investigated
before any conclusion. A subprocess/parse failure is technical, not method failure.

Primary observations: perrow whole-prefix and selected-interval monotonicity,
changed raw-to-best ticks and maximum difference, explicit added/removed decoded
ordinals and native16 stride2 window counts, independent RGB/order/clock gates.
Best-effort is decoder-estimated, NOT true annotation/playback ground truth.
All-pass correspondence licenses considering a future adapted protocol only,
does not rewrite original MSNATIVECOVER001 failure or authorize task extraction.
Quality/gallery inaccessible here; no model calls, metric fitting or target fit.

Resource limits D063: actualCPU900s/GPU0, decoder/filter singlethread; eachchild
timeout30s and stdout/stderr1MiB postcapture ceiling; <=4096frames/32MiB perframe,
512MiB per parent. RAM256MiB plus2GiB reserve. No persistent RGB/source copies,
16MiB recordcap and complete20MiB store incl context/snapshots/logs/status/manifest.
Python/native deadline and postcapture bytes are application checks, not hard OS
quotas. Errors retained under registered exclusive output. No scientific utility,
source authentication or identity certification from these diagnostic counts.
