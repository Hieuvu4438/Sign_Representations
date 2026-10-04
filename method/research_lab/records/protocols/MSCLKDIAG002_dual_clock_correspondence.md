# MSCLKDIAG002 — unchanged all38 clock/RGB diagnostic with logical-byte hashing

Fresh retry1 of failed MSCLKDIAG001; follow entire bound D063 and
MSCLKDIAG001_dual_clock_correspondence contract: SAME38IDs/sixsources, raw/best
integer clocks/ordinal selections, absolute seek/preroll/vsync0/FFmpeg framehash,
exact count/hash/clock gates, no sorting/dedup/pad/resample/drop/quality/model calls.
Only hash serialization uses logical C-order HWC uint8 bytes, validated by fresh
MSCLKTEST002 manual noncontiguous byte fixture. Preserve original failure untouched.
Bind both prior protocol and D064, oldfailure status/config/error and old/successful
software reports. CPU900s/GPU0; perchild30s/1MiBpostcapture,4096frames/32MiBframe/
512MiBsource, RAM256MiB+2GiBreserve,20MiB whole store with16MiBrecords; one transient
<=32MiB byte buffer added, never retained. Gates/clock tolerances fixed before run.
Estimated clock is not ground truth or a new authorized input/cohort contract.
