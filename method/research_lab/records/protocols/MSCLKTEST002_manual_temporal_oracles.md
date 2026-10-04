# MSCLKTEST002 — original temporal fixtures plus strided RGB byte oracle

Fresh version, five groups. Inherit exact manual clock/parser fixtures from
MSCLKTEST001; additionally manually enumerate12 RGB bytes for2x2HWC view whose
row stride includes unused pixels. Hash must match contiguous copy and independent
hashlib(bytes oracle); reject wrong shape/dtype. This tests the actual serializer
failure, not just an implementation mirror. Softwareonly/CPU60s/GPU0/2MiB.
No tolerance/clock/membership changes or target samples. D064 bounds retry1.
