# MSFEATTEST001 — collector and inherited pooling manual oracles

Diagnostic CPU60s/GPU0/2MiB, before any actual cost inference. Five groups in D069:
manual constant RGB byte/crop/hash/estimated ordinals with raw backward timestamp;
21nativeframes/3complete16stride2windows/one uncovered tail; changed RGB/PTS/crop/
count/short guards; manually calculated clipped-max raw activity weighted FP64
mean/L2 (weights.25/1 -> [.2,.8] -> [1,4]/sqrt17); zero-weight fallback and reject
nonfinite or zero vectors. Original activity_mean/pooled_vector reused unchanged.
No actualtarget/model/GPU/quality. Code/config/output bound before launch.
