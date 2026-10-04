# NEAR001 — bounded perceptual duplicate screen

Registered2026-10-04 local (2026-10-03 UTC), before pair scores examined.
H001/P001_v2 prerequisite; input frozen VID002 fingerprints, no model/test.
Compare all unique pairs within/between selected train/dev, independently
of labels. Five normalized-time native-frame64-bit dHashes; only compare
positions with thumbnail gray std≥5 in both clips. Candidate if≥3 positions,
mean Hamming≤6,max≤12,frame-count ratio≥.8 (min/max). Record every candidate
with IDs, split/participant/class metadata, distances and information counts.
At most20 cross-split candidates, ordered mean/max/IDs, get visual review
first if needed; larger sets require another explicit review plan, no silent
filtering. No threshold tuning after results, no automatic sample removal.

Budget30s CPU, no GPU/download/source video read; output≤1MiB. Exclusive
NEAR001_20261004_P001v2_s1701. Independent oracle: identical hashes distance0,
one-bit difference1,63-bit complement64; reject low-information fixtures.
Fingerprint matching is a screening heuristic, not proof of complete duplicate
absence, identity, session independence or label quality. Unresolved candidate
cross-split recordings stop pilot until reviewed; benign same-background
matches do not justify exclusions. SHA exact audit COHORT001 already separate.

Observed completed exit0,1.0183s CPU.1,428,895 total unique pairs examined;
496,314 cross-split,932,581 within-split. Threshold-eligible cross-split161,705;
none flagged.861 within-split candidates remain **screen positives**, not
established duplicates; no automatic exclusions. Reduced eligibility and
background sensitivity mean this is not exhaustive absence proof.
Sixteen unit checks passed, including0/1/64 bit-distance and low-information
rejection. Source hashes/screen/status preserved; no video/person identification.
