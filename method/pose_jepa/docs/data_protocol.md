# Data protocol

Stage 1 returns pose streams and masks only. Manifest labels/text are excluded from
the SSL dataset return value and losses. Stage 2/ISLR/CSLR use verified train labels
with `(language, namespace, gloss)` keys; SLT reads paired train translations.
Training vocabulary never includes validation/test labels.

Use `in_domain_ssl` when target train clips enter unlabeled pretraining. Use
`cross_corpus_transfer` only when the target corpus is absent from SSL/metric
training. The gallery may then contain target train clips for frozen retrieval;
no gallery classifier/projection/prototypes are fitted. Record whether model
selection uses source or target validation. Test is kept out of selection.

`validate_data` checks schema, checksums, source identities, exact pose duplicates,
canonical pose near-duplicate candidates and coverage. `build_protocol` preserves
evaluation rows and removes training candidates identified by protected source,
pose/video hash, URL and reviewed near-duplicate exclusions. Near-duplicate
thresholds are review aids, not proof that every semantically similar sign is a
duplicate. Official signer/split policies are retained; unknown signers stay null.

Confidence is bounded detector reliability, never depth. Default `raw_probability`
rejects scores outside [0,1]. Existing local ASL-Citizen caches have unbounded
positive SimCC maxima. The explicit `simcc_positive_rational_v1` legacy profile
maps `q=s/(1+s)` and preserves all-person raw scores with a checksummed sidecar.
This is uncalibrated input adaptation, not the main detector's calibrated
confidence or an exact RTMPose/RTMW reproduction. Training rejects mixed score
profiles. Thresholds stay fixed; evaluation does not silently discard hard clips.

Reference origin/scale come from the full sample before windows. Unknown timing
uses original source frame indices for position and disables motion supervision.
The real sanity subset uses decoded source PTS, checked against the native cache
frame count. Motion excludes invalid endpoints, discontinuity and gaps >0.2 s.

Primary retrieval is `gloss_max_single_relevant`: max video cosine per gloss,
deterministic gloss-key ties, 1-based ranks and R@1/5/10, MRR, single-relevant DCG.
Invalid/zero queries and absent gallery glosses count as failures in the fixed
query denominator. Secondary video ranking separately reports multi-positive
mAP/nDCG. Sequence caches retain calendar segments/internal missingness and carry
checkpoint, manifest, preprocessing and stitching provenance.

Synthetic fixtures prove code paths only. The prepared 4-class ASL-Citizen subset
is a small official-split sanity experiment, not WLASL2000, full ASL-Citizen,
NMFs-CSL or a sign-language quality benchmark. It has no translations or full
continuous gloss transcripts; CSLR on isolated glosses is adapter sanity only.
