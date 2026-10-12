# Data card — current diagnostic scope

Phase3 adds a locked8-class development pilot:32train/16val, official signer-disjoint
membership, three seeds, actual availability/source hashes and temporal fingerprint
candidate audit. Selection uses official class labels and availability, before model scores;
this is a label-curated pool, not label-free corpus selection. SSL sampler/loss does not read
those labels. No test CSV/videos were accessed. See `evidence/phase3_pilot_plan.json`.

The real integration gate uses two existing local videos from the official ASL Citizen
**train** CSV: APPLE and IMPOSSIBLE, with 71/76 decoded frames and 59/63 selected samples.
CSV membership, current video SHA-256 and cache source SHA-256 are checked before use.
The downstream class IDs come from the vocabulary fitted to that official train CSV.
The classifier in this gate is fitted and predicted only on the two train examples;
these predictions are not held-out accuracy or an official benchmark result.

SSL providers receive source/sample IDs, features, validity, timestamps and supports;
they do not receive gloss/text/label IDs or signer metadata. Geometry statistics use
only the declared training sources for each stage. No validation/test examples or labels
are used by the original integration gates; phase3uses validation labels explicitly for
probe/checkpoint selection. Full source/signer/hash/near-duplicate corpus audit remains
required before research claims; exact manifest audit APIs alone are insufficient.

MediaPipe outputs are estimated observations. Raw missing values remain NaN in the
landmark cache; model coordinates/quality become zero with validity false. Hand
category scores are not per-joint localization probabilities. IMAGE-mode processing
and conservative same-frame supports are recorded. Source mirror metadata remains
unknown, and quality/handedness review is limited to diagnostic clips.

The synthetic provider generates explicit `synthetic_correctness_only` features with
variable lengths, temporal patterns and missing streams. It is available only under
the fixture backend and never substitutes for absent scientific data/assets.

Dataset videos and identifying overlays stay in local/ignored artifacts. This method
folder does not redistribute the corpus or assert additional dataset/model rights.
Author terms/model notices remain applicable; the run provenance records actual
local asset hashes and the supplied CSV version.


Final preprocessing also stores immutable recipe-hashed coordinate sidecars for all
32 train and 16 validation clips: original pixel xy, frame W/H, shoulder center/scale,
normalization validity, original timestamps and source indices. These sidecars bind
the original raw landmark/source hashes and preserve the historical feature cache.
Final prediction JSONL keeps official signer IDs locally for grouped bootstrap,
source IDs, quality proxy buckets and model hashes. It is not a redistribution export.
