# H003 — Annotation intake and review specification, v1

Status: **prospective feasibility specification; no annotations received or produced**.
Decision [D020](../decisions/D020.md). This is not a registered downstream pilot,
human-study approval, or a request to collect or upload restricted video.

Purpose: determine whether natural hand/face tracking errors can be identified
independently of the detector on the existing ASL data. Returned handedness scores,
API absence, source reuse and filled crops must remain separate from correctness
labels. [CONFSEM001](../inventory/CONFSEM001_confidence_semantics.json) establishes
the exported score semantics; it does not establish tracking accuracy.

## Accepting an existing annotation source

Record the source owner, release/version, annotation method, license/access terms,
sign language, subject identity convention, coordinate system and whether labels
are manual, adjudicated, fitted, tracked or synthetic. Missing provenance is
`unknown`, not manual ground truth. English gloss overlap is not a video join.

Each label must resolve to an exact source video SHA256, native decoded frame
index (zero based), native presentation timestamp in seconds, image dimensions,
and documented annotation-to-image transform. A crop offset, resize or re-encoding
requires explicit verification. Reject guessed frame offsets and name-only joins.
Record annotation-file SHA256 and preserve original bytes alongside any conversion.
Do not overwrite source videos or annotations. An unavailable licensed asset is
a dependency; it does not authorize an external registration form or submission.

## Reviewing the four existing diagnostic clips

The fixed LANDMARK001 cohort contains 156, 87, 82 and 71 frames (396 total), with
source identities in [its manifest](../inventory/LANDMARK001_landmark_audit.json).
Review all frames if this cohort is used; do not select only the 14 duplicate
assignments or frames with low scores. These clips were inspected during software
debugging and cannot become an untouched confirmation set. They can establish
annotation feasibility and diagnostic examples only.

First label the original native RGB frames without detector scores, predicted
landmarks or algorithm names. The target is the primary signer's anatomical left
hand, right hand and face; image left/right is a separate convention. Annotate
other people separately or mark target identity unresolved. An annotator may view
neighboring original frames to identify a hand; record this offline context access.
Do not infer hidden coordinates by drawing a plausible hand.

For each target and frame record:

- Visibility: `visible`, `partly_visible`, `occluded`, `out_of_frame`, or `uncertain`.
  Occluded and out of frame are distinct; detector absence does not decide either.
- Target identity: anatomical left/right or face, plus whether identity is certain.
- When observable, a tight bounding box around the visible target region in native
  pixel coordinates, and optionally a visible palm center/nose tip. State the
  landmark definition. Unknown coordinates remain null, never zero or sentinel.
- Annotator ID, annotation version, uncertainty and reason. Preserve both reviews
  and adjudication if two reviewers disagree; do not silently average labels.

Then compare the frozen detector output and crop against the manual region.
Record wrong-person/wrong-hand assignment, shared source detection, missed visible
target and predicted region without a visible target as separate outcomes.
Compute continuous location distance and visible-region containment only when
their manual quantities exist. A visible-region box is a geometric proxy;
it does not certify handshape, intelligibility or non-manual grammar.

Example field names (schema only, not fabricated annotation):
`source_sha256, frame_index, pts_seconds, width, height, target,
visibility, identity_certain, visible_bbox_xyxy, landmark_xy,
annotator_id, annotation_version, uncertainty_reason, context_access`.
Boxes use native image boundary coordinates: x in [0,width], y in [0,height],
with x0<x1 and y0<y1. A visible landmark uses pixel-center coordinates in the image.
Specify fractional coordinates and rounding in any conversion. Keep prediction
fields in a separate table linked by source/frame/target.

## Evidence and measurement gates

Verify frame/target uniqueness, complete expected coverage and explicit unknowns
before scoring. Report denominators separately for visible, ambiguous and absent
targets. Reviewers do not use a model to adjudicate unknown manual labels. Two
qualified reviewers and preserved adjudication are preferred; a single review
must be disclosed and is not independent consensus. Anatomical localization
review alone cannot establish linguistic usefulness; that interpretation requires
appropriate sign-language expertise.

Do not fit reliability thresholds or report calibration on these same four clips
as a prospective generalization result. A later pilot must register a larger
train/dev cohort, separate calibration and evaluation groups, labeling coverage,
practical effect, uncertainty unit, evaluator fixtures and compute ceiling before
outcomes are measured. Clip frames are not independent population observations.

Required later controls: frozen native SHuBERT; fixed zero/previous fill and offline
interpolation with equal future access; mask-only adaptation; matched-capacity
dropout/imputation; and simple temporal/injective assignment if the proposed
mechanism addresses assignment. Confidence filtering is a control only with an
explicitly defined confidence quantity. SignBERT+ already uses confidence filtering
and weighted reconstruction ([S17](../literature/sources/S17.md)); previous-state
fill/ID handling has prior art ([S24](../literature/sources/S24.md)).

Ready-to-register outcome requires valid joined labels and a specific downstream
task plus a falsifiable reliability mechanism. No available labels means H003
stays held. A detector-availability or synthetic corruption study would require
its own explicit scope; neither substitutes for natural tracking-failure truth.
