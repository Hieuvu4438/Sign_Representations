# Pilot perception QA

Reviewed all four contact sheets `contact_0.jpg` through `contact_3.jpg`: 20 distinct real clips, up to four traced frames per clip, red anatomical LH and green RH. This is an agent visual review, not expert annotated detector accuracy.

Observed: single seated signer; crops generally follow anatomical hands while hands approach face/chest, overlap or move. Missing detections under overlap/blur are visible rather than filled. Out-of-frame resting hand and partially clipped crop appear in contacts 1 and 3. Contact 3 final row includes an unusually large red crop around forearm while hands cross: landmark/crop errors can survive confidence thresholds. No quantitative hand-swap accuracy claim is supported. The synthetic crossing association test independently validates slot assignment given correct landmarks, not MediaPipe tracking quality.

Decision: usable for a bounded diagnostic pilot with detection error/coverage disclosed, not a proof of phonological fidelity. Teacher suitability variance is nonconstant (artifacts/teacher_suitability.json). Freeze this cache policy for all controls. A future perception revision requires new cache fingerprints and rerun controls.

Preview frame indices are the first four eligible canonical indices divisible by 15, recorded in each clip's frame diagnostic JSON; actual times and source-frame mapping are in the corresponding audited NPZ.

## Evaluation-domain QA

Reviewed eval_contact.jpg, five ASL Citizen clips. Some previews contain resting/out-of-frame hands; visible signing under close bimanual overlap also loses detections. No independent frame-level annotation is available, so the measured hand-valid fractions (.219/.259) are observation coverage, not detector recall. Pose exists on8182/8188 valid grid frames, with475 dropped category-slot conflicts; confidence, wrist association, clipping and missing hands also affect eligibility. Stored numeric body coverage averages.533. All120 gallery/dev videos have at least one valid hand feature and all80 queries/40 glosses enter A0 retrieval. Preserve these eligibility rules for every trained control; disclose the domain difference.
