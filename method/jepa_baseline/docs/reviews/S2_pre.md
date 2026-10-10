# S2 pre-implementation review

Use original teacher skeleton objects from its checkpoints, not recreated Euler
parameters. The full-body FK skeleton has three virtual nodes; emitted pose must
have 19 body/face + 42 hand points. Verify body=22 and single-hand=41 Euler parameters
against loaded weights before constructing SignRep targets.

The public demo drops body failures and strips point confidence. A new exporter must
keep every decoded frame and retain point masks/confidence before lifting. Body
visibility and hand detection indicators are different semantics; never relabel a
handedness score as joint confidence. Preserve this distinction in cache metadata.

Corpus-standardization operates on **full FK offset vectors**, with median bone
lengths and directions retained. The demo's reduced offset parameters/proportions
must not substitute for independent per-bone corpus medians. Exclude missing hand
observations from its statistics. Reuse original FK with expanded offsets for the
standardized output and verify lengths independent of rotation and signer scale.

Angle masks conservatively require valid observed points for each part (U05), pending
an author mask specification. Distance masks require both endpoints. Wrist-relative
keypoint masks require the wrist and the joint. A missing hand gives unknown activity,
not a confidently inactive label. Compute activity per **selected training clip**,
not once per long video. Keep confidence threshold strictly >0.5.

Decision: proceed; public-teacher identity is U01, coordinate-distance convention U03.
