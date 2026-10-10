# S2 post-implementation review

Seven independent geometry/cache tests PASS (`evidence/S2_pytest.txt`). Actual pinned
teacher weights load and real-video extraction/FK/normalization/target generation PASS
(`evidence/S2_pose_pipeline.json`). All 36 frames retained; appended four blank frames
have zero confidence and contribute no target loss. Body/right FK coordinates agree
with the public networks to <2e-7; every full-skeleton edge retains its corpus median
length under changed angles. Hand bone statistics exclude failed observations.

Review found and fixed global-hand masking: body-global pose and interaction distances
require the corresponding body wrist, while valid wrist-relative hand geometry remains
usable. Raw caches and dropped-frame caches are rejected before RGB supervision.
Weights are hash checked before original trusted checkpoint deserialization.

Limits: this bone table is from a technical clip, not the YT-SL training corpus. Body
angles can be entirely masked by the conservative all-body-points confidence rule
(observed on this clip); U05 remains unresolved. MediaPipe does not provide hand
per-joint confidence; observation indicators are recorded as such. Original image-hand
fallback/filtering/clamping/interpolation are present in pinned vendor source but
not applied to supervision, because missing-source confidence must remain masked.
Body camera orientation is retained for FK consistency (unverified historical choice).
These choices prevent a claim of exact author-target equality.

Decision: functional module checks PASS; historical target equivalence UNVERIFIED.
