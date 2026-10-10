# S2 body-root convention — pre-implementation review

Recheck against the original teacher's actual public demo: hand root orientation is
replaced with cam_rot; **body root remains the native angle output**. Initial adapter
instead retained body cam_rot so FK matched camera-space pose_3d. That was an explicitly
unverified U01 convention, not a faithful copy of the demo's exported angles.

Body root Euler indices are [16,17,18], not the first three body angles. Released
angle-head sine-row weight norms for these indices are .0060/.0063/.0065 versus
~.453 for other sine rows, consistent with near-zero native body-root targets; this
is empirical support, not a complete original-preprocessing proof.

Decision before implementation: default to original native/canonical body angles as
in the public demo. Keep camera orientation as an explicit alternative, record selected
mode in every cache, and reject normalization with a different mode. FK oracle must
compare canonical body output to original pose_3d_can and hand output to original
camera-aligned pose_3d. Re-lift existing technical caches with unchanged pinned
networks/2-D detections; rebuild statistics/standardized caches and repeat training.
