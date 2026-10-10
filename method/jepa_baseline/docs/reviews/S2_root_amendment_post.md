# S2 body-root convention — post-implementation review

Default extraction now preserves the native/canonical body angles of the original
teacher demo. Hand root angles still use cam_rot as in that demo. The alternate
camera body root is explicit and recorded; normalization rejects a root-mode mismatch.
Production corpus preparation accepts canonical body-root caches only.

Both real technical-video caches were re-lifted from their unchanged 2-D detections
and pinned teacher weights. Body root indices [16,17,18] are exactly zero. Bone
statistics and standardized targets were rebuilt. In the refreshed real-video oracle
(`evidence/S2_pose_pipeline.json`), canonical body FK equals the original teacher's
pose_3d_can exactly; right-hand FK differs by at most 8.94e-8 from its camera-aligned
pose_3d. All 36 decoded frames, including four blank frames, remain aligned. An
entirely missing clip has no supervision; observed body angles retain their masks.

All 28 mathematical, model-parity and provenance tests pass in
`evidence/all_pytest.txt`. Affected trainer/replay and production-CLI evidence is
refreshed separately in S6; the final audit binds the current artifacts.

This review supersedes the initial S2 camera-body-root convention. Public teacher
demo fidelity is established; SignRep's unpublished historical weight/preprocessing
identity remains U01, not a claim of exact training-run reproduction.

Decision: canonical-root implementation and FK/missingness verification PASS.
