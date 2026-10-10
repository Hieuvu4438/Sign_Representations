# S2 mask amendment: pre/post review

Pre-review after the real training smoke: conservative all-body-keypoint visibility
masked every body angle, despite the teacher emitting a valid pose each signer frame.
Main §5.1 specifies >50% **keypoint** confidence; it does not specify an angle confidence
score or requiring every unrelated body point. This choice would discard a major prior.

Decision before implementation: expose both policies, default to explicit teacher
part-detection flags plus finite angle values. Preserve strict point visibility for
keypoints/distances. Never infer an observed angle solely from finite zero-filled cache
values. Require the frame-indexed part flags, already retained by extraction.

Post-review: independent test covers low hip confidence with observed body angles,
missing right teacher observations, and rejection of omitted angle flags. All-missing
clip remains fully masked. Rerun S2 extraction/FK targets and S4 replay after this change.
Historical angle masking remains U05; this policy is source-compatible, not author-code
verified. Earlier S2_post conservative-mask finding records why the default was amended.
