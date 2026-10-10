# S1 pre-implementation review

Public wrapper's `init_weights=True` calls an undefined `_init_weights` inside a broad
exception, potentially hiding a broken head. Reproduce checkpoint names and layers
with an explicit constructor, without this fallback. Preserve the source modules
unchanged in vendor; namespace imports in an adapted package to avoid collisions
with the pose repo's top-level `config`, `utils`, and `models`.

Expose final encoder tokens for the Gram style representation. Reuse upstream
`forward_encoder` path (mask including a zero-ratio all-keep mask) so pooling and
released `features` are unchanged. Validate direct upstream output parity with a
strict load, not only matching parameter shapes. Test 64-frame inflation with an
independent kernel-index oracle, and verify token count for the expanded input.

Decision: architecture implementation may proceed. CPU validation first; no large
pretraining or benchmark run has been requested as an execution target yet.
