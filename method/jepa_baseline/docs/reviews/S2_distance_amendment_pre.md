# S2 distance convention amendment — pre-implementation review

Initial U03 default destination-minus-source was explicitly unverified. Inspecting
the **actual released checkpoint's simultaneous pose and distance heads** provides an
independent author-trained oracle. On a real16-frame clip, body_dist MAE against
source-minus-destination geometry reconstructed from body_pose is .003254; opposite
direction .213328; absolute_xyz .132531. Roughly half its predictions are negative.
Right/left distance heads also favor source-minus-destination.

Interleaved sin/cos norms fit the released angle heads much better than concatenating
all sine values followed by all cosine values (.01468 vs .74106 body mean norm error).
This supports the selected angle serialization and signed coordinate convention.

Decision before amendment: default to signed_xyz_source_minus_destination; retain the
opposite signed convention and absolute_xyz as explicit ablations. Keep the paper's
index sets, tensor shapes and flattening order. Update the independent arithmetic
oracle and add a released-checkpoint head-consistency test. Re-run affected targets,
training replay, downstream tests and CLI orchestration. Historical preprocessing
code remains absent; distinguish strong checkpoint evidence from direct code proof.

Evidence: evidence/checkpoint_target_diagnostic.json (clip/input/checkpoint hashes).
