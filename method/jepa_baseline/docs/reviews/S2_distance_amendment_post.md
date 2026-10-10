# S2 distance convention — post-implementation review

Default target construction and pretrain configuration now use signed coordinate
vectors source-minus-destination. A1 source/destination index sets, dimensions and
interleaved angle serialization are retained. Opposite signed and absolute coordinate
choices remain explicit alternatives.

Independent coordinate arithmetic checks the new default. A second independent
test checks the released linear heads themselves: the author-trained body-distance
head is more consistent with source-minus-destination combinations of the body-pose
head's weights and bias than the opposite direction. The data diagnostic in
`evidence/checkpoint_target_diagnostic.json` additionally compares actual predictions:
body MAE 0.003254 versus 0.213328 for the opposite sign, and 0.132531 for absolute xyz.
Both hand heads prefer the same signed direction; negative outputs rule out an
implicitly nonnegative metric. Angle outputs also support interleaved sin/cos.

The complete 28-test suite passes (`evidence/all_pytest.txt`). Trainer/replay and
end-to-end CLI evidence are refreshed in S6 after rebuilding the target caches.

This amendment supersedes the initial U03 default and its corresponding arithmetic
tests. Checkpoint consistency is strong independent evidence; unavailable original
target-building code still prevents a claim of direct source-level equivalence.

Decision: revised default and independent released-head oracle PASS.
