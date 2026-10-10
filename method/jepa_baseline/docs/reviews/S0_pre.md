# S0 pre-implementation review

Scope reviewed against final main-paper text and supplementary A1–A8, upstream
SignRep tree and checkpoint, and the pose author's public source.

Findings: reuse the released sign decoder; no pixel decoder is required for SignRep.
Do not reuse project's existing 2-D adaptations as 3-D teacher priors. The pose demo
drops undetected-body frames, discards per-keypoint confidence, and uses per-video
offset statistics. An aligned adapter must address all three. Body virtual nodes
must be retained internally for FK then excluded from 61-point exported pose.

Decision: proceed with source pins and a staged contract. Unspecified conventions
must be exposed and recorded; full training/metric reproduction is not yet proven.
