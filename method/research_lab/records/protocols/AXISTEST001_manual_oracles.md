# AXISTEST001 — Axis/spectrum independent manual oracles

Three fixed analytic oracle groups registered before execution, D038.
2D R covariance diag(1,4), T45degree rotation diag(4,9); expected W_RR diag(1,.5),
W_TR [[.75,.25],[.25,.75]], W_RT diag(.5,1/3), W_TT [[5/12,1/12],[1/12,5/12]].
Independent metric forTR [[.625,.375],[.375,.625]] gives matching normalized
inner products. Coordinate rotation after whitening/overall scalar scaling
preserves cosine, unlike swapped weighting axes. Identity/repeated and
near-degenerate spectra reject, along with nonpositive/asymmetric/nonfinite
covariance/zero vectors. Tolerances1e-14/1e-15 for analytic checks; actual eigensystem
floor/gap1e-12, ortho/reconstruction1e-10 are fixed. Tests are software only.
CPU120s/GPU0/HPO0/<2MiB complete bounded CPU context; exclusive outputs and
snapshotted bound imported own modules. No original file or dataset changes.
