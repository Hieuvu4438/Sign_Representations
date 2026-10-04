# H006_v2 draft — Geometry orientation versus eigenvalue spectrum

Prospective analysis candidate after D030/D034/D037, not a promoted method or
launch authorization. No outcomes for these controls yet. Existing residual LW
cosine wins the fixed PCA recipe; this tests a separate explanation of that
known standard baseline, not a reopening of rank8 superiority or H002 search.

Use case remains cross-signer one-example isolated ASL dictionary lookup with a
frozen SignRep encoder and extra class-labeled auxiliary train examples. Problem
evidence C066: residual geometry DCG.829746 vs total geometry.762639 under same
fit/query/gallery access. Different covariance eigenvectors AND eigenvalues
(and automatic shrinkage coefficients) remain confounded in that comparison.
Thus the current result alone cannot attribute utility to learned directions.

Hypothesis: using the true-class residual eigenvector directions improves DCG
by at least.005 relative to total-covariance directions when both use exactly
the same residual eigenvalue spectrum, with no gallery-wise negative difference.
Failure kills this orientation-specific explanation/scaling, not all geometry
or the actual earlier standard-baseline result. No hyperparameter/seed search.

Let C_R=V_R diag(lambda_R) V_R^T and C_T=V_T diag(lambda_T) V_T^T, eigenvalues
in ascending order. Compare a fixed2x2 factorial of direction basis R/T and
spectrum R/T. Original RR/TT are parent standard residual/total LW cosine;
TR uses total directions/residual spectrum; RT uses residual directions/total
spectrum. Whitening W_ab=V_a diag(lambda_b^-1/2) V_a^T, then center by the same
base mean and L2-normalize. This changes weighting axes in original feature
space; merely rotating already-whitened vectors would preserve cosine and is
not a discriminating control. Overall scalar covariance differences disappear
under L2; no target covariance, class means or new labels used in the hybrids.

Primary fixed RR-minus-TR meanDCG>=.005 plus all3gallery deltas>=0. Secondary
RT-minus-TT and spectrum effects RR-minus-RT/TR-minus-TT are descriptive, report
all factorial cells and signed deltas, no best-cell selection or new primary.
All383prior dev queries/100local target classes/three fixed train galleries,
1313base train rows/native caches/full ranking/scorer unchanged. Parent RR/TT
complete ranks must match exactly; direct metric reconstruction must verify
new full ranks. Mathematical fixtures must distinguish identity, diagonal,
rotated anisotropic, and repeated-eigenvalue cases; positive-eigenvalue,
orthogonality/reconstruction and eigenspace-identifiability policy must be
locked in the implementation protocol before any query outcome is computed.
Eigenvectors at repeated/near-degenerate eigenvalues are not unique; interpretation
and a registered failure/hold rule are required, no arbitrary post-outcome rescue.

No test/new encoder/GPU, one fixed diagnostic/HPO0. Same-corpus reused dev,
extra base labels/five participants, unknown pretraining overlap and ontology
scope stay explicit. Paired gallery-averaged conditional participant bootstrap
2000/seed1701, not population/causal or independent confirmation. Proposed
CPU120s, full20MiB logical output ceiling including config/source/env/log/status/
manifest/rewrite scratch; actual array/header/file bounds not yet locked. New
wrapper must use verified bounded CPU context, preserve original runs, and
preflight the entire output plan. No launcher/config/run registered yet.

Closest reviewed mechanisms S32/NAP, S34/SimpleShot, S35/SimpleCNAPS,
S37/FROST and S38/standard LW already constrain method novelty. Factorial metric
controls are ordinary numerical analysis, no claim of a novel whitening method
or unique prior-art absence. A single additional same-dev diagnostic is still
insufficient for paper-level analysis: an informative finding, refreshed closest
analysis review and valid independent validation remain required. Scorecard
problem3/5, noveltyunknown(0credited), feasibility4/5, compute5/5,
mechanism-discrimination4/5 prospective. Full research objective unchanged.

Next actual work: implement the fixed factorization and independent manual
oracles, specify degeneracy handling and exact byte caps, then bind code/protocol/
parent/cache/scorer hashes in a fresh prelaunch registration before measurement.
