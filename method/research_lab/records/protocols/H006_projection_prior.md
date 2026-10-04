# H006 projection family — own covariance identity, prior art explicit

For one base gloss c with n_c rows x_i and mean m_c, define R_c rows x_i-m_c.
Expanding ordered pair differences gives

`(1/(2*n_c)) sum_{i,j in c} (x_i-x_j)(x_i-x_j)^T = R_c^T R_c`.

Summing over classes equals R^T R. Therefore leading right singular vectors of
the stacked residual matrix are leading eigenvectors of a weighted within-class
pair-variation criterion. Applying I-UU^T removes that subspace. This is an own
algebraic identity, checked independently by enumerated GEOMTEST002 fixture;
it does not establish numerical/linguistic invariance, novelty or causal nuisance.

S32§3.1 describes the NAP orthogonal-projection/eigenfamily for weighted pair
variation; choosing different group labels or using SignRep/cosine dictionaries
does not create a new projection algorithm. Its speaker-session/kernel/SVM
setting differs, so this is a known-family diagnostic, not a reproduction of
that full method or a priority assertion for sign retrieval. A useful empirical
transfer result would still need a defensible analysis contribution and stronger
independent controls/evaluation before conference/PAPER_READY claims.
