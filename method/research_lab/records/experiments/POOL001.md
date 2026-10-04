# POOL001 — pooling/probe contract fixtures

Registered2026-10-04 local before software smoke, H001/P001_v3 draft.
Synthetic arrays only; zero model/data performance, GPU or HPO. Implement:
mean, bounded-regression activity weights with logged mean fallback;
four bins defined by relative window-center range, mean-filled empty bins;
fixed Gaussian3072→768 projection, seed1337; per-video shuffled bins using
source SHA256+seed2718, independent of labels/splits/support subsets.
Control repeated clip-mean through the same projection distinguishes
projection/capacity from temporal aggregation. All heads see768 dimensions.

Choose feature key `features` before performance feedback. No corpus fitting
of projections. Normalize clip vectors per sample before similarities/probes;
zero/near-zero norms rejected, not filled. Candidate deterministic ridge probe
with intercept: training-centered X/one-hotY, alpha>0, direct regularized solve;
prototype cosine comparator with all classes covered. Full pilot/head search
registration separate; no prototype/gallery metric substitution.

Independent fixtures: hand-computed weighted mean/zero fallback; irregular
timestamps bin IDs; projection control algebra; source-keyed shuffle reproducibility;
one-dimensional class-one ridge slope1/4/intercept1/2 for alpha2;
missing classes/nonfinite data/zero norms rejected. ≤30s CPU, one BLAS thread,
no installs; logs/config/code hashes under exclusive POOL001 runtime directory.
Passing means software contracts only; no temporal benefit/novelty claim.

Observed completed exit0,.04453s CPU;20 unit checks pass, including pooling
and ridge independent fixtures. Report/config/code snapshots/tests/status
retained. No model/dataset feature fitting or scientific performance measured.
