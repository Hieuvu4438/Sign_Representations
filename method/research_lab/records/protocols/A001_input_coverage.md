# A001 — Auxiliary native input coverage, after metadata and before quality feedback

Registered2026-10-04 while CONFIRM001 actual inference runs. No held-out quality
score has been computed/read. This is an additional descriptive analysis, **not**
the original P003 primary confirmatory plan. No changes to its methods, galleries,
source IDs, ranking, metric, margin, selection or slice/case thresholds.

Question: how much of the dense complete native-window input domain do uniform8
and its two same-count random8 policies expose to the encoder? This tests a
structural coverage explanation for their previously observed dev tradeoff;
it does not test linguistic information preservation or causal representation.

Use all1691 COHORT001 audited train/dev sources and all1217 PRETEST001 test sources,
fixed native PTS/16-frame-stride2 plan. Policies full_activity/uniform8/random8_s2718/
random8_s31415 unchanged, same source-SHA deterministic choices as inference.
No labels, embeddings, predictions, metrics or test scores used. For each selected
native window form half-open frame and PTS intervals; merge overlaps/touching
intervals, sum union, normalize by full dense-domain span. Report longest uncovered
leading/interior/trailing gap, coverage fraction, union components and exact calls.
Dense drops incomplete tail, so denominator is dense-covered source domain rather
than the complete raw video. Native PTS last boundary remains median-delta estimate.

Summary allsplit/methodN, mean frame/time coverage, complete-coverage count,
median/max longest gap and window counts. No inferential interval/significance,
selection/fitting, new random seeds or primary quality decision changes. Later
joining exposure to observed errors must be explicitly exploratory/observational;
correlation cannot establish causality because positions and activity also differ.

CPU-only, single process/BLAS thread, max120s, noGPU. Three manual union/temporal
oracles include incomplete-tail, overlapping windows, large holes, nonuniformPTS
and invalid boundaries/indices. Artifact sources/summaries retained; no videos or
features released. Same-agent role review: structural measurements can clarify a
failure explanation, cannot fill missing scientific novelty or human annotation.
