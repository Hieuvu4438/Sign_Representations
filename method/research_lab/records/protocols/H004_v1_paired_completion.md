# H004_v1 — Shared-query pairwise completion diagnostic

Prelaunch [D022](../decisions/D022.md). Inherit H004_v0 cohort, fixed gallery,
native activity/fallback, all1149 pairs, source-hash permutation1701, CPU180s/GPU0,
no test or query-label access in stopping. New variable: direct pairwise gallery
direction bound instead of independent per-instance envelopes. No other change.

For prefix score sums S, anchor a=argmax_i S_i and its class c, define direct
ideal-real bounds l_aj on one unseen feature dot(g_a-g_j). Unknown clamped weight
gives lower min(0,l_aj). Certify only if min over all j outside class c of
S_a-S_j+(N-n)*min(0,l_aj)>0 and observed total weight>1e-12. Otherwise continue;
all-window fallback uses literal native pooled-vector/search reference.
Choosing an observed anchor is allowed; it never uses the dense winner or labels.

Primary gate unchanged: half-budget stopping on>=50% pairs, zero envelope
violations/reference disagreements. Report every pair and seed, used-window ratio,
and paired-v0 stop-count changes as descriptive diagnostics. No selection of a
new winner/policy from P003 or P004 test outcomes. No measured latency/quality.

Required independent fixture: two gallery directions share a large common feature
coefficient but differ on a small second coordinate; derive the exact LayerNorm
pair bound by hand and enumerate remaining completions. Additional zero-weight,
tie and winner-reversal cases retained. Empirical CPU/cache agreement is not a
formal floating-point guarantee; production certification remains a separate gate.
