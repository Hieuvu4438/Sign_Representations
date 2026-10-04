# H004_v0 — Ideal arithmetic completion envelope feasibility

Prelaunch [D021](../decisions/D021.md), [H004](../hypotheses/H004.md).
Comparability: diagnostic, exploratory train/dev reuse; no benchmark claim.

Fixed383 dev sources × three fixed one-example/class train dictionaries (1149
pairs). Encoder/checkpoint/cache/source identities inherited read-only from
REPQUAL001 and CACHE006; own script/config newly hashed before launch. Use only
`full_activity` dictionary vectors. Output every pair's window count, stopping
count, strict bound gap, dense/stopped class, agreement and fallback.

Source SHA plus seed1701 produces one permutation without replacement. At each
prefix, scalar lower/upper score sums use the architecture envelope in D021;
no unseen feature values, labels, dense winner or empirically fitted bound enters
the stopping decision. Dense cached values are a separate verifier. All-zero
activity, ties and no valid stop fall back to all windows. Count prefix windows
once per gallery experiment; they are simulations, not new encoder calls.

Primary gate: at least50% of all1149 pairs stop at n<=floor(N/2), with zero
reference disagreements and zero envelope violations. Also report pooled query
window ratio and every seed separately. No confidence interval/population claim;
few signer groups and prior dev access disclosed. Only Top1/reference fidelity,
not original label accuracy/DCG, evaluated. No test scoring or timing result.

Manual mathematical fixtures required before run: tight directional bound on a
two-dimensional LayerNorm/Linear example; unknown activity includes zero; unseen
adversarial completion prevents premature stop; true separated interval permits
stop; all-zero-weight and ties force dense fallback. Independent enumerated
completion checks validate the stopping predicate. Tests are not scientific pairs.

CPU180s/single thread, GPU0. Check immutable source/config hashes, saved cache
status/manifest/all shard hashes and finite arrays before measurement. Existing
RAM workspace reserve required. Completed output remains immutable; technical
failure gets a new registered run/config with failed attempt retained.

Formal claim is real-arithmetic only. FP32 LayerNorm and matrix-product rounding
are not proved enclosed; fixture/cache agreement cannot close that gate. Failing
the practical gate rejects this deterministic envelope for scaling, not every
adaptive retrieval approach. No bound relaxation or policy choice from outcomes.
