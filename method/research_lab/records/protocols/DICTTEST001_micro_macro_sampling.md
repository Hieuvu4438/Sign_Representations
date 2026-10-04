# DICTTEST001 — prospective adapted dictionary metrics/sampling fixture gate

H006_v3_MSASL_transfer_DRAFT prerequisite only. Synthetic class rankings/supports,
no MS-ASL gallery selection, native/cohort promotion, model calls or task quality.
Original metrics.py remains immutable and previously passed P001_v1 primary
ASLCitizen binaryDCG specification/reference parity. New evaluator adds explicit
micro and macro across all declared classes: require each class query support,
complete one-label uniqueID joins and fullclass permutations; reject missing IDs,
classes, duplicate predictions, NaN/Inf/ties ambiguous without declared rule.
DCG1/log2(rank+1), MRR1/rank, TopK rank<=K; K>=C gives1. Micro averages queries,
macro averages eachclass equally. Local tie = descending score/ascendingclass.
No upstream MS-ASL pipeline/paper score parity claim; adapted class lookup only.

Seven fixture groups:9correct A/1wrong B independently known Top1micro.9/macro.5,
DCGmicro.9630929753571458/macro.8154648767857288; rank3=.5 with ID-order independent
joins; missingclass/prediction/duplicate/extra rejects; deterministictie/nonfinite/
badpermutation; samplerorder independence/perclass membership/seeds; train-query
signer/sourceparent leakage rejects; no silently discarded incomplete/native-short
row, missingclass/sampleID or invalidseed. Sampling rule fixed: perseed PCG64
numpy.default_rng, ascending classIDs, uniforminteger in sorted train sampleID
pool, oneexample/class, exactsharedsupports for allmethods. Three seeds1701/2/3,
one fixed trial/no selection. This sampler accepts an explicitly prevalidated
eligible cohort, never selects it itself or retries a leaked gallery.

Register CPU120s/GPU0/2MiB fullbounded context/snapshots/input SHA/log/report/env/
status/error/manifest. CPU/RAM minimal synthetic work while native audit live;
one BLASthread, no source/model/data/encoder imports. Primary softwaregate allseven
groups pass. Scientific protocol stillheld by full3425 native incomplete rows and
GPUfeature/cost. Quality protocol will separately bind fullcohort/hash/supportIDs/
metric/scorer fixtures before actual ranking. Extra1313base labels/pretraining
unknown remain disclosed; no taskgain/newmethod/independent scientific claim.
