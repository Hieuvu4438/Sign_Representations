# HEADANALYSIS001 — separate saved-head recount and locked utility screen

2026-10-05, D089. Implements analysis already specified in P007_v2/P007_v3 after
all three initialization jobs complete. No new data selection, parameter
learning, target calibration or change to primary thresholds/falsifiers.

Material Passport: current-stage exploratory experiment audit; own new analysis
code; parents HEADPILOT001/002/003, HEADPAIRPOOL001, REJECTEVAL001 and accepted
REJECTVERIFY002; original native features/checkpoint and all source manifests
preserved. No external content transfer or human-data study.

Repool all1027 source vectors from SHA-checked native window caches, manual
activity clipping/max and weighted mean/unit norm; tolerance1e-12. Verify all
16000 paired queries' positive membership/51 unique candidate classes/no source
or signer match; independently derive correctly recognized positives and1000
update/count flags, compare every seed. This is saved-window replay, not raw
RGB redecoding or optimizer training reproduction.

Load explicit flat2411650 FP32 weights/schema and execute NumPy FP64 affine/
ReLU layers; residual subtraction remains FP32(query)-FP32(nearest support).
Use unchanged saved cosine predictions to select supports. Combined tolerance
abs(CPU-GPU)<=1e-4+1e-4*abs(savedGPU). Use original saved GPU logits for ALL
threshold/count/utility decisions, irrespective of numerical replay proximity.

Manual highest source-known threshold accepts ceil(19*N/20), >=inclusive ties;
18 calibration cells/108 metric rows per initialization. Independently count
known/unknown/correct-known accepts and FP64 rank-based retrieval. Group equal
scores and cumulative integer counts for162 full head curves with reject-all
endpoint; grouped trapezoid tolerance1e-12, no official OSCR scalar parity claim.
Recount216 old control rows, retain original FP32 DCG fields; new exports use
separately recomputed DCG_FP64. Export540 rows/108 equally averaged cells.

Primary conjunction UNCHANGED: MLP-minus-maxcos at50/FROZEN50, equal mean3train
by3support; MS CCRgain>=.02/all3initialization gainspositive/conditional95%lower>0;
MS FAR meanincrease<=.02/conditional95%upper<=.02; base CCRloss<=.02. Original
2000 MS/base signer multiplicity arrays reused across every cell/head/control;
equal mean per-cell ratios, no pooled denominator or dropped invalid draws.
Validated cluster_mean_rate cross-checked by separate per-group integer recount
for all2000 draws. Invaliddraws => CIunavailable/no screen pass. Report three
initialization support means explicitly, conditional bootstrap scope.

CPU300s/GPU0/10MiB total registered artifact/header/config/source/log/status
replacement plan; tmpfs512MiB+2GiBreserve. Packed head curves77000x4FP64, prefix
length and162segments, unusedtailzero. Bootstrap2000x3 FP64delta/Boolvalid;
columns MSCCR/MSFAR/baseCCR. Output per_seed.csv/mean.csv/head_analysis.json.
Manual5group oracles: wrong-known/inclusive ties, grouped AUC, known95 threshold,
tieovershoot and affine forward. Preserve failures; no silent retry/mutation.

Passing audit is software/numerical validity, independent of scientific screen.
Screen failure => kill scaling this fixed recipe, no same-MS rescue HPO. Even
pass demonstrates only existing control utility with1027 extra fit+286cal labels,
not novelty/full jointly trained FR-Disc/classifier/ranking/encoder improvement.
Both corpora ASL, exposed MSval exploratory; physicalsigner/session/pretraining/
crop/corpus confounds and genuinely fresh confirmation unresolved.
