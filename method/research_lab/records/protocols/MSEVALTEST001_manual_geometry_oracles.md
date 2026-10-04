# MSEVALTEST001 — synthetic fixed-geometry evaluation oracles

CPU60s/GPU0/2MiB; no realtargetfeatures, similarities, rankings or taskquality.
Fourgroups: manually normalized 2D RR/TR/RT/TT/raw/centered projections from known
axispermutations/scales; exact query/gallery union/classorder/train-val/source/
signer joins with invalidfixtures; all-tied complete rankings ascendingclassID,
micro2/3 versusmacro1/2 for unequalclasssupport; primary threegallery DCG and
unequalgroup bootstrap (Ahas two clips/Bone), manual flat allsampledclip oracle
versus forbidden averageofgroupmeans. No refitting/secondaryselection.
