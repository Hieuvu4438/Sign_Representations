# OSCRADAPTTEST001 — adapted joint FAR/CCR specification v1

Software diagnostic only, CPU60s/GPU0/HPO0/seed1701/complete2MiB. Does not read
numeric signfeatures, fit threshold, extract or evaluate H007 quality/test data.
Manual specs fixed before run; tolerance1e−12. Source D078/referencefixture proof
and primary selected OSCR definitions bound. No exactofficial-comparable claim.

Inputs: distinct nonempty string sampleIDs exactly matching expected ordered
cohortIDs; groundtruth global nonnegative integerclass labels; predictions each
a class in the nonempty distinct nonnegative integer gallery; finite1D numeric
acceptance scores. Boolean/fractional/negative classIDs disallowed. Exact counts
and equal vectorlengths; no missing/duplicates/emptycohort/nonfinite scores.
Known label is memberofgallery, unknown outsidegallery; no remapping by column
index. Both known and unknown must be present for joint evaluator; fail instead
of dropping/no-denominator NaN. Gallery may have knownclasses without queries;
this software test doesn't assert scientific macroclass coverage.

Threshold numeric scalar, not Boolean/NaN; ±infinity allowed for complete
accept/reject limits, serialized as "+inf"/"-inf" in JSON point records.
Accept score>=threshold for bothgroups. Counts:
knownacceptedKA, knowncorrectacceptedKCA, unknownacceptedUA; NknownNK,NunknownNU.
FAR=UA/NU, CCR=KCA/NK, knownacceptance=KA/NK, closedknownaccuracy=Kcorrect/NK.
Record integer numerators/denominators with every metric. Correctness compares
predicted globalclass to trueclass; accepting misclassified known NEVER addsCCR.

Curve: threshold+infinity withzeroaccepted, then all distinctfinite scores in
descending order, tiedrows grouped. Retain duplicateFAR values (vertical
segments), never reorder curve by FPRties independently. LastFAR1,
CCR=closedknownaccuracy; no fabricated endpoint. SecondaryAUC=sum adjacent
deltaFAR*(CCRleft+CCRright)/2. This grouped-trapezoid/inclusive-tie scalar is
explicitly adapted; diagonal tie segments interpolate randomized acceptance,
discrete deterministic operating points remain primary evidence. Not a new
metric contribution or claimed parity with original strictknownthreshold code.

Manualareas, two known/two unknown, gallery[0,1], eachprediction0:

| fixture | true labels | scores | area |
| --- | --- | --- | ---: |
| wrongknown | [1,1,2,2] | [.8,.8,.2,.2] | 0 |
| perfect | [0,0,2,2] | [.8,.8,.2,.2] | 1 |
| reversed | [0,0,2,2] | [.2,.2,.8,.8] | 0 |
| alltied | [0,0,2,2] | [.9,.9,.9,.9] | .5 |
| halfcorrect | [0,1,2,2] | [.9,.7,.8,.6] | .5 |

Halfcorrect thresholds/count tuples(KA,KCA,UA):+inf(0,0,0),.9(1,1,0),
.8(1,1,1),.7(2,1,1),.6(2,1,2),−inf(2,1,2). Eachdenominator2.
Alltied at.9 acceptsall/curve(0,0)→(1,1) area.5; .9001 none.
Halfcorrect finalFAR1/CCR.5/knownacceptance1; closedaccuracy.5. Perfect at.8
acceptsbothknown/rejectsbothunknown, proving inclusion ties explicitly.
Permutation ofrows with expectedcohort permutedtogether must leave countcurve/
area unchanged; unsynchronized ordering fails. Empty/missing/duplicate/invalid
ID/class/gallery/dimension/nonfinite/group/threshold fixtures expect ValueError.

Independent reference: Python loops/counts over fixed small fixtures plus
30deterministic seed1701 synthetic cases(scoreintegers−2..2, exactties). Fraction
rational arithmetic and explicit threshold counting verify each countpoint and
trapezoidarea versus NumPy curve. Synthetic case count not HPO/trial search.
Six groups: manualareas; manualcounts/endpoints; tieandpermutation; schemaerror
guards; thresholderrorguards; independentlycounted30cases. Save allcases/counts
and groupreport inadapted_tests.json. Successfulchecks validate only this fixed
adapted scorer specification; no published-score reproduction/taskquality,
thresholdgeneralization/method/contribution/freshconfirmation claim.
