# REJECTSCORETEST001 — fixed score/calibration software oracles

CPU60s/GPU0/HPO0/seed1701/full2MiB; referencecapture5648B<64KiB. Exact fixed
authorutils/MITlicense hashes D080, AST NAC/cosine only, CPU existingTorch,
no module/training/data/environment install. Floatconfidence/cosineatol1e−12
using float64; non-tie prediction globalIDs exact. NumPy/Torch versions saved.

For cosine logits [ln2,0,−1] atglobalclassIDs[7,11,18]: prediction7,
maxcosln2, marginln2, MSP=2/(3+exp(−1)), NACk2=2/3.
For [0,0,−1]: lexicalprediction7, maxcos0, margin0, MSP=1/(2+exp(−1)),
NACk2=.5. Mathematically no parameter fitting here; actualstudyfixedk16/scale1.
Four rejection scores share rawcosine prediction/rank; do not treat margin/
softmax/NAC as trained models or new methods. Unitvectorcosine oracle axis
query[1,0]/[0,1] and prototypes[1,0]/[0,1]/[−1,0] gives [[1,0,−1],[0,1,0]].
Compare auditedTorchcosine and NumPy exact/manual. AuthorNAC compare30fixed
seed1701 synthetic no-tie matrices, gallerysizes10/25/50, k=min(16,N), scale1.
Compare confidence and prediction fornon-ties. Tiedfixture compares confidence
only; ownglobal-ID lexicalpredictionmanual. Upstreamdefaultk16 onN10 explicitly
RuntimeError, ownedclamp result equals MSP there; no silentparity claim.

Known-only empiricalcalibration: numerator19/denominator20, ceil integer rank,
highest threshold satisfying empiricalknownacceptance with inclusive>=ties.
Nknown20/scores1..20 ->threshold2/19accepted/.95;
scores[1,1,2]/target2/3 ->threshold1/3accepted/tieovershoot;
Nknown1 ->its score/allaccepted. Changing excludedunknownscores from−100 to100
must leave threshold/knowncounts unchanged. Permuting rows withknownmask same
leaves fit unchanged. No accepted-as-correct substitution or unknownrisk promise.
Empty/no-known/nonfinite/maskshape/nonboolean/invalidfraction errors ValueError.

Scoreguards: no empty/dimension/nonfinite matrix, wrong/unsorted/duplicate/
negative/boolean classIDs, fewerthan2galleryclasses, invalidk/scale. Ranking is
stable descendingcosine withglobalclassesascending for exactties. Registered
groups: manualformulas/cosine;30authorCPUparities; lowgallery/tiecontrol;
empiricalquantile/manual/excludedunknown/permutation; invalidinputs. Save
source/license/version/parity/manualvalues inrejection_score_tests.json. Software
oracles only; no sign feature/scientificscore, method or officialfacebenchmark.
