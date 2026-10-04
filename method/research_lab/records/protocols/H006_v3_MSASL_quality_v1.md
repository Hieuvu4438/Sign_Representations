# H006_v3_MSASL_quality_v1 — fixed geometry transfer, independent-corpus validation

Prospective adapted protocol locked before fullfeature access; originalH006_v3
DRAFT remains immutable. RQ: do within-class covariance directions estimated
from labeled ASL Citizen retain advantage over total-covariance directions on
separately collected MS-ASL dictionary lookup, with fixed matched spectra?
Known standard whitening/analysis hypothesis, no newalgorithm or linguistic
causal invariance claim. SameASL collection-domain transfer; lexeme meaning/base
classes may overlap, encoderpretraining unknown, no encoder-unseen/crosslanguage.

Input cohort MSASL_ESTCLK_v1/3425candidate availability:2674eligibletrain/748val,
100classes0..99,3native-short13/15/10 recorded withoutpadding. Newestimatedclock
explicit/train_11136 membership59->60 beforequality, oldrawclock failure retained.
SharedMSDICT001 galleryIDs seed1701/1702/1703 oneeligibletrainvideo/class, all748
valqueries;289unique supports/1037featureclips, sources/signerIDs train-val disjoint.
171queryparents nested19officialsigner IDs, physicalidentity notcertified. No
reselection/alternateclasses/seeds or changingquerydenominator afterscores.

Strict frozenSignRep sameSHA/upstream/mask0/eval/no-grad/FP32/native16stride2/all
windows/bodycrop/normalization. Original clipped-max activity FP64pool/L2, extra
geometry fit1313fullylabeled ASLCitizenclips/100classes disclosed in addition to
100targetgallerylabels/seed. No target refit/mean/PCA/threshold/layer/rank/lambda
selection or training. Original AXISTRANSFER001 savedR/Tbases, ascendingR/Tspectra
and commonbase mean are the only geometry parameters; preserve arraySHAs before
scoring. Modes RR/TR/RT/TT use (U_direction/sqrt(spectrum))@U_direction.T then
L2 of (clipvector-basemean)@transform. Strongcontrols rawclipcosine and base-mean
centeredL2cosine. No recomputing eigensystems or selecting secondarywinner.

Primary: mean overthree fixedgalleries of query-MICRO binaryDCG1/log2(rank+1),
effect RR minus TR. Practicalgate>=.005 and eachgalleryeffect>=0, both fixed;
failure kills scaling fixed orientation-transfer explanation, not all covariance
methods. Secondary allfourcells/raw/centered mean andpergallery DCG/MRR/Top1/Top5
micro andmacro overall100 supportedclasses; recognition derivedfromsame one-shot
rankings, not separately trainedclassifier or officialMS-ASL leaderboard. No
inference acrossmultiplesecondary tests. Exact100class rankings perquery; finite
FP64scores and deterministicascendingclassID ties (metrics.ranked_classes),
report actualties/min scoregaps/numericalstability ratherthan silentlyperturbing.
Independent original-space metric replay/rank/metric recount before promotion.

Uncertainty primarypaired perquery RR-TR differences averaged overfixedgalleries;
2000bootstrap/seed1701 resample19signerIDs withreplacement, carryall clips ofeach
sampledsigner, recomputemicro ratio eachreplicate. Conditional onfixedsupports/
encoder/collections and baseparameters, not independentseed/population/selection-
adjusted confirmation. Reportpergalleryeffects andconditional95%interval, no
extra primary significancecriterion or supportbootstrap claim. All val labels
used only for relevance/evaluator and grouped reporting, nevermodel/geometryfit.

No test access: independent-corpus MS-ASL validation task, same-corpusdevreuse
limitations remain in priorASLCitizen evidence. Crosscorpus labelmapping unnecessary
because dictionaryqueries/supports use sameMS-ASL ontology; no verifiednewlexeme.
Seven technicalcostclips mayreuse validpooledvectors, no taskscore previouslyread.
Fullfeature extractioncap/source/input/count/shard bindings recordedMSFULLFEAT001.
Evaluation implementation, scorer/source/basearraySHAs, exact fullpredictions/
metric/rank schemas, budget/artifactplan/config/run must be registered before
FIRSTscoring; fixtures testedwithouttargetquality. No scientificpromotion from
native/feature/software compatibility. Closestliterature/contribution review and
honestresource/failure/pretraining disclosures needed before anypaperclaim.
