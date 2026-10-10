# P007_v1 — frozen finite-gallery rejection diagnostic, before firstscores

Task/regime: adapted exploratory isolatedsign one-shot dictionaries with rejection.
Unmodified frozen SignRep/native16stride2/full_activity FP64pool/L2, ASLCitizen
fullframe vs MS_ESTCLK_v1bodycrop input. Existingapprovedfeatures only, no model/
fine-tune/head/geometryfit/newencoder/GPU/test/ontologyalignment. No official
benchmark/encoder-unseen/languagevalidity/causalindependentdomain claim.
MSval already exposedD072, base reuseddev; freshconfirmation unknown.

Cohort/support/source/calibration membership EXACT from terminalOSMETA001/
independentlyverified27cells/3389metadata rows. Sourcebase100classes train1313:
cal286/8signers vs pool1027/24, dev378/5. Three fixedsupportseeds1701/1702/1703,
onevideo/class, nested10/25/50 according frozenclassorder. MS748val/19official
IDs, existingMSDICT001supports289union full100classes; dictionary50 subsets.
No reselection/shrink/failedtargetpool branch repair. Base805numericfeatureclips
only (286cal+378dev+141uniongallery50supports); entirebasecache integrity hashed,
unusedpoolclip numericarrays not loaded. MScomplete1037vectors verified/reused.
Sample/source/signer joins, labelmap global0..99 percorpus, strictIDs allqueries.
CrosscorpusnumericclassIDs not linguisticmapping; featureinput differences
confound causal attribution. SignerIDs officialmetadata, physicalidentity and
session/pretrainingmembership unknown.

Four scoredefinitions: z_i=clipped[−1,1] cosine ofL2normalizedquery/support,
ascendingglobalgalleryclassIDs columns. Stabledescendingz/lowestclassID exact
ties, common prediction argmaxz, sharedrawcosine ranking. Maxcos=maxz;
margin=z_top1−z_top2; MSP=exp(z_top1)/sum_allexp(z); NAC16=exp(z_top1)/
sum_topmin(16,N)exp(z). Scale1/temperature1 fixed, no hyperparametersearch.
ScoreNACcomponent author971f6767/knownadaptation kclamp/tiepolicy, not fullface
BNfine-tuning reproduction. Higherallfour meansacceptance. No newmethodclaim.

Two policies, frozen BEFORE any rejectionquality:
FROZEN50: for eachseed/score, fit onsourcecal dictionary-present rows at50 only.
MATCHED_SOURCE_SIZE: similarlyfit onknownsourcecal rows at eachN10/25/50.
Threshold is highest observedscore accepting≥ceil(19*Nknown/20), inclusive>=;
integerceil, tiedscores mayovershoot. No correctnessfilter; unknowncalrows and
allsource/targetdev scores do not fit. Bothpolicies identical atN50 bydefinition.
No calibration refit inbootstrap. Empiricalknownacceptance target95%, not95%CCR
or unknownFAR/population/confidencecalibration guarantee. Moretrainlabels beyond
one-shots explicitly286cal and fullsourcepool labelavailability; no targetfit.

Metrics everyrole/seed/N/score/policy: count-based FAR/CCR/knownacceptance,
closedknownTop1, knownTop5/DCG/MRR onsharedcosineranks, full known/unknown N.
Unknowntrue-rank stored0 and excluded fromknownretrieval denominator explicitly.
Also commonknownfirst10 accuracy/CCR/acceptance and persistentunknownlast50 FAR
for pairedgallerysize comparison. Never drop unknownqueries from open-set rates.
Fulljointcurves withalreadyvalidated adaptedinclusivegroupedties/secondaryAUC,
lastFAR1/CCRclosedaccuracy. Primary is operating-point FARdrift, not detection
AUROC. Allseeds/perseedpoints/equal3seed means, no selectedbestseed/method.

PRIMARY: FROZEN50 maxcos gallery50 meanseed(MSvalFAR−baseASLdevFAR).
Materialdrift screen absolutevalue≥.05 AND all3perseed signed changes have same
sign AND conditional95%bootstrapCI entirely>0 or<0. Negative sign describes
lower targetFAR, not automatically harmfulfailure. Kill scaling premise if
screenfails; no sameMSval tuning. Pointdelta/CI/gate/knownCCR/coverage reported
evenwhen screenfails. Other score/policy/galleries/metrics exploratory; no
unadjustedsecondary significance/superiority/equivalence promotion.

Uncertainty: conditional clusterbootstrap2000fixedmultinomialdraws seed1701.
Sortedbase5devsigners draw first, sortedMS19 draw second fromsamePCG64 rng;
counts total#groups, same drawmatrix reused acrossall methods/seeds/N/policies
within eachcorpus. Perseed unknown counts/rates recomputed inweightedquery
cohort, thenequal3seed mean; no poolingseed queries asindependent. Corpora
resampledindependently forprimarydifference. Thresholds/galleries/calibration
fixed; no calibration/training/pretraining/population uncertainty covered.
Anyzero-denominator replicate explicitlyretainedinvalid; no dropping/retry/
nanmean, CIunavailable and no primarygatepass. Pairedpersistentunknown gallery
50−10 changes withinMS/base use sameweights/fixedabsentquerysets; moving full
unknownrates also reported but cannot isolate membership/composition effects.

Artifacts/regime: CPU300s/GPU0/20MiB fullboundedruntime,512MiB RAMworkspace.
Base805x768FP64pooledvectors/IDs, cosine_seed50[3,1412,50],
acceptance_scores[3,3,4,1412], predictions/knownranks[3,3,1412]int16;
1412 orderedqueryrows=basecal286, basedev378, MSdev748. Features/source/config/
checkpoint parenthashbindings and source snapshots beforelaunch. Complete
compactcurves ≤50940x4FP64 [threshold,knownaccept,knowncorrectaccept,unknownaccept]
withsegmentstart/stop/role/seed/N/score/Nknown/Nunknown. Firstpoint +inf encoded
threshold0 plus explicit firstpoint-kind reject_all, zero counts; never a real
theta0 claim. Unusedtailzero with actualcursorreported. Arraysfinite, integer
counts exactly represented; no JSONtensofthousands repeatedmetricrecords.
Bootstrapdraw matrices/replicatedprimarydeltas/validmask saved; reports/metadata/
context/status/logs/manifest and replacementreserves all preregisteredbound.
No largefeaturecopy/oldstoreconversion or silentcapincrease. 20%bulkrootreserve
stillrestrictslargeoutput; existing boundedlightweightstore exception only.

Before measurement, REJECTSCORETEST001 fixedscore/calibration manual+authorCPU
gate PASSED; OSCRADAPTTEST001 jointmetricfixtures PASSED; REJECTANALTEST001
conditionalcluster ratio/seedmean/invaliddraw/compactcurve manualoracles must
pass. Then newexclusive config/run registration and source/currentfeatureSHA
checks, full805basepool/nativecontract/sourcejoins/MS1037shardchecks, complete
1412query prediction/scorecoverage. IndependentREJECTVERIFY001 directcosine/
scalar math.fsum scoring/orderstatistic/groundtruth rates/curve/primary recount
after run, ownwrappercode and config bound before verification; no tuning/re-score
variant/new scientifichypothesis. Technicalfailure retainsalloutputs and fresh
version mayrepairparseronly; no completedrun overwrite or empiricalrescue.
