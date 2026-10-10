# H007 evidence dossier r2 — fixed supervised residual controls

2026-10-05, [D090](../decisions/D090.md). **NOT PAPER_READY**; no novel method
or promoted recipe. Exploratory evidence extends the immutable
[r1 diagnostic](H007_EVIDENCE_DOSSIER_r1.md); original files/results retained.

## Material Passport

Own new frozen-feature adaptation and numerical audit, selected pinned primary
FRDisc source. Source cards S47/S48/S49 and bound S48_METHOD_r1/NOVELTY memo,
P007_v2/P007_v3, three preregistered initialization runs, separate analysis and
descriptive export. No original-paper score reproduction, full-paper read,
linguistic annotation/Deaf-user study, independent scientific review or submission.

## Question, task and supervision

Can an existing supervised query-minus-nearest-support discriminator improve
source-calibrated correct recognition and rejection together on a separate
collection? Frozen SignRep isolated one-shot dictionary retrieval and
dictionary-absent rejection; dictionaries10/25/50, three fixed supports seeds.
Unknown means class absent from the selected dictionary, not a certified novel
sign form absent from encoder pretraining. Both corpora are ASL. Continuous
recognition, translation and cross-language transfer remain outside this protocol.

Sourcefit1027clips/100classes/24officialsigners, calibration286/8, base378/5,
MS-ASL748validation/19. Fit/cal/dev IDs/signers/sourcebytegroups disjoint; encoder
and ranking frozen. Heads receive extra1027 fit labels plus286 calibration labels.
One-shot describes dictionary supports, not the whole supervision budget. Source
classes exposed to fitting; no unseen-source-class claim or target fitting.

## Tested system and protocol deviations

Same source query is paired with two50class dictionaries sharing49 supports,
one adds its class, one substitutes another class.1000episodes/16000pairs fixed
before fitting, signer-uniform thenclip-uniform PCG64seed1701. All24signers/P38
retained; original P007_v2 two-class-per-signer law actually stopped at P38 and
its partial run remains. Sampling adaptation was registered before head quality.

Two existing signed-residual heads: linear768→1/769parameters and author SAFSAR
MLP768→1536→768→64→1/2410881parameters. Frozen FP64cosine selects nearest support;
residual is FP32(query)-FP32(support). Heads use raw logits/stableBCE, correct-known
positive selection and paired unknown counterparts, Adamlr.001/weightdecay.0001,
1000finalupdates, seeds1701/1702/1703. No HPO, earlystop, best checkpoint, CE or
encoder gradients. Joint author backbone/CE+BCE training and full FRDisc method
remain untested. This adaptation and sampler control are not a new method.

Source-known calibration highest observed threshold acceptsceil(19*Nknown/20),
inclusive>=ties, unknown/target labels unused. FROZEN50 and MATCHED_SOURCE_SIZE,
both identical at50. Known95% source acceptance does not guarantee CCR95%, heldout
known acceptance or unknown FAR. Existing maxcos/margin/MSP/NAC16 controls retained.
Adapted grouped-tie FAR/CCR/AUC, no official scalar OSCR parity or leaderboard claim.

## Observed results

Equal support-cell mean at50/FROZEN50; heads average3initializationsx3supports,
fixed scores average3supports. Percent units. No best-seed selection.

| Score | Base CCR | Base FAR | MS CCR | MS FAR |
| --- | ---: | ---: | ---: | ---: |
| maxcosine | 66.25 | 68.80 | 46.66 | 49.83 |
| margin | 65.90 | 86.69 | 50.93 | 87.94 |
| maxsoftmax | 65.90 | 74.04 | 49.51 | 75.86 |
| NAC16 | 66.25 | 76.68 | 50.14 | 79.43 |
| residual linear | 63.53 | 82.78 | 50.04 | 92.70 |
| frozen FRDisc MLP | 67.20 | 72.62 | 50.65 | 77.71 |

The registered MLP-minus-maxcos screen **FAILS**: MS CCRgain+3.9879347pp,
conditional95%[2.0218957,5.9561677]pp; all3initialization support means positive
(+3.8848,+4.0778,+4.0013pp). MS FARincrease+27.8800957pp,
[18.4748893,40.6524082]pp exceeds both2ppmean and conditionalupper limits.
BaseCCR+0.9453104pp passes floor. Four of six conditions pass, which does not
pass their conjunction. Source95% calibration selects a much more permissive
target operating point here: target knownacceptance74.36→91.93%.

Bootstrap2000original fixed official signer multiplicity arrays, equal mean
per-cell ratios, invalid0. Conditional on source calibration, parameters and
galleries; not training/support/calibration uncertainty or physical signer proof.
All3initialization means positive does not mean everygallery/signerslice positive;
MS CCRcellgains positive6/9, FARincreases positive9/9. BaseCCR/FARpositive6/9.
Denominators at50: baseknown187–188/unknown190–191; MSknown340–382/
unknown366–408 dependinggallery. No deployment prevalence/precision guarantee.

All [540per-cell rows](../../../frozen_probe/runtime/HEADANALYSIS001_20261005_ba5a2a86_s1701/per_seed.csv)
and [108equal means](../../../frozen_probe/runtime/HEADANALYSIS001_20261005_ba5a2a86_s1701/mean.csv),
[full analysis](../../../frozen_probe/runtime/HEADANALYSIS001_20261005_ba5a2a86_s1701/head_analysis.json).
Secondary linear/score/size/AUC comparisons descriptive, no equivalence/winner.
Rawcosine closedTop1 remains67.4925%base/52.2412%MS for everyhead/score.

## Numerical audit, figures and cases

Separate NumPyCPU affine/ReLU replay from explicit saved flatweights/schema:
76248logits, maxerror1.8392557450397362e-6 within prospective combined1e-4.
Decisions use original savedGPU logits, not approximateCPU replay. Source1027
repooling/39880windows/all16000paired memberships/1000counts,54calibrationcells,
324headmetricrows+216oldcontrol rows/FP64DCG,162curves/76409points and2000signer
bootstrap recount pass. No rawRGB/optimizer-training replay or exact threshold
equivalence acrossdevices. Original FP32 DCG erratum remains in r1 untouched.

[PNG](../../../frozen_probe/runtime/HEADREPORT001_20261005_6ff0120a_s1701/joint_head_curves.png)/
[PDF](../../../frozen_probe/runtime/HEADREPORT001_20261005_6ff0120a_s1701/joint_head_curves.pdf)
visually inspected:60thin individual curves/12equalcellmean operating points,
FAR/CCRaxes0..1/noaveragedcurve/CIbands. Different head and frozen-score supervision
explicit in caption. Posthoc [numerical disagreements](../../../frozen_probe/runtime/HEADREPORT001_20261005_6ff0120a_s1701/numerical_disagreements.json)
retain10buckets/20uniformselectedexamples atinit1701/support1701. MS28correctknown
recovered/3lost,143unknown additionally accepted/13newly rejected. No visual,
linguistic or human groundtruth interpretation or target-based rescue design.

## Resources, limits and decision

[Cycle closure](../inventory/H007_HEAD_CYCLE_BINDING_VERIFICATION_r1.json)
checks2857currentbindings/10originalcaps/9completestores+1realmetadatafailure,
exportmeans/bootstrap/cases. Runtime46129167logicalbytes; summedcapturedwall
84.9264439237304s; chargedGPU69.3861866821535s includingsourcecost. Pinned public
model/train/loader/config/license61837B separate. Program7592.469239038299/14400s,
H00769.3861866821535/420;173terminalstatuses/noownedlivejob. GPU contended,
cost is head+savedfeature pipeline, not full rawvideo/inference benchmark.

MSval cumulativeexploration, physicalsigner/session/pretraining/ontology/input
crop/corpusdifficulty confounds remain. Prior MSMETA002 metadata4172testrows/
2847rawcandidates/2train-testparents is not native100class testcoverage or quality;
no fresh confirmation run occurred. Extra model capacity/source labels make
comparison a declared supervised regime, not equal-label frozen score contest.

D090: kill scaling this fixed recipe/no same-MS rescue tuning. Continue full
program at DISCOVER, closest full-method audit and train-only mechanistic
falsifier before any new H/P. Existing-control failure is useful evidence;
contribution/novelty/fresh confirmation/reproducible paper package remain open.
