# MSASL_ESTCLK_v1 — adapted estimated-clock MS-ASL input availability

Prospective quality-blind adaptedprotocol, full3425 MSCOHORT003 train/val IDs in
originalorder/vocabulary0..99. Officialannotations fixed; existing metadata/source
exclusions retained. INPUT gate only, gallery and quality remain unselected.

Everycandidate clock: pinned ffprobe best_effort_timestamp integer ticks in
actualstream timebase minus stream.start_pts, halfopen annotationseconds. This
is a heuristic estimatedpresentation clock, not recoveredtruth/sourceauthenticity
or officialbenchmark reproduction. Retain rawticks/changes and originalstatus.
Rawframe order unchanged. Native16/stride2, yxyx bodycrop on decodeddimensions,
floorleft/top and ceilright/bottom, OpenCVbilinear224RGB24, unchangedSignRep
normalization later; shortnative inputs explicitly unavailable withoutpadding.

Affected38: rerunraw/bestreference identical priorD065; selectedbest ordinals must
be contiguous/strictclockincreasing, everypreclock<start/everypostclock>=end.
Require decodedrawprefix EXACT throughlastselected and independent selectedRGB
hashes matching bound FFmpeg D065. Allow outside interval disorder/suffixclock
disagreement prospectively, retainold WHOLE-prefix failedgate. Newmanualcropbounds,
allresizedRGB and firstnative16hashes, raw/bestselectedticks/hashes/frame/window
counts retained. No sort/dedup/resample/replace/drop selectedframe or relax selected
correspondence. Sourcechanges/mismatch/newincomplete holds fullcohort gate.

Unaffected3387: inherit bound MSNATIVECOVER001 successfulperrow raw==best ENTIRE
reference condition/exactselectedrawcounts/fullcropRGB hashes, inclthree short
13/15/10. Newuniform estimatedclock exactlysame for them. AllsourceparentoldSHA
identities linked, unchangedbyte/mtime statcheck; sixaffected sourcesrehashed.
No newclaim allsourcesrehashed, linguisticbox correctness or exacttrueboundaries.

Fullinput cohort gate: every3425ID exactlyonce, no incompleterows, all100classes
eligible in train+val. Failures retained/no qualityselected exclusions. Output
validationorigin/priorstatus/split/class/signer/source/annotationjoins exact.
No labelsfit/head/evaluator/model/GPUcalls. Existingpretrainingoverlapunknown,
official signer IDs/sourceparents separated by existingmetadata gate, physical
identity not independentlycertified. Parentmixing inside same split disclosed.

MSESTTEST001 pass prerequisite. CPU600s/GPU0, one decoder/OpenCVthread, source
512MiB/selected1024/decode4096/1MiBreferences30s/RAM256MiB+2GiBreserve; whole20MiB
cap incl12MiBcohort+4MiBchanged+512KiBsourcechecks+128KiBsummary/context/snapshots/
status/log/manifests. No storedRGB/video. Binding/prospectivecount/resource D066.
Any futurequality must use freshH006actualprotocol and exactsharedgalleries,
controls/scorer/uncertainty/resourceallocation; oldH006_v3DRAFT remains immutable.
