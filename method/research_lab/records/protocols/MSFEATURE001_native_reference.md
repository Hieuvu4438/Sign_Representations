# MSFEATURE001 — Four-window checkpoint reference and GPU smoke

PrelaunchD049. FixedfourMSNATIVE002 first16bodycropped nativewindows/PTS and
MSPREPROC001 normalizedinputSHA; no quality/label-based selection/test access.
CheckpointSignRep SHA f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b,
upstream06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2, strictfullstateload/evalmode/
mask_ratio0/no-grad/FP32/TF32false. Existingvenv only; no upstream/shared edits.

MSFEATURECPU001 reference CPU120s/GPU0/2MiB. Wrapperpreprocess exactmatches
read-only upstreamTransformation and previousnormalizedSHA; infer_windows
length16/stride2/batch1 vs direct model(TCHW→BCTHW), everyoutput features768/
latent768/cls_active2 finiteFP32 and bitwise same. One warmup, three timedwrapper
calls plus three directcalls perwindow=28calls total/fourwindows. Exactrepeat
required; no testdriventolerance widening. Store4×dimoutputs/12×dimrepeats,
timings/preprocessing/inputSHA/load/env/source/config/log/status/manifest.

MSFEATUREGPU001 prospective CPUjobdeadline120s/GPU120s cumulativeH006smoke
ceiling120s within program14400s,2MiB durable. Config registered only after
completedCPUreference and passedGPUCTXTEST001. Recheckactualthree-root effective
ledger+SHA-boundGLAUDIT001erratum; no unresolvedownedGPU. PhysicalGPU0 granted
UUIDGPU-3775c606-49da-34fe-322c-756749074f5b, min8GiBfree/maxutilization10,
8GiBallocatorcap, preserve schedulerCUDAvisibility/allotherprocesses. GPUstart
persisted beforeconfigure/modelmove, conservativewholeallocatedtimecharged on
completion/failure/interrupt. Fresh boundedGPUcontext, completefileplan.

GPUdirect-wrapper/repeatedbitwise checks identical. CPU_GPU comparison allthree
outputmatrix np.allclose(atol1e-5,rtol1e-4), fixedbeforeCPU/GPUresults, all outputs
reported regardlesspass. Failure gate persists report and failedstatus, no
tolerance/rank/seed/source rescue. No science/performance inference fromfeatures.
Timings synchronized before/afterGPUwrapper, warmupseparate; includesinputtransfer/
outputcopy, excludesdecode/preprocessing; no end-to-end or corpuscostclaim.
PeakGPUallocated/reservedlog. Scratchplanning256MiB+2GiBexistingworkspace reserve,
notmeasuredRSS. Sourceinputs stored/read-only; no extraction orvideo download.
Rootbulk20%reservefails; bounded2MiBrecord exception. ExactNPYheaders/filecaps/
allsource/upstreamsnapshot/config/env/log/report/manifest/rewriteplanbeforelaunch.

Sameagentindependentinterfacepath, samecheckpoint/model, not externalscientific
review. Successfulsmoke clears fourtechnicalfeaturecontract only; wholeadapted
cohort/evaluator/scientificutility/novelty stillopen. BusyGPU: CPUindependentwork,
no failedGPUattempt launched merely to discover alreadyknownresourcecondition.
