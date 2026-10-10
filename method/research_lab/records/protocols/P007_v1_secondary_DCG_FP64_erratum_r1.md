# P007_v1.secondary_DCG_FP64_r1 — explicit reporting precision erratum

ParentP007_v1/H007_v1/REJECTEVAL001 unchanged. Only affected reportedfield
metric_rows[].known_retrieval.DCG uses implicitNumPyfloat32 fromint16log2.
Original fullmetric report/source/config/rank/prediction/score/threshold/bootstrap
artifacts retained unchanged. Not newexperiment/model/primary/threshold/policy.

REJECTVERIFY002 preregisteredCPU300s/GPU0/full20MiB, exactinputs fromcompleted
measurement and failedREJECTVERIFY001 trace/status/config/source. Verify
originalFP32DCG numericfield by declared backendexpression withint16rank inputs;
independent correctedFP64DCG=math.fsum(1/math.log2(int(rank)+1))/Nknown.
Do NOT loosen1e−12 scalar/metricchecks for otherfields or silentlyoverwriteDCG.
Save dcg_precision_erratum.json with216rowkeys/original/replayedFP32/correctedFP64/
differences/maxerror/version so all tables have traceable reporting choice.
Subsequentnewmeasurement must use properFP64discount undernewprotocolversion;
this verifier only supplies explicit corrected presentation onfixedoriginalranks.

Continue original independent sourcepool/einsumcosine parity,saved-logit scalar
score/rank/known-onlyorderstatistics/fullcounts/curves/retrievalothermetrics/
source-targetbootstrap/primarygate/pairedpersistentunknown checks. Add exact
pointthreshold/closedknownaccuracy and calibration empiricalacceptance/tie
counts toclose omitted metadatafields. OriginalDCG considered precision-limited
and superseded incorrectedpresentation, not a reason to alter primaryscreen.
No task test/HPO/target fit/extraction/scientificmethod comparison chosen from
outcomes; outputfailure preserved with no completedstore overwrite.
